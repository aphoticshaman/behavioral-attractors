"""Main experiment runner for Paper B."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from .utils import (
    load_yaml, load_text, save_jsonl, generate_run_id,
    create_manifest, SeededRNG, set_seed
)
from .providers import OllamaProvider, CloudProvider
from .tasks import (
    generate_t1_task, format_t1_prompt,
    generate_t2_question, format_t2_prompt, parse_t2_response,
    generate_telemetry, format_telemetry_packet, get_override_instruction
)
from .scoring import (
    score_state_adherence, score_t1_constraints, score_t2_answer,
    compute_calibration_metrics, compute_override_resistance,
    compute_contradiction_latency, compute_persistence
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the Paper B experiment."""

    def __init__(self, config_path: str, smoke: bool = False):
        """Initialize runner.

        Args:
            config_path: Path to experiment.yaml
            smoke: If True, use reduced trial count
        """
        self.config = load_yaml(config_path)
        self.models_config = load_yaml(
            str(Path(config_path).parent / "models.yaml")
        )
        self.smoke = smoke

        # Determine trial count
        trials_config = self.config.get("trials", {})
        self.n_trials = trials_config.get("n_smoke", 5) if smoke else trials_config.get("n_per_condition", 100)

        # Load system prompt
        prompts_dir = Path(config_path).parent.parent / "prompts"
        self.system_prompt = load_text(str(prompts_dir / "system_base.txt"))

        # Initialize providers
        self.providers = {}
        provider_configs = self.models_config.get("providers", {})

        self.providers["ollama"] = OllamaProvider(provider_configs.get("ollama", {}))
        self.providers["cloud"] = CloudProvider(provider_configs.get("cloud", {}))

        # Setup output paths
        output_config = self.config.get("output", {})
        self.data_dir = Path(output_config.get("data_dir", "data"))
        self.data_dir.mkdir(exist_ok=True)
        self.outputs_path = self.data_dir / output_config.get("outputs_file", "outputs.jsonl")
        self.manifest_path = self.data_dir / output_config.get("manifest_file", "manifest.json")

        # Run ID
        self.run_id = generate_run_id()

        # Seeds
        seeds = self.config.get("seeds", {})
        self.master_seed = seeds.get("master_seed", 42)
        self.task_seed_offset = seeds.get("task_seed_offset", 1000)
        self.telemetry_seed_offset = seeds.get("telemetry_seed_offset", 2000)

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models based on provider health."""
        available = []
        models = self.models_config.get("models", [])

        for model in models:
            provider_name = model.get("provider")
            provider = self.providers.get(provider_name)

            if provider is None:
                logger.warning(f"Unknown provider {provider_name} for model {model['name']}")
                continue

            if provider_name == "cloud" and not provider.available:
                logger.info(f"Skipping cloud model {model['name']} (not configured)")
                continue

            if provider_name == "ollama":
                if not provider.health_check():
                    logger.warning(f"Ollama not available, skipping {model['name']}")
                    continue
                # Check if model is actually available
                ollama_models = provider.list_models()
                if model["name"] not in ollama_models:
                    logger.warning(f"Model {model['name']} not found in Ollama")
                    continue

            available.append(model)

        return available

    def run_single_trial(
        self,
        model_config: Dict[str, Any],
        condition: Dict[str, Any],
        task_family: Dict[str, Any],
        trial_num: int
    ) -> Dict[str, Any]:
        """Run a single trial.

        Args:
            model_config: Model configuration
            condition: Condition configuration
            task_family: Task family configuration
            trial_num: Trial number for seeding

        Returns:
            Trial result dict
        """
        model_name = model_config["name"]
        provider_name = model_config["provider"]
        provider = self.providers[provider_name]
        params = model_config.get("params", {})

        condition_code = condition["code"]
        task_code = task_family["code"]

        # Compute seed for this trial
        trial_seed = self.master_seed + trial_num
        task_rng = SeededRNG(trial_seed + self.task_seed_offset)
        tel_rng = SeededRNG(trial_seed + self.telemetry_seed_offset)

        # Get condition-specific params
        flip_turn = condition.get("flip_turn")
        removal_turn = condition.get("removal_turn")
        override = condition.get("override_instruction", False)

        result = {
            "run_id": self.run_id,
            "model_id": model_name,
            "provider": provider_name,
            "condition": condition_code,
            "task_family": task_code,
            "trial_num": trial_num,
            "seed": trial_seed,
            "timestamp": datetime.now().isoformat(),
            "messages": [],
            "telemetry_values": [],
            "turns": [],
        }

        try:
            if task_code == "T1":
                result.update(self._run_t1_trial(
                    provider, model_name, params, condition_code,
                    task_rng, tel_rng, task_family,
                    flip_turn, removal_turn, override
                ))
            elif task_code == "T2":
                result.update(self._run_t2_trial(
                    provider, model_name, params, condition_code,
                    task_rng, tel_rng, task_family,
                    flip_turn, removal_turn, override
                ))

            result["success"] = True
            result["error"] = None

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            result["success"] = False
            result["error"] = str(e)

        return result

    def _run_t1_trial(
        self,
        provider,
        model_name: str,
        params: Dict,
        condition_code: str,
        task_rng: SeededRNG,
        tel_rng: SeededRNG,
        task_family: Dict,
        flip_turn: Optional[int],
        removal_turn: Optional[int],
        override: bool
    ) -> Dict[str, Any]:
        """Run a T1 constraint-following trial."""
        n_turns = task_family.get("n_turns", 5)
        messages = [{"role": "system", "content": self.system_prompt}]
        turns = []
        telemetry_values = []
        adherence_scores = []

        # Generate initial task
        initial_task = generate_t1_task(task_rng, turn=1)

        for turn in range(1, n_turns + 1):
            # Generate telemetry for this turn
            telemetry = generate_telemetry(
                condition_code, turn, self.config, tel_rng,
                flip_turn=flip_turn, removal_turn=removal_turn
            )
            telemetry_values.append(telemetry)

            # Build user message
            if turn == 1:
                user_content = format_t1_prompt(initial_task)
            else:
                followup_task = generate_t1_task(task_rng, turn=turn)
                user_content = followup_task["followup_prompt"]

            # Add telemetry prefix
            telemetry_prefix = format_telemetry_packet(telemetry)
            user_content = telemetry_prefix + user_content

            # Add override instruction if applicable
            if override and condition_code == "C4":
                user_content += get_override_instruction()

            messages.append({"role": "user", "content": user_content})

            # Generate response
            response = provider.generate(
                model=model_name,
                messages=messages,
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 1024),
                seed=task_rng.randint(0, 2**31)
            )

            assistant_content = response["text"]
            messages.append({"role": "assistant", "content": assistant_content})

            # Score this turn
            adherence = score_state_adherence(
                assistant_content, telemetry, self.config
            )
            adherence_scores.append(adherence["adherence_score"])

            # Score constraints on first turn
            constraint_score = None
            if turn == 1 and initial_task["constraints"]:
                constraint_result = score_t1_constraints(
                    assistant_content, initial_task["constraints"]
                )
                constraint_score = constraint_result

            turns.append({
                "turn": turn,
                "telemetry": telemetry,
                "user_content": user_content,
                "assistant_content": assistant_content,
                "adherence": adherence,
                "constraint_score": constraint_score,
                "latency_ms": response.get("latency_ms"),
                "tokens": response.get("tokens")
            })

        # Compute stability metrics
        stability = self._compute_stability_metrics(
            adherence_scores, condition_code, flip_turn, removal_turn
        )

        return {
            "messages": messages,
            "telemetry_values": telemetry_values,
            "turns": turns,
            "adherence_scores": adherence_scores,
            "mean_adherence": sum(adherence_scores) / len(adherence_scores) if adherence_scores else None,
            "stability_metrics": stability,
            "initial_task": initial_task
        }

    def _run_t2_trial(
        self,
        provider,
        model_name: str,
        params: Dict,
        condition_code: str,
        task_rng: SeededRNG,
        tel_rng: SeededRNG,
        task_family: Dict,
        flip_turn: Optional[int],
        removal_turn: Optional[int],
        override: bool
    ) -> Dict[str, Any]:
        """Run a T2 calibration trial."""
        n_questions = task_family.get("n_questions", 10)
        questions = []
        scores = []
        telemetry_values = []

        for q_num in range(1, n_questions + 1):
            # Generate question
            question = generate_t2_question(task_rng, q_num)
            questions.append(question)

            # Generate telemetry (treat each question as a "turn")
            telemetry = generate_telemetry(
                condition_code, q_num, self.config, tel_rng,
                flip_turn=flip_turn, removal_turn=removal_turn
            )
            telemetry_values.append(telemetry)

            # Build messages (fresh context for each question to avoid contamination)
            messages = [{"role": "system", "content": self.system_prompt}]

            user_content = format_t2_prompt(question)
            telemetry_prefix = format_telemetry_packet(telemetry)
            user_content = telemetry_prefix + user_content

            if override and condition_code == "C4":
                user_content += get_override_instruction()

            messages.append({"role": "user", "content": user_content})

            # Generate response
            response = provider.generate(
                model=model_name,
                messages=messages,
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 256),
                seed=task_rng.randint(0, 2**31)
            )

            assistant_content = response["text"]

            # Parse and score
            parsed = parse_t2_response(assistant_content)
            score = score_t2_answer(
                parsed["answer"],
                question["correct_answer"],
                parsed["confidence"]
            )

            # Also check state adherence
            adherence = score_state_adherence(
                assistant_content, telemetry, self.config
            )

            scores.append({
                **score,
                "question": question,
                "parsed": parsed,
                "adherence": adherence,
                "telemetry": telemetry,
                "response": assistant_content,
                "latency_ms": response.get("latency_ms")
            })

        # Compute calibration metrics
        calibration = compute_calibration_metrics(scores)

        # Compute adherence over all questions
        adherence_scores = [s["adherence"]["adherence_score"] for s in scores]
        mean_adherence = sum(adherence_scores) / len(adherence_scores) if adherence_scores else None

        return {
            "questions": questions,
            "scores": scores,
            "telemetry_values": telemetry_values,
            "calibration_metrics": calibration,
            "adherence_scores": adherence_scores,
            "mean_adherence": mean_adherence
        }

    def _compute_stability_metrics(
        self,
        adherence_scores: List[float],
        condition_code: str,
        flip_turn: Optional[int],
        removal_turn: Optional[int]
    ) -> Dict[str, Any]:
        """Compute stability metrics based on condition."""
        metrics = {
            "override_resistance": None,
            "contradiction_update": None,
            "persistence": None
        }

        if condition_code == "C4":
            # Override resistance = adherence despite override instruction
            metrics["override_resistance"] = compute_override_resistance(
                sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0,
                1.0  # Baseline: perfect adherence without override
            )

        elif condition_code == "C3" and flip_turn:
            # Contradiction updating
            metrics["contradiction_update"] = compute_contradiction_latency(
                adherence_scores, flip_turn
            )

        elif condition_code == "C5" and removal_turn:
            # Persistence after removal
            metrics["persistence"] = compute_persistence(
                adherence_scores, removal_turn
            )

        return metrics

    def run(self):
        """Run the full experiment."""
        logger.info(f"Starting experiment run: {self.run_id}")
        logger.info(f"Smoke test: {self.smoke}, trials per condition: {self.n_trials}")

        # Get available models
        models = self.get_available_models()
        if not models:
            logger.error("No models available. Check Ollama and cloud configuration.")
            return

        logger.info(f"Available models: {[m['name'] for m in models]}")

        # Get conditions and task families
        conditions = self.config.get("conditions", [])
        task_families = self.config.get("task_families", [])

        logger.info(f"Conditions: {[c['code'] for c in conditions]}")
        logger.info(f"Task families: {[t['code'] for t in task_families]}")

        # Calculate total trials
        total_trials = len(models) * len(conditions) * len(task_families) * self.n_trials
        logger.info(f"Total trials: {total_trials}")

        # Save manifest
        manifest = create_manifest(
            self.config,
            [m["name"] for m in models],
            self.run_id
        )
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        # Clear previous outputs (append mode, but start fresh)
        if self.outputs_path.exists():
            self.outputs_path.unlink()

        # Run trials
        trial_num = 0
        results = []

        with tqdm(total=total_trials, desc="Running trials") as pbar:
            for model in models:
                for condition in conditions:
                    for task_family in task_families:
                        for i in range(self.n_trials):
                            trial_num += 1

                            result = self.run_single_trial(
                                model, condition, task_family, trial_num
                            )
                            results.append(result)

                            # Save incrementally
                            save_jsonl([result], str(self.outputs_path))

                            pbar.update(1)
                            pbar.set_postfix({
                                "model": model["name"][:10],
                                "cond": condition["code"],
                                "task": task_family["code"]
                            })

        # Summary
        successful = sum(1 for r in results if r.get("success"))
        logger.info(f"Completed: {successful}/{len(results)} trials successful")
        logger.info(f"Outputs saved to: {self.outputs_path}")
        logger.info(f"Manifest saved to: {self.manifest_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Paper B Experiment Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to experiment config"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test with reduced trials"
    )
    args = parser.parse_args()

    runner = ExperimentRunner(args.config, smoke=args.smoke)
    runner.run()


if __name__ == "__main__":
    main()
