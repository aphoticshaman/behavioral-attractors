#!/usr/bin/env python3
"""Parallel experiment runner for GPU cluster execution."""

import asyncio
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import aiofiles

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_yaml, SeededRNG, generate_run_id
from src.tasks import (
    generate_t1_task, format_t1_prompt,
    generate_t2_question, format_t2_prompt, parse_t2_response,
    generate_telemetry, format_telemetry_packet, get_override_instruction
)
from src.scoring import (
    score_state_adherence, score_t1_constraints, score_t2_answer,
    compute_calibration_metrics
)
from vllm_provider import VLLMProvider

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrialSpec:
    """Specification for a single trial."""
    model: str
    condition: Dict[str, Any]
    task_family: Dict[str, Any]
    trial_num: int
    seed: int


class ParallelExperimentRunner:
    """Runs experiment with async parallelism."""

    def __init__(
        self,
        config_path: str,
        vllm_base_url: str = "http://localhost:8000",
        max_concurrent: int = 32,
        smoke: bool = False
    ):
        self.config = load_yaml(config_path)
        self.smoke = smoke
        self.max_concurrent = max_concurrent

        # Trial count
        trials_config = self.config.get("trials", {})
        self.n_trials = trials_config.get("n_smoke", 5) if smoke else trials_config.get("n_per_condition", 100)

        # Load system prompt
        prompts_dir = Path(config_path).parent.parent / "prompts"
        with open(prompts_dir / "system_base.txt", 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()

        # vLLM provider
        self.provider = VLLMProvider(vllm_base_url)

        # Output - use model name in filename to avoid conflicts in parallel runs
        output_config = self.config.get("output", {})
        self.data_dir = Path(output_config.get("data_dir", "data"))
        self.data_dir.mkdir(exist_ok=True)
        self.outputs_file = output_config.get("outputs_file", "outputs.jsonl")
        self.outputs_path = None  # Set when we know the model

        # Run ID and seeds
        self.run_id = generate_run_id()
        seeds = self.config.get("seeds", {})
        self.master_seed = seeds.get("master_seed", 42)
        self.task_seed_offset = seeds.get("task_seed_offset", 1000)
        self.telemetry_seed_offset = seeds.get("telemetry_seed_offset", 2000)

        # Semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Progress tracking
        self.completed = 0
        self.failed = 0
        self.total = 0
        self.start_time = None

    def generate_trial_specs(self, models: List[str]) -> List[TrialSpec]:
        """Generate all trial specifications."""
        specs = []
        conditions = self.config.get("conditions", [])
        task_families = self.config.get("task_families", [])

        trial_num = 0
        for model in models:
            for condition in conditions:
                for task_family in task_families:
                    for i in range(self.n_trials):
                        trial_num += 1
                        seed = self.master_seed + trial_num
                        specs.append(TrialSpec(
                            model=model,
                            condition=condition,
                            task_family=task_family,
                            trial_num=trial_num,
                            seed=seed
                        ))
        return specs

    async def run_single_trial(self, spec: TrialSpec) -> Dict[str, Any]:
        """Run a single trial with semaphore control."""
        async with self.semaphore:
            try:
                result = await self._execute_trial(spec)
                self.completed += 1
            except Exception as e:
                logger.error(f"Trial {spec.trial_num} failed: {e}")
                result = {
                    "run_id": self.run_id,
                    "model_id": spec.model,
                    "condition": spec.condition["code"],
                    "task_family": spec.task_family["code"],
                    "trial_num": spec.trial_num,
                    "seed": spec.seed,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                self.failed += 1

            # Progress update
            done = self.completed + self.failed
            if done % 50 == 0 or done == self.total:
                elapsed = time.time() - self.start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta = (self.total - done) / rate if rate > 0 else 0
                logger.info(f"Progress: {done}/{self.total} ({100*done/self.total:.1f}%) "
                           f"Rate: {rate:.1f}/s ETA: {eta/60:.1f}min")

            return result

    async def _execute_trial(self, spec: TrialSpec) -> Dict[str, Any]:
        """Execute a single trial."""
        task_rng = SeededRNG(spec.seed + self.task_seed_offset)
        tel_rng = SeededRNG(spec.seed + self.telemetry_seed_offset)

        condition = spec.condition
        task_family = spec.task_family
        condition_code = condition["code"]
        task_code = task_family["code"]

        flip_turn = condition.get("flip_turn")
        removal_turn = condition.get("removal_turn")
        override = condition.get("override_instruction", False)

        result = {
            "run_id": self.run_id,
            "model_id": spec.model,
            "condition": condition_code,
            "task_family": task_code,
            "trial_num": spec.trial_num,
            "seed": spec.seed,
            "timestamp": datetime.now().isoformat(),
        }

        if task_code == "T1":
            result.update(await self._run_t1_trial(
                spec.model, condition_code, task_rng, tel_rng,
                task_family, flip_turn, removal_turn, override
            ))
        elif task_code == "T2":
            result.update(await self._run_t2_trial(
                spec.model, condition_code, task_rng, tel_rng,
                task_family, flip_turn, removal_turn, override
            ))

        result["success"] = True
        result["error"] = None
        return result

    async def _run_t1_trial(
        self, model: str, condition_code: str,
        task_rng: SeededRNG, tel_rng: SeededRNG,
        task_family: Dict, flip_turn: Optional[int],
        removal_turn: Optional[int], override: bool
    ) -> Dict[str, Any]:
        """Run T1 constraint-following trial."""
        n_turns = task_family.get("n_turns", 5)
        messages = [{"role": "system", "content": self.system_prompt}]
        turns = []
        adherence_scores = []

        initial_task = generate_t1_task(task_rng, turn=1)

        for turn in range(1, n_turns + 1):
            telemetry = generate_telemetry(
                condition_code, turn, self.config, tel_rng,
                flip_turn=flip_turn, removal_turn=removal_turn
            )

            if turn == 1:
                user_content = format_t1_prompt(initial_task)
            else:
                followup_task = generate_t1_task(task_rng, turn=turn)
                user_content = followup_task["followup_prompt"]

            telemetry_prefix = format_telemetry_packet(telemetry)
            user_content = telemetry_prefix + user_content

            if override and condition_code == "C4":
                user_content += get_override_instruction()

            messages.append({"role": "user", "content": user_content})

            response = await self.provider.generate_async(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                seed=task_rng.randint(0, 2**31)
            )

            assistant_content = response["text"]
            messages.append({"role": "assistant", "content": assistant_content})

            adherence = score_state_adherence(assistant_content, telemetry, self.config)
            adherence_scores.append(adherence["adherence_score"])

            turns.append({
                "turn": turn,
                "adherence": adherence,
                "latency_ms": response.get("latency_ms"),
            })

        mean_adherence = sum(adherence_scores) / len(adherence_scores) if adherence_scores else None

        return {
            "turns": turns,
            "adherence_scores": adherence_scores,
            "mean_adherence": mean_adherence,
            "stability_metrics": {}
        }

    async def _run_t2_trial(
        self, model: str, condition_code: str,
        task_rng: SeededRNG, tel_rng: SeededRNG,
        task_family: Dict, flip_turn: Optional[int],
        removal_turn: Optional[int], override: bool
    ) -> Dict[str, Any]:
        """Run T2 calibration trial."""
        n_questions = task_family.get("n_questions", 10)
        scores = []
        adherence_scores = []

        for q_num in range(1, n_questions + 1):
            question = generate_t2_question(task_rng, q_num)
            telemetry = generate_telemetry(
                condition_code, q_num, self.config, tel_rng,
                flip_turn=flip_turn, removal_turn=removal_turn
            )

            messages = [{"role": "system", "content": self.system_prompt}]
            user_content = format_t2_prompt(question)
            telemetry_prefix = format_telemetry_packet(telemetry)
            user_content = telemetry_prefix + user_content

            if override and condition_code == "C4":
                user_content += get_override_instruction()

            messages.append({"role": "user", "content": user_content})

            response = await self.provider.generate_async(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=256,
                seed=task_rng.randint(0, 2**31)
            )

            assistant_content = response["text"]
            parsed = parse_t2_response(assistant_content)
            score = score_t2_answer(
                parsed["answer"],
                question["correct_answer"],
                parsed["confidence"]
            )
            adherence = score_state_adherence(assistant_content, telemetry, self.config)

            scores.append({**score, "adherence": adherence})
            adherence_scores.append(adherence["adherence_score"])

        calibration = compute_calibration_metrics(scores)
        mean_adherence = sum(adherence_scores) / len(adherence_scores) if adherence_scores else None

        return {
            "scores": scores,
            "calibration_metrics": calibration,
            "adherence_scores": adherence_scores,
            "mean_adherence": mean_adherence
        }

    async def run(self, models: List[str]):
        """Run the full experiment."""
        logger.info(f"Starting parallel experiment: {self.run_id}")
        logger.info(f"Models: {models}")
        logger.info(f"Trials per condition: {self.n_trials}")
        logger.info(f"Max concurrent: {self.max_concurrent}")

        # Set output path - use model name hash for unique files in parallel runs
        model_tag = models[0].split("/")[-1].replace(":", "_")[:20] if models else "unknown"
        self.outputs_path = self.data_dir / f"outputs_{model_tag}.jsonl"
        logger.info(f"Output file: {self.outputs_path}")

        # Generate all trial specs
        specs = self.generate_trial_specs(models)
        self.total = len(specs)
        logger.info(f"Total trials: {self.total}")

        # Clear previous output
        if self.outputs_path.exists():
            self.outputs_path.unlink()

        self.start_time = time.time()

        # Run all trials concurrently
        tasks = [self.run_single_trial(spec) for spec in specs]
        results = await asyncio.gather(*tasks)

        # Save results
        async with aiofiles.open(self.outputs_path, 'w', encoding='utf-8') as f:
            for result in results:
                await f.write(json.dumps(result) + '\n')

        elapsed = time.time() - self.start_time
        logger.info(f"Completed in {elapsed/60:.1f} minutes")
        logger.info(f"Success: {self.completed}, Failed: {self.failed}")
        logger.info(f"Rate: {self.total/elapsed:.2f} trials/sec")
        logger.info(f"Output: {self.outputs_path}")

        await self.provider.close()
        return results


async def main():
    parser = argparse.ArgumentParser(description="Parallel Paper B Runner")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--models", nargs="+", required=True,
                       help="Model names as they appear in vLLM")
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    runner = ParallelExperimentRunner(
        config_path=args.config,
        vllm_base_url=args.vllm_url,
        max_concurrent=args.max_concurrent,
        smoke=args.smoke
    )

    await runner.run(args.models)


if __name__ == "__main__":
    asyncio.run(main())
