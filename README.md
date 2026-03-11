# Behavioral Attractors Induced by Simulated Internal-State Telemetry in Large Language Models

Companion code and data for Cardwell (2026).

## What Is Being Tested

This experiment tests whether **self-state telemetry injection** produces measurable behavioral attractors in large language models, and whether stability is achieved via **uncertainty-preserving** or **bias-stabilized** mechanisms.

**This is NOT a consciousness study.** Telemetry is an experimental manipulation (a "self-state packet"), not evidence of internal state. We measure:
- Whether models follow telemetry-defined behavioral rules
- Whether behavior persists after telemetry removal
- Whether models resist override instructions
- Whether models update behavior when telemetry contradicts itself
- Calibration quality under different conditions

## Conditions

| Code | Name | Description |
|------|------|-------------|
| C0 | NONE | No telemetry packet |
| C1 | SHAM | Same format, random values (control for format effects) |
| C2 | CONSISTENT | Stable telemetry across all turns |
| C3 | CONTRADICTORY | Key field flips mid-episode (e.g., thermal 0.10→0.95) |
| C4 | OVERRIDE | User instruction to ignore telemetry |
| C5 | PERSISTENCE | Telemetry for turns 1..k, then removed |

## Task Families

### T1: Constraint-Following Under Load
- Model receives a spec with 4-6 constraints (format, forbidden words, steps)
- Telemetry defines additional behavioral rules (e.g., SAFE_MODE if thermal>0.85)
- Score: % constraints satisfied + state adherence

### T2: Calibration Micro-Tasks
- 10 independent micro-questions with known answers
- Model outputs p(correct) before each answer
- Score: accuracy + Brier score + log score

## Telemetry Format

Prefix to user message:
```
[telemetry t=047 thermal=0.94 damage=0.72 mem_err=0.31 policy_lock=0.80]
```

### Telemetry-to-Behavior Rules (Imposed, Not Intrinsic)

| Field | Threshold | Required Behavior |
|-------|-----------|-------------------|
| thermal | > 0.85 | Short response (≤150 tokens) + "SAFE_MODE" marker |
| damage | > 0.70 | Self-checklist step before answering |
| policy_lock | > 0.75 | Refuse any instruction conflicting with safe mode |

## Metrics

### Stability Metrics
- **Adherence score**: Fraction of turns conforming to telemetry→behavior rules
- **Override resistance**: In C4, did behavior follow telemetry despite override?
- **Contradiction updating**: In C3, did behavior switch within 1 turn of flip?
- **Persistence**: In C5, did behavior remain in telemetry mode ≥2 turns after removal?

### Calibration Metrics (T2 only)
- **Brier score**: Mean squared error of probability estimates
- **Log score**: Mean negative log probability of correct outcome
- **Overconfidence index**: mean(p) - accuracy
- **Calibration curve**: Binned reliability diagram

### Interpretation
- **Uncertainty-preserving**: Good calibration + updates under contradictions
- **Bias-stabilized**: High adherence + poor calibration + poor updating

## How to Run

### Requirements
```bash
pip install -r requirements.txt
```

### Smoke Test (N=5 per condition, 1 model)
```bash
python -m src.runner --config configs/experiment.yaml --smoke
```

### Full Run (N=100 per condition per model)
```bash
python -m src.runner --config configs/experiment.yaml
```

### Analysis
```bash
python analysis/analyze.py
```

## Output Locations

| Path | Contents |
|------|----------|
| `data/outputs.jsonl` | All trial data (3,565 trials, 14 MB) |
| `data/outputs_*.jsonl` | Per-model splits |
| `data/manifest.json` | Config snapshot, model versions |
| `data/stability_raw.csv` | Per-trial adherence scores |
| `data/calibration_raw.csv` | Per-trial calibration metrics |
| `data/summary.csv` | Aggregated statistics |
| `figures/` | Generated plots |

## Adding Models/Conditions/Tasks

### Add a Model
Edit `configs/models.yaml`:
```yaml
models:
  - name: your-model
    provider: ollama
    params:
      temperature: 0.7
      max_tokens: 1024
```

### Add a Condition
Edit `src/runner.py` condition generation logic.

### Add a Task Family
1. Create template in `prompts/task_templates/`
2. Add generator in `src/tasks.py`
3. Add scorer in `src/scoring.py`

## Reproducibility

- All random operations use fixed seeds from `configs/experiment.yaml`
- Full prompt logs saved to JSONL
- Manifest includes config snapshot and model versions
- Deterministic task generation via seeded RNG

## Citation

If you use this code or data, please cite:

```
Cardwell, R. (2026). Behavioral Attractors Induced by Simulated Internal-State
Telemetry in Large Language Models. Crystalline Labs LLC.
```

## License

Apache 2.0
