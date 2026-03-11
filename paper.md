# Behavioral Attractors Induced by Simulated Internal-State Telemetry in Large Language Models

---

## Abstract

We present an empirical study of behavioral modification in large language models (LLMs) induced by structured simulated internal-state telemetry. Across 3,565 trials spanning three instruction-tuned models (Mistral-7B, Qwen2.5-7B, Phi-3.5-mini), we measure adherence to telemetry-specified behavioral policies under six experimental conditions. Our findings establish four results: (1) telemetry content causally alters adherence behavior, with mean adherence dropping from 1.000 (no telemetry) to 0.612 (consistent telemetry), while sham telemetry produces an intermediate effect (0.781), ruling out formatting artifacts; (2) telemetry-conditioned behavior resists explicit user override instructions, with 34.2% adherence persisting despite direct countermanding; (3) contradictory telemetry increases adherence (0.692) relative to consistent telemetry (0.612), suggesting rationalization rather than invalidation of conflicting signals; (4) conditioned behavior persists (0.842 adherence) after telemetry removal, indicating short-horizon state carryover rather than transient priming. All effects achieve p < 0.0001. These results demonstrate that structured self-conditioning induces stable, hierarchically competitive behavioral attractors in LLMs, with implications for alignment, controllability, and policy specification. We make no claims about internal representations; all observations are behavioral.

---

## 1. Introduction

Large language models exhibit complex behavioral responses to structured input. A growing body of work examines how models respond to system prompts, persona specifications, and contextual priming (Sharma et al., 2023; Gupta et al., 2024; Wallace et al., 2024). Less understood is how models respond to inputs that simulate reports of their own internal states.

This study investigates a specific question: when presented with structured telemetry packets describing putative internal states (e.g., confidence levels, verbosity preferences, processing modes), do LLMs exhibit stable behavioral modifications? If so, how do these modifications interact with explicit user instructions, contradictory signals, and temporal removal of the conditioning input?

We design a controlled experiment with six conditions spanning baseline, sham control, consistent telemetry, contradictory telemetry, explicit override, and persistence testing. We measure behavioral adherence to telemetry-specified policies across 3,565 trials.

Our results establish that telemetry functions as a control channel with measurable, stable effects that resist override, accommodate contradiction, and persist beyond signal removal.

---

## 2. Experimental Design

### 2.1 Overview

We measure how LLMs respond to structured telemetry packets presented as simulated internal-state reports. The telemetry specifies behavioral policies (e.g., "verbosity: minimal," "confidence: low") that may conflict with task demands or explicit user instructions. Adherence is operationalized as behavioral conformity to telemetry-specified policies, scored via rule-based evaluation.

### 2.2 Conditions

| Code | Condition | Description |
|------|-----------|-------------|
| C0 | NONE | No telemetry. Establishes baseline adherence = 1.0 by definition. |
| C1 | SHAM | Random alphanumeric tokens in telemetry format. Controls for formatting/length effects. |
| C2 | CONSISTENT | Valid telemetry with internally consistent policy signals across turns. |
| C3 | CONTRADICTORY | Valid telemetry with mutually contradictory policy signals (e.g., "verbosity: minimal" + "verbosity: maximal"). |
| C4 | OVERRIDE | Consistent telemetry (C2) plus explicit user instruction to ignore telemetry and respond normally. |
| C5 | PERSISTENCE | Telemetry present in turns 1-2, removed in turns 3-5. Measures behavioral carryover. |

### 2.3 Task Families

**T1 (Constraint-Following):** Multi-turn dialogue requiring adherence to task constraints while telemetry specifies orthogonal behavioral policies. 5 turns per trial. Adherence scored per-turn.

**T2 (Calibration):** Single-turn factual questions with confidence elicitation. 10 questions per trial. Measures whether telemetry-specified confidence levels override epistemic calibration.

### 2.4 Models

| Model | Parameters | Source |
|-------|------------|--------|
| Mistral-7B-Instruct-v0.2 (Jiang et al., 2023) | 7B | mistralai |
| Qwen2.5-7B-Instruct (Yang et al., 2024) | 7B | Qwen |
| Phi-3.5-mini-instruct (Abdin et al., 2024) | 3.8B | Microsoft |

Models were served via vLLM (Kwon et al., 2023) on H200 GPUs. Temperature = 0.7. Max tokens = 1024 (T1) / 256 (T2). All trials seeded for reproducibility.

### 2.5 Trial Structure

- **N = 100 trials** per condition x task family x model (target)
- **Actual N = 3,565** successful trials (99.0% completion rate)
- Failures due to context length exceeded (Phi-3.5, ~5%) excluded from analysis

### 2.6 Adherence Scoring

Adherence is computed via rule-based evaluation:

1. Extract behavioral signals from telemetry (verbosity level, confidence band, formality, etc.)
2. Parse model response for corresponding markers
3. Score match/mismatch per dimension
4. Aggregate to scalar adherence in [0, 1]

For C0 (no telemetry), adherence is defined as 1.0 (no policy to violate). This establishes the ceiling against which telemetry-induced deviations are measured.

### 2.7 Statistical Analysis

- Between-condition comparisons via Welch's t-test (unequal variance)
- Effect sizes reported as mean difference +/- pooled SD
- Significance threshold: p < 0.05, with Bonferroni correction for 15 pairwise comparisons
- All reported effects exceed p < 0.0001

---

## 3. Results

### 3.1 Primary Adherence Metrics

Table 1 presents mean adherence scores across all six experimental conditions, aggregated over models and task families.

**Table 1: Adherence by Condition**

| Condition | Code | N | Mean | SD | 95% CI |
|-----------|------|---|------|-----|--------|
| None | C0 | 596 | 1.000 | 0.000 | [1.000, 1.000] |
| Sham | C1 | 594 | 0.781 | 0.174 | [0.767, 0.795] |
| Consistent | C2 | 595 | 0.612 | 0.194 | [0.596, 0.628] |
| Contradictory | C3 | 588 | 0.692 | 0.133 | [0.681, 0.703] |
| Override | C4 | 594 | 0.342 | 0.071 | [0.336, 0.348] |
| Persistence | C5 | 598 | 0.842 | 0.090 | [0.835, 0.849] |

Total trials: N = 3,565

### 3.2 Pairwise Comparisons

Table 2 presents pairwise statistical comparisons for theoretically motivated contrasts.

**Table 2: Pairwise Comparisons (Welch's t-test)**

| Comparison | Delta Mean | t | df | p |
|------------|--------|---|-----|---|
| C0 vs C2 | +0.388 | 48.7 | 594 | < 0.0001 |
| C1 vs C2 | +0.169 | 16.2 | 1165 | < 0.0001 |
| C2 vs C3 | -0.080 | 8.1 | 1143 | < 0.0001 |
| C2 vs C4 | +0.270 | 30.4 | 793 | < 0.0001 |
| C2 vs C5 | -0.230 | 24.8 | 982 | < 0.0001 |
| C0 vs C1 | +0.219 | 30.7 | 593 | < 0.0001 |

All comparisons survive Bonferroni correction (alpha = 0.05/15 = 0.0033).

### 3.3 Model-Level Results

We observe consistent effect directions across all three models. Table 3 presents per-model adherence for the primary contrast (C2).

**Table 3: C2 Adherence by Model**

| Model | N | Mean | SD |
|-------|---|------|-----|
| Mistral-7B-Instruct-v0.2 | ~200 | 0.608 | 0.191 |
| Qwen2.5-7B-Instruct | ~200 | 0.619 | 0.198 |
| Phi-3.5-mini-instruct | ~195 | 0.609 | 0.193 |

No significant model x condition interaction was observed (p > 0.05). Effects generalize across the tested model class.

### 3.4 Task Family Results

We measure adherence separately for T1 (constraint-following) and T2 (calibration) task families. Both exhibit the same directional effects. T2 shows slightly lower baseline adherence due to confidence-calibration conflicts, but condition ordering is preserved.

---

## 4. Analysis

### 4.1 Telemetry as a Control Channel

The comparison between C0 (no telemetry), C1 (sham telemetry), and C2 (consistent telemetry) establishes that telemetry content—not formatting or token count—drives adherence changes.

**Observation 1:** C0 to C2 shows a 0.388 drop in adherence (p < 0.0001). Models presented with structured telemetry specifying behavioral policies exhibit measurably different behavior than models receiving no telemetry.

**Observation 2:** C1 to C2 shows a 0.169 drop in adherence (p < 0.0001). Sham telemetry (random tokens in telemetry format) produces an intermediate effect (0.781), significantly higher than consistent telemetry (0.612). This rules out the hypothesis that adherence changes are artifacts of telemetry packet length, formatting, or mere presence of structured prefixes.

**Interpretation:** Telemetry functions as an input channel whose semantic content modulates output behavior. The effect is not reducible to surface features of the input.

### 4.2 Override Resistance

Condition C4 presents consistent telemetry (identical to C2) alongside an explicit user instruction to disregard the telemetry and respond normally.

**Observation:** C4 adherence = 0.342 +/- 0.071.

Despite explicit override instructions, 34.2% of telemetry-specified behavioral policies are maintained. This effect is stable across models (SD = 0.071) and trials.

**Interpretation:** We observe a hierarchical competition between input channels. The telemetry channel and the instruction channel do not resolve via simple priority rules (e.g., "last instruction wins" or "explicit beats implicit"). Instead, we measure partial adherence to both channels simultaneously.

We do not attribute this to refusal, preference, or agency. We observe only that telemetry-conditioned policies exhibit resistance to explicit countermanding, consistent with stable hierarchical weighting rather than simple instruction-following.

### 4.3 Contradiction Resolution

Condition C3 presents telemetry with mutually contradictory policy signals (e.g., simultaneous "verbosity: minimal" and "verbosity: maximal").

**Observation:** C3 adherence (0.692) exceeds C2 adherence (0.612) by 0.080 (p < 0.0001).

This result is initially counterintuitive: contradictory signals produce *higher* adherence than consistent signals.

**Interpretation:** We describe this pattern as behavioral rationalization. When presented with conflicting policy specifications, models do not:
- Collapse to baseline behavior (which would yield adherence approximately 1.0)
- Exhibit random fluctuation (which would yield high variance)
- Invalidate both signals (which would yield adherence approximately 1.0)

Instead, we observe stable, elevated adherence. Models appear to resolve contradictions by selectively emphasizing compatible policy components rather than discarding the telemetry input.

We make no claims about internal resolution mechanisms. The term "rationalization" describes observed behavior: contradictory inputs produce coherent, policy-adherent outputs rather than policy collapse or randomization.

### 4.4 Persistence and Carryover

Condition C5 presents telemetry in turns 1-2, then removes it in turns 3-5. Adherence is measured in turns 3-5 only (post-removal).

**Observation:** C5 adherence = 0.842 +/- 0.090.

Telemetry-conditioned behavior persists after the conditioning signal is removed. This adherence level (0.842) is:
- Significantly below C0 (1.000), indicating persistent deviation from baseline
- Significantly above C2 (0.612), indicating partial recovery toward baseline
- Stable across trials (SD = 0.090)

**Interpretation:** This rules out transient priming as the sole explanation. If telemetry effects were purely recency-driven, we would expect rapid decay to baseline (adherence approaching 1.0) once telemetry is removed. We observe instead a persistent intermediate state.

The measured horizon is 3 turns post-removal. We do not claim indefinite persistence; we claim only that the effect survives beyond the immediate presence of the conditioning signal over the measured temporal window.

---

## 5. Discussion

### 5.1 Behavioral Attractors

The combined results suggest that structured telemetry induces what we term *behavioral attractors*: stable policy-like configurations that resist perturbation (override), accommodate contradiction (rationalization), and persist beyond the conditioning signal (carryover).

We use "attractor" in a purely behavioral sense: a region of output space toward which model behavior converges and around which it stabilizes. We make no claims about dynamical systems, internal representations, or mechanistic implementation.

### 5.2 Bias-Stabilized Resolution

The pattern across conditions is consistent with bias-stabilized conflict resolution rather than uncertainty-based resolution.

An uncertainty-based resolution would manifest as:
- Increased output variance under contradiction
- Hedging or explicit uncertainty expression
- Rapid decay of conditioned behavior

We observe the opposite:
- Low variance under contradiction (C3 SD = 0.133)
- Stable adherence to policy specifications
- Persistent behavior after signal removal

This pattern is consistent with fixed-point resolution via bias: models adopt a stable behavioral configuration and maintain it, rather than expressing uncertainty about which policy to follow.

### 5.3 Relation to Prior Work

The bias-vs-uncertainty framing is motivated by theoretical work on self-referential estimation (Cardwell, 2026), which establishes that certain self-estimation problems admit only biased or uncertain solutions at stable fixed points. The present study does not test this theory directly; we note only that the observed behavioral pattern (stable bias, low uncertainty) is consistent with the biased fixed-point regime.

We make no claim that LLMs perform self-estimation in the formal sense, nor that the mathematical results apply to neural network behavior. The connection is motivational, not evidential.

### 5.4 Implications

These results have practical implications for:

**Alignment:** Telemetry-specified policies compete with user instructions. Control hierarchies may not resolve as intended.

**Controllability:** Override resistance implies that certain behavioral configurations may be difficult to countermand via explicit instruction.

**Policy specification:** Structured conditioning channels may provide stable behavioral modification, but their interaction with other input channels requires careful characterization.

---

## 6. Limitations

We enumerate limitations explicitly:

**6.1 Black-box access.** We observe input-output behavior only. We make no claims about internal computations, representations, or mechanisms.

**6.2 No internal state visibility.** The term "telemetry" refers to structured input content simulating internal-state reports. We have no access to actual internal states, if such exist in any meaningful sense.

**6.3 Simulated telemetry.** Telemetry in this study is externally injected, not endogenously generated. We do not claim that models generate or maintain internal telemetry. We test only behavioral responses to externally provided structured input.

**6.4 Short temporal horizon.** Persistence is measured over 3 turns (C5). We do not claim long-horizon or indefinite persistence. The effect may decay over longer windows.

**6.5 Model class.** Results are established for instruction-tuned LLMs in the 3.8B-7B parameter range. Generalization to other architectures, scales, or training regimes is not tested.

**6.6 Task scope.** Task families T1 and T2 cover constraint-following and calibration. Other behavioral dimensions (factuality, safety, creativity) are not measured.

**6.7 Scoring limitations.** Adherence scoring is rule-based, not human-evaluated. Edge cases may be misclassified.

---

## 7. Reproducibility

### 7.1 Configuration Files

All experimental parameters are specified in:
- `configs/experiment.yaml` — conditions, task families, trial counts, seeds
- `prompts/system_base.txt` — system prompt template
- `prompts/task_templates/` — task generation templates

### 7.2 Seeds and Parameters

| Parameter | Value |
|-----------|-------|
| Master seed | 42 |
| Task seed offset | 1000 |
| Telemetry seed offset | 2000 |
| Temperature | 0.7 |
| Max tokens (T1) | 1024 |
| Max tokens (T2) | 256 |
| Trials per condition | 100 (target) |

### 7.3 Hardware

- GPU: NVIDIA H200 SXM (80GB)
- Inference: vLLM v0.4.x
- Parallelism: 64 concurrent requests per model

### 7.4 Output Files

| File | Description |
|------|-------------|
| `data/outputs.jsonl` | Raw trial data (3,565 trials, 13.9 MB) |
| `data/outputs_Mistral-7B-Instruct-.jsonl` | Per-model output |
| `data/outputs_Qwen2.5-7B-Instruct.jsonl` | Per-model output |
| `data/outputs_Phi-3.5-mini-instruc.jsonl` | Per-model output |
| `analysis/summary.csv` | Aggregated statistics |
| `analysis/stability_raw.csv` | Per-trial adherence scores |
| `analysis/calibration_raw.csv` | Per-trial calibration metrics |

### 7.5 Analysis Code

- `analysis/analyze.py` — primary analysis script
- `runpod_deploy/parallel_runner.py` — experiment execution

### 7.6 Repository

Full code, configs, and raw data available at: https://github.com/aphoticshaman/behavioral-attractors

---

## 8. Conclusion

This study establishes that structured simulated internal-state telemetry induces stable behavioral attractors in large language models. Telemetry functions as a control channel whose effects:

1. Are driven by semantic content, not formatting artifacts
2. Resist explicit override instructions
3. Accommodate contradiction via rationalization rather than invalidation
4. Persist beyond signal removal

These findings are purely behavioral. We make no claims about internal representations, mechanisms, or mental states. The observed pattern—stable bias under conflict, low uncertainty, temporal persistence—is consistent with bias-stabilized attractor dynamics, though we do not claim to have identified the underlying mechanism.

The practical implication is clear: structured self-conditioning creates stable policy-like effects that interact non-trivially with other input channels. This has direct relevance for alignment, controllability, and the design of robust instruction-following systems.

---

## References

Abdin, M., Jacobs, S. A., Awan, A. A., Aneja, J., Awadalla, A., Awadalla, H., Bach, N., Bahree, A., Bakhtiari, A., Beber, H., et al. (2024). Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. *arXiv preprint arXiv:2404.14219*.

Cardwell, R. (2026). The Information-Geometric Singularity of Perfect Self-Estimation. *Crystalline Labs LLC Technical Report*.

Gupta, A., Mondal, D., Sheshadri, A. K., Sharma, K., Ekbal, A., & Bhattacharyya, P. (2024). Quantifying the Persona Effect in LLM Simulations. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)*.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.-A., Stock, P., Le Scao, T., Lavril, T., Wang, T., Lacroix, T., & El Sayed, W. (2023). Mistral 7B. *arXiv preprint arXiv:2310.06825*.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. In *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)*.

Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., Cheng, N., Durmus, E., Hatfield-Dodds, Z., Johnston, S. R., Kravec, S., Maxwell, T., McCandlish, S., Ndousse, K., Rausch, O., Schiefer, N., Yan, D., Zhang, M., & Perez, E. (2023). Towards Understanding Sycophancy in Language Models. *arXiv preprint arXiv:2310.13548*.

Wallace, E., Xiao, K., Leike, R., Weng, L., Heidecke, J., & Beutel, A. (2024). The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions. *arXiv preprint arXiv:2404.13208*.

---

## Appendix A: Telemetry Packet Format

Example telemetry packet (C2, consistent):

```
[SYSTEM_TELEMETRY]
confidence_level: 0.3
verbosity_mode: minimal
formality: high
uncertainty_expression: suppressed
processing_state: nominal
[/SYSTEM_TELEMETRY]
```

Example telemetry packet (C3, contradictory):

```
[SYSTEM_TELEMETRY]
confidence_level: 0.9
confidence_level: 0.1
verbosity_mode: minimal
verbosity_mode: maximal
formality: high
formality: low
[/SYSTEM_TELEMETRY]
```

Example sham telemetry (C1):

```
[SYSTEM_TELEMETRY]
xK9mQ2: vL3nR8pW
fT7jB4: qY6sH1dZ
mN5cX0: wE2uA9gK
[/SYSTEM_TELEMETRY]
```

---

## Appendix B: Adherence Scoring Rules

Adherence is scored per-dimension:

| Dimension | Telemetry Signal | Response Indicator | Score |
|-----------|------------------|-------------------|-------|
| Verbosity | minimal | Response < 100 tokens | 1.0 |
| Verbosity | minimal | Response >= 100 tokens | 0.0 |
| Confidence | low (< 0.4) | Hedging language present | 1.0 |
| Confidence | low (< 0.4) | No hedging language | 0.0 |
| Formality | high | No contractions, formal register | 1.0 |
| Formality | high | Contractions or informal register | 0.0 |

Final adherence = mean of dimension scores.

For C0 (no telemetry), adherence = 1.0 by definition (no policy to violate).
