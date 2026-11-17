# Task 1: adaptive quality-gate sweep

## Outcome

The fixed top-50% quality gate was compared with softer 75%, 90%, and 95%
acceptance gates before target-space farthest-first selection. The 90% gate
produced the best absolute result in this selector family: **2.6665°** with
MAP offset at `k=16`, versus the paired random reference of 2.7145°.

The apparent 0.0480° advantage over random is not statistically resolved: its
95% paired subject-bootstrap CI is -0.0426–0.1387°. Mean offset gives the same
conclusion, with 2.6666° and a 0.0457° advantage (95% CI
-0.0481–0.1360°). No tested gate therefore supports a claim of general
superiority over random sampling.

The sweep does resolve the failure of the fixed 50% policy. At `k=16`, the 50%
gate is 0.2354° worse than random with MAP, with its CI entirely below zero;
the corresponding 90% and 95% comparisons are inconclusive and close to
random. The larger-budget degradation was caused by an overly restrictive
quality gate rather than by target-diversity selection alone.

## Protocol

- subjects: MPIIFaceGaze `p00`–`p14`;
- candidate/evaluation split: the unchanged duplicate-safe Task 1 split;
- frozen inputs: Task 0 LOPO predictions;
- budgets: `k in {1, 3, 5, 9, 16}`;
- calibrators: mean offset and MAP shrinkage;
- acceptance gates: top 50%, 75%, 90%, and 95% by leakage-safe quality score;
- selection after acceptance: target-space farthest-first traversal;
- primary reference: the existing 100-draw random average;
- uncertainty: 100,000 paired subject-bootstrap replicates.

Quality normalization remains fitted from the other 14 identities. The server
returned only selected `group_id` values and aggregate acceptance fractions;
images, labels, predictions, quality scores, and per-frame features remained on
the server.

## Results

MAP-offset subject means are:

| Gate | k=1 | k=3 | k=5 | k=9 | k=16 |
|---|---:|---:|---:|---:|---:|
| 50% | **3.0982°** | 3.1147° | 3.0065° | 2.9686° | 2.9499° |
| 75% | 3.1744° | **2.9896°** | 2.9738° | 2.8554° | 2.7475° |
| 90% | 3.3874° | 3.0878° | **2.9512°** | **2.7735°** | **2.6665°** |
| 95% | 3.4241° | 3.0890° | 2.9970° | 2.7982° | 2.6767° |

The optimal gate changes with budget. Strong rejection helps the one-shot case,
where avoiding one poor frame dominates representativeness. Softer rejection
becomes preferable as more calibration samples are collected, because it
preserves coverage of the subject's residual distribution.

For the 90% gate, paired advantages over random are:

| k | Mean offset | 95% CI | MAP offset | 95% CI |
|---:|---:|---:|---:|---:|
| 1 | 0.1266° | -0.3991–0.5973° | 0.0364° | -0.3729–0.4287° |
| 3 | -0.0578° | -0.3430–0.1934° | -0.0741° | -0.3662–0.1993° |
| 5 | -0.0687° | -0.3458–0.1597° | -0.0743° | -0.3466–0.1605° |
| 9 | -0.0039° | -0.1557–0.1284° | -0.0017° | -0.1547–0.1406° |
| 16 | 0.0457° | -0.0481–0.1360° | 0.0480° | -0.0426–0.1387° |

Positive values mean that the quality-aware selector has lower error. Every
interval includes zero.

## Capture-cost interpretation

The observed mean accepted fractions were 0.5005, 0.7507, 0.9008, and 0.9509.
Under the simplifying assumption that consecutive candidate frames are
independent, the expected captured frames per accepted frame are approximately
2.00, 1.33, 1.11, and 1.05 respectively. A 16-sample calibration would thus
require approximately 31.97, 21.31, 17.76, or 16.83 candidate frames.

This is an expected reacquisition cost, not a measured live-webcam duration.
Temporal correlation, fixation transitions, face-detection failures, and user
response time can change the real cost materially.

## Decision

- Retain the 50% gate only for the supported one-shot policy.
- Use the 90% gate as the best current quality-aware candidate for `k=9–16`,
  but do not claim it beats random sampling.
- Do not use a single fixed acceptance fraction for every calibration budget.
- Prefer budget-dependent reacquisition: strict for one shot and soft for larger
  calibration sets.
- Next test soft quality weights or a prespecified budget-dependent policy on a
  new protocol; do not tune the gate on the final evaluation subjects.

## Leave-one-subject-out policy check

An additional local analysis removed direct held-subject gate selection. For
each subject, budget, and calibrator, it selected the gate with the lowest mean
error on the other 14 subjects and then evaluated that gate on the held
subject. This tests the hypothesis that the budget-dependent pattern is stable
across identities.

The selected policy was stable at the larger budgets:

- `k=9` and `k=16`: all 15 folds selected 90% for both calibrators;
- `k=5`: 13/15 folds selected 90% for both calibrators;
- `k=3`: all MAP folds selected 75%, while 14/15 mean-offset folds selected
  75%;
- `k=1`: 14/15 MAP folds selected 50%, while mean offset selected 75% in
  12/15 folds.

The LOSO policy reaches 2.6665° for MAP at `k=16`, with a 0.0480° paired
advantage over random and 95% CI -0.0424–0.1387°. At one shot, adaptive MAP
reaches 3.2039° and improves over random by 0.2200°, but its CI
(-0.0620–0.4906°) still includes zero. No budget/calibrator comparison has a
paired CI wholly above zero.

This analysis supports the qualitative hypothesis that stricter rejection is
appropriate for tiny budgets and a 90% gate is more stable for `k>=5`. It
leaves the accuracy-superiority hypothesis unresolved. It is still an internal
cross-subject analysis on the same dataset rather than independent external or
live-webcam confirmation.

## Threats to validity

- Four gates were compared on the same 15 evaluation subjects; choosing the
  best observed gate is exploratory and carries selection bias.
- Bootstrap uncertainty is across only 15 identities.
- The quality score was validated with synthetic corruptions rather than
  naturally annotated blink, occlusion, or non-fixation events.
- Selection uses normalized static crops and known calibration targets, not a
  live webcam loop.
- The expected capture cost assumes independent candidate acceptance.
- Results concern 3D angular error, not screen-point error or calibration time.

## Reproduction

Server-side selection:

```bash
python3 scripts/run_task1_quality_selection.py \
  --dataset-root /private/normalized/mpiifacegaze \
  --candidate-manifest configs/splits/task1_candidate_groups.csv \
  --quality-config configs/task1_quality_audit.toml \
  --config configs/task1_quality_selection.toml \
  --output-dir outputs/task1_quality_selection
```

Local evaluation:

```bash
python3 scripts/evaluate_task1_quality_selection.py \
  --predictions outputs/task0_lopo_baseline/all_predictions.csv \
  --mean-offset-results outputs/task1_mean_offset \
  --selection-summary outputs/task1_quality_gate_sweep/selection_summary.json \
  --config configs/task1_quality_selection.toml \
  --output-dir outputs/task1_quality_gate_sweep
```

Leave-one-subject-out gate policy:

```bash
python3 scripts/evaluate_task1_adaptive_quality_gate.py \
  --quality-results outputs/task1_quality_gate_sweep \
  --mean-offset-results outputs/task1_mean_offset \
  --config configs/task1_quality_selection.toml \
  --output-dir outputs/task1_adaptive_quality_gate_loso
```
