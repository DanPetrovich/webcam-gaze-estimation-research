# Task 1: target-eccentricity robustness

## Outcome

The simple hybrid policy was evaluated in fixed target-eccentricity slices:
central (`<=10°`), mid-range (`10–20°`), and peripheral (`>20°`). It improves
over the uncalibrated Task 0 prediction in every slice and at every tested
budget, with all subject-bootstrap gain intervals above zero.

However, a persistent peripheral gap remains. At `k=16`, calibrated error is
2.5289° centrally, 2.5758° at mid range, and 3.1506° peripherally. Subjects
`p00`, `p06`, and `p09` are harmed in the peripheral slice at every budget,
including `k=16`.

Ridge-affine calibration does not repair this failure mode. It shifts the
tradeoff toward the centre and is significantly worse than Huber in the
periphery at `k=3`, `k=5`, and `k=9`.

## Protocol

- subjects: MPIIFaceGaze `p00`–`p14`;
- evaluation unit: duplicate-safe unique observation group;
- eccentricity: angular target distance from the forward camera axis;
- fixed slices: `<=10°`, `10–20°`, and `>20°`;
- hybrid policy: soft-quality 0.25 plus MAP at `k=1`, random Huber offset at
  `k>=3`;
- affine comparator: the existing LOSO-tuned random ridge-affine draws;
- uncertainty: 100,000 paired subject-bootstrap replicates.

Each subject contributes approximately 394 central, 969 mid-range, and 395
peripheral evaluation groups on average. Subject means are aggregated with
equal subject weight.

## Hybrid-policy slice results

| k | Central `<=10°` | Mid `10–20°` | Peripheral `>20°` |
|---:|---:|---:|---:|
| 1 | 2.6988° | 2.9702° | 3.7076° |
| 3 | 2.8744° | 2.9007° | 3.4013° |
| 5 | 2.7183° | 2.7566° | 3.2853° |
| 9 | 2.5907° | 2.6260° | 3.1773° |
| 16 | **2.5289°** | **2.5758°** | **3.1506°** |

The uncalibrated slice means are 4.2256°, 4.5657°, and 5.0631° respectively.
Thus calibration provides a meaningful peripheral gain despite the higher
remaining absolute error.

At `k=16`, the hybrid gains are 1.6967° centrally (95% CI
1.2339–2.2143°), 1.9899° at mid range (1.3969–2.6076°), and 1.9125°
peripherally (1.1696–2.6703°). The aggregate peripheral gain therefore hides
three identities whose already-good peripheral predictions are worsened by a
global user offset.

## Ridge-affine comparison

Positive values below mean ridge has lower error than Huber:

| k | Central advantage | Mid advantage | Peripheral advantage |
|---:|---:|---:|---:|
| 3 | **0.1077°** | -0.0274° | **-0.1340°** |
| 5 | 0.0850° | -0.0185° | **-0.0928°** |
| 9 | 0.0627° | -0.0161° | **-0.0590°** |
| 16 | 0.0526° | -0.0028° | -0.0161° |

At `k=3`, the central ridge advantage has CI 0.0255–0.1876°. The peripheral
ridge disadvantages have CIs wholly below zero at `k=3` (-0.1813–-0.0895°),
`k=5` (-0.1527–-0.0325°), and `k=9` (-0.1055–-0.0120°). All `k=16`
slice differences are inconclusive.

The existing two-angle affine map is therefore not a general spatial
correction at few-shot budgets. Its shrinkage favours central accuracy while
the sparse calibration set does not constrain peripheral behaviour well
enough.

## Decision

- Keep the hybrid Huber policy as the simple default.
- Report target eccentricity as a mandatory robustness slice.
- Do not promote ridge-affine as a peripheral correction.
- Treat `p00`, `p06`, and `p09` as fixed failure-case subjects for qualitative
  analysis without tuning on them.
- A future spatial method must explicitly cover screen/target regions, use
  enough calibration points to constrain them, and be evaluated against the
  same peripheral slice.
- The next experiment should test stratified target-region coverage rather than
  increasing affine flexibility on randomly drawn samples.

## Threats to validity

- Eccentricity is relative to the camera forward axis, not physical screen
  distance from its centre.
- Screen geometry and head pose are unavailable in the local prediction file.
- Slice boundaries are fixed but dataset-specific.
- Random-draw averaging simulates calibration sets rather than a live stimulus
  sequence.
- The same 15 subjects were used in prior exploratory experiments.

## Reproduction

Hybrid slices:

```bash
python3 scripts/evaluate_task1_hybrid_robustness.py \
  --predictions outputs/task0_lopo_baseline/all_predictions.csv \
  --selection-summary outputs/task1_soft_quality_selection/selection_summary.json \
  --robust-results outputs/task1_robust_offset \
  --config configs/task1_hybrid_robustness.toml \
  --output-dir outputs/task1_hybrid_robustness
```

Ridge comparison:

```bash
python3 scripts/evaluate_task1_ridge_robustness.py \
  --predictions outputs/task0_lopo_baseline/all_predictions.csv \
  --ridge-results outputs/task1_ridge_affine \
  --hybrid-slice-results outputs/task1_hybrid_robustness \
  --config configs/task1_ridge_robustness.toml \
  --output-dir outputs/task1_ridge_robustness
```
