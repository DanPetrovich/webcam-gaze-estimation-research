# Task 1: soft quality-coverage selection

## Outcome

A soft selector was evaluated as an alternative to hard quality rejection. At
every selection step it combines normalized target-space coverage with the
leakage-safe image-quality score, without discarding any candidate frames.

The quality weight `0.25` is the useful operating point. In the one-shot case
it improves over random by **0.3776°** with mean offset (95% CI
0.0560–0.6746°) and by **0.3442°** with MAP (95% CI 0.1343–0.5711°). The
resulting errors are 3.2776° and 3.0797°, respectively.

Unlike the fixed top-50% gate, soft weight 0.25 shows no statistically
supported degradation at larger budgets. At `k=16` it reaches 2.7142° with
mean offset and 2.7208° with MAP, effectively matching random; the paired
advantages are -0.0019° and -0.0064°, with both confidence intervals spanning
zero.

The hypothesis that a moderate soft quality preference can preserve the
one-shot benefit without the large-budget harm is supported. The stronger
hypothesis that it improves every calibration budget is not supported.

## Protocol

- subjects: MPIIFaceGaze `p00`–`p14`;
- frozen inputs: Task 0 LOPO predictions and the duplicate-safe Task 1 split;
- budgets: `k in {1, 3, 5, 9, 16}`;
- calibrators: mean offset and MAP shrinkage;
- quality weights: 0.10, 0.25, 0.50, and 0.75;
- primary reference: the existing 100-draw random average;
- uncertainty: 100,000 paired subject-bootstrap replicates.

For the first sample, the selector combines target centrality and quality. For
later samples, it combines quality with the candidate's minimum angular target
distance from the already selected set. The coverage term is normalized to
`[0, 1]` at every step. Weight zero exactly reproduces target farthest-first;
weight one reduces to quality ranking.

Quality normalization uses the other 14 identities only. All candidates remain
eligible, so this method has no gate-induced reacquisition cost. The private
server returned selected `group_id` values only; images, labels, quality scores,
and per-frame features were not transferred.

## Fixed-weight results

MAP-offset subject means are:

| Weight | k=1 | k=3 | k=5 | k=9 | k=16 |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 3.1406° | **3.0566°** | 2.9471° | **2.7666°** | 2.7225° |
| 0.25 | **3.0797°** | 3.1555° | **2.9372°** | 2.7937° | **2.7208°** |
| 0.50 | 3.2164° | 3.2476° | 3.0363° | 2.9515° | 2.9304° |
| 0.75 | 3.3227° | 3.3794° | 3.0316° | 3.0307° | 3.0177° |

For weight 0.25, paired advantages over random are:

| k | Mean offset | 95% CI | MAP offset | 95% CI |
|---:|---:|---:|---:|---:|
| 1 | **0.3776°** | 0.0560–0.6746° | **0.3442°** | 0.1343–0.5711° |
| 3 | -0.1700° | -0.5474–0.1730° | -0.1419° | -0.4830–0.1740° |
| 5 | -0.0801° | -0.2831–0.1029° | -0.0603° | -0.2592–0.1291° |
| 9 | -0.0324° | -0.2492–0.1258° | -0.0219° | -0.2429–0.1457° |
| 16 | -0.0019° | -0.1431–0.1078° | -0.0064° | -0.1468–0.1054° |

Positive values mean lower error than random.

Weights 0.50 and 0.75 overemphasize quality and reduce coverage. Weight 0.75 is
significantly worse than random at `k=3`, `k=9`, and `k=16`; weight 0.50 is
significantly worse for MAP at `k=16`. This confirms that image cleanliness is
not a substitute for a representative calibration set.

## Comparison with hard gating

At one shot, weight 0.25 and the top-50% gate are statistically
indistinguishable: the soft selector is better by 0.0284° for mean and 0.0185°
for MAP, with both paired CIs including zero.

At `k=16`, the soft selector is better than the top-50% gate by 0.2310° for
mean (95% CI 0.1011–0.3807°) and 0.2290° for MAP (95% CI
0.0979–0.3796°). It removes the hard gate's confirmed large-budget harm.

The adaptive 90% gate remains numerically better than soft weight 0.25 at
`k=16` by 0.0477–0.0543°, but the paired intervals include zero. The soft
policy avoids its expected 11% reacquisition overhead.

## Leave-one-subject-out weight selection

For each held subject, the weight was selected using mean error on the other 14
subjects only.

- one-shot mean offset selected 0.25 in all 15 folds and retained a 0.3776°
  advantage over random (95% CI 0.0548–0.6739°);
- one-shot MAP selected 0.25 in 14/15 folds and 0.10 once; its 0.2769°
  advantage became inconclusive (95% CI -0.0156–0.5480°);
- `k=3` and `k=9` predominantly selected 0.10;
- `k=16` alternated between 0.10 and 0.25 and remained statistically
  indistinguishable from random.

This supports weight 0.25 as a stable one-shot mean-offset policy. It does not
support data-driven per-fold weight selection as a general improvement.

## Decision

- Prefer soft weight 0.25 over the hard 50% gate when one policy must support
  both tiny and larger calibration budgets.
- Keep MAP shrinkage as the best one-shot calibrator, while acknowledging that
  the LOSO weight-selection MAP comparison is inconclusive.
- Do not use weights 0.50 or 0.75.
- For `k>=3`, random sampling with robust residual offset remains the supported
  conservative default; soft quality selection is an accuracy-neutral fallback.
- The next experiment should test a prespecified hybrid: soft 0.25 for one shot
  and random plus robust offset for larger budgets, then move to calibration
  sample reliability or uncertainty rather than further threshold search.

## Threats to validity

- The four weights were compared on the same 15 subjects; fixed-weight findings
  beyond the LOSO check are exploratory.
- Quality was validated against synthetic corruptions, not naturally annotated
  blink, occlusion, or non-fixation.
- Static normalized crops omit live capture timing and temporal correlation.
- Known targets are used for coverage, so the selector applies to guided
  calibration rather than passive gaze frames.
- The paired bootstrap covers only 15 identities and one dataset.

## Reproduction

Server-side group selection uses `scripts/run_task1_quality_selection.py` with
`configs/task1_quality_selection.toml`. Local fixed-weight evaluation uses:

```bash
python3 scripts/evaluate_task1_quality_selection.py \
  --predictions outputs/task0_lopo_baseline/all_predictions.csv \
  --mean-offset-results outputs/task1_mean_offset \
  --selection-summary outputs/task1_soft_quality_selection/selection_summary.json \
  --config configs/task1_quality_selection.toml \
  --output-dir outputs/task1_soft_quality_selection
```

LOSO weight policy:

```bash
python3 scripts/evaluate_task1_soft_quality_policy.py \
  --quality-results outputs/task1_soft_quality_selection \
  --mean-offset-results outputs/task1_mean_offset \
  --config configs/task1_quality_selection.toml \
  --output-dir outputs/task1_soft_quality_policy_loso
```
