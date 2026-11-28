# Task 0: 15-fold LOPO reference baseline

## Outcome

Task 0 is complete as a **reference uncalibrated baseline**, not as the final
deployment model. All 15 identity-disjoint folds completed successfully on
MPIIFaceGaze. Each fold used 13 subjects for training, the next subject for
validation/checkpoint selection, and one unseen subject for the one-time test.
Every test fold contains 3,000 samples, for 45,000 predictions in total.

The primary result is the equally weighted mean of the 15 subject-level mean
angular errors:

| Metric | Result |
|---|---:|
| Subject mean angular error | **4.5799°** |
| 95% subject-bootstrap CI | **3.9353–5.2072°** |
| Median of subject means | 4.5202° |
| Subject-level standard deviation | 1.3057° |
| Subject-level p90 / p95 | 6.2480° / 6.3347° |
| Pooled sample median | 4.3535° |
| Pooled sample p90 / p95 | 7.4188° / 8.3823° |
| Constant median-angle baseline | 11.3339° |
| Paired absolute gain over constant | 6.7540° (95% CI 5.8981–7.5980°) |
| Relative gain over constant | 59.59% |
| Subjects better than constant | 15 / 15 |

The subject-level bootstrap resamples the 15 held-out identities 100,000 times
with seed 42. It is the inferential result. Pooled sample statistics are only
descriptive because frames from the same identity are not independent.

## Per-subject results

| Test | Validation | Best epoch | Mean | Median | p95 | Constant |
|---|---|---:|---:|---:|---:|---:|
| p00 | p01 | 3 | 2.2650° | 2.1135° | 4.3069° | 11.2436° |
| p01 | p02 | 4 | 3.5907° | 3.4322° | 6.2463° | 10.7334° |
| p02 | p03 | 4 | 5.4338° | 5.4394° | 7.9871° | 10.4986° |
| p03 | p04 | 4 | 5.3658° | 5.1349° | 8.5051° | 14.8891° |
| p04 | p05 | 2 | 4.2527° | 4.2462° | 6.8661° | 12.0463° |
| p05 | p06 | 4 | 4.5202° | 4.4429° | 7.0980° | 11.0384° |
| p06 | p07 | 2 | 2.5166° | 2.3288° | 5.0607° | 11.4733° |
| p07 | p08 | 4 | 5.1775° | 4.5985° | 10.5756° | 9.9457° |
| p08 | p09 | 4 | 5.4222° | 4.8398° | 10.0553° | 12.0196° |
| p09 | p10 | 3 | 3.8059° | 3.5825° | 7.3724° | 10.4413° |
| p10 | p11 | 2 | 6.3229° | 6.0776° | 10.7344° | 10.5915° |
| p11 | p12 | 4 | 4.0633° | 3.8157° | 7.6854° | 9.8480° |
| p12 | p13 | 2 | 6.1356° | 6.1429° | 9.2225° | 10.0406° |
| p13 | p14 | 3 | 3.4633° | 3.3962° | 5.9664° | 11.5015° |
| p14 | p00 | 4 | 6.3624° | 6.3720° | 9.1434° | 13.6973° |

Eight folds selected epoch 4, four selected epoch 2, and three selected epoch
3. Seven folds therefore had not degraded by the four-epoch limit. For the
other eight, the validation increase from the selected epoch to epoch 4 was
small to moderate (0.043–0.364°). The full LOPO run does not support the earlier
claim that every fold exhibits immediate catastrophic overfitting; it supports
early stopping and shows that optimal training duration varies by validation
identity.

## Configuration and execution

All folds used the same effective training configuration except for the
explicit test-subject list and the derived `config_hash`:

- UniGaze MAE-B16 with all parameters trainable;
- official normalized MPIIFaceGaze input cached as 3,000 samples per subject;
- `AdamW`, backbone learning rate `3e-6`, head learning rate `3e-5`;
- weight decay `1e-4`;
- 5% warmup followed by cosine decay;
- at most 4 epochs, early-stopping patience 2;
- batch size 8, AMP enabled, seed `20261004`;
- checkpoint selected only on the fold's validation subject;
- test labels never used for optimization or checkpoint selection.

The dataset manifests have different hashes because Kaggle assigned different
input file modification timestamps. After removing path and timestamp fields,
all subject entries have identical sample counts, byte sizes, and tensor shapes.
The shard A/B notebook source hash is identical. Shard C used a separate
dual-GPU worker wrapper, but its resolved training fields match A/B.

Source executions:

- [Kaggle shard A, p00–p04](https://www.kaggle.com/code/daniilpet/task0-unigaze-b16-lopo-shard-a?scriptVersionId=366263194), runtime 4:00:08;
- [Kaggle shard B, p05–p09](https://www.kaggle.com/code/daniilpet/task0-unigaze-b16-lopo-shard-b?scriptVersionId=366263803), runtime 4:09:37;
- [Kaggle shard C, p10–p14](https://www.kaggle.com/code/daniilpet/task0-unigaze-b16-lopo-shard-c?scriptVersionId=366324250), runtime 3:21:49.

## Hard-subject analysis

The hardest identities are `p14` (6.3624°), `p10` (6.3229°), and `p12`
(6.1356°). Their target-angle magnitudes are not consistently more extreme than
those of the other subjects: mean magnitudes are 15.73°, 14.87°, and 12.82°.
Across all samples, error increases with target-angle magnitude, but only
moderately:

| Target-angle magnitude proxy | Samples | Mean error |
|---|---:|---:|
| <10° | 10,195 | 4.3118° |
| 10–20° | 24,706 | 4.5170° |
| >=20° | 10,099 | 5.0042° |

The hard identities instead exhibit large signed residuals:

- `p14`: mean yaw residual −5.52°;
- `p10`: mean pitch residual +6.02°;
- `p12`: mean pitch residual +4.44° and yaw residual −3.10°.

This is evidence of identity-specific directional bias and motivates a simple
person-specific mean-offset/MAP calibration as the next baseline. It does not
prove the physical cause of the bias. The saved prediction tables contain no
illumination, glasses, blur, landmark-quality, or head-pose fields, so those
factors require a separate annotated or proxy-based audit.

## Interpretation and decision

The 4.58° result is frozen as the Task 0 **reference baseline** because it is a
complete, reproducible, identity-disjoint measurement against which later
methods can be compared. “Baseline” does not mean “acceptable final quality” or
“state of the art.” The quality is insufficient for precise point-of-regard use
without calibration, especially for the hard identities and the p90/p95 tail.

The next experiment must keep these exact test predictions fixed and report,
for each calibration budget `k`, the paired post-calibration change on the same
subjects and held-out samples. The first method is mean offset with shrinkage;
it directly targets the observed signed biases before any MAML, polynomial
mapping, or uncertainty-aware selection is added.

Do not improve the reported baseline by excluding hard identities, choosing a
different checkpoint using test labels, or silently replacing failed folds.
Any stronger uncalibrated model is a new baseline version and must be evaluated
under the same 15-fold protocol.

## Reproduction and artifacts

The aggregation command is:

```bash
python3 scripts/aggregate_task0_lopo.py \
  --run-dir <shard-a-run-dir> \
  --run-dir <shard-b-run-dir> \
  --run-dir <shard-c-combined-dir> \
  --output-dir outputs/task0_lopo_baseline \
  --bootstrap-replicates 100000 \
  --seed 42
```

Generated artifacts (ignored by Git because they include per-sample results):

- `summary.json`;
- `per_subject_metrics.csv`;
- `all_predictions.csv`;
- `all_history.csv`;
- `hard_subject_analysis.csv`;
- `error_by_target_magnitude.csv`;
- `per_subject_error.svg`.

## Threats to validity

- MPIIFaceGaze is a single dataset and capture domain; this is within-dataset,
  static-frame evaluation, not cross-camera or real-webcam validation.
- The validation identity changes between folds, so checkpoint-selection noise
  remains part of the measured procedure.
- Samples are upsampled to 3,000 per subject; the reported `samples` count is not
  the number of unique raw frames for every identity.
- A constant median-angle predictor is only a sanity baseline. A fair claim of
  competitiveness requires comparison with published methods under a matched
  protocol and with a frozen linear-probe baseline.
- Angular error alone does not establish reliable fixation, saccade, or screen
  point estimation.
