# Task 1: target-diversity selection baseline

## Outcome

Deterministic target-space k-center selection completed for all 15 subjects,
five calibration budgets, and both mean/MAP offset calibration. It reduces the
candidate-pool covering radius as intended, but **does not improve over random
selection at any tested budget**: every paired subject-bootstrap confidence
interval for target-diversity versus random includes zero.

The best target-diversity result is 16-shot MAP at **2.6997°**, compared with
2.7145° for the paired random MAP baseline. The apparent 0.0148° advantage has
a 95% CI of **-0.0804–0.1165°**. For mean offset at `k=16`, the result is
2.7024° versus 2.7123° random, a 0.0099° difference with CI
**-0.0874–0.1104°**.

## Hypothesis

At a fixed calibration budget, spreading selected samples across known gaze
targets may estimate user bias more reliably than redundant random samples.

The current result does not support that hypothesis for a constant offset
calibrator. A plausible explanation is that an offset needs a representative
estimate of subject bias rather than geometric coverage of extreme targets;
quality and residual stability may therefore matter more than target coverage
alone. This explanation remains a hypothesis until quality-aware selection is
tested.

## Protocol

- frozen Task 0 predictions and duplicate-safe temporal split;
- candidate/guard/evaluation fractions: 20%/10%/70%;
- subjects: MPIIFaceGaze `p00`–`p14`;
- budgets: `k in {1, 3, 5, 9, 16}`;
- calibrators: mean offset and leave-one-subject-out MAP shrinkage;
- initial sample: candidate closest to the candidate target mean;
- subsequent samples: farthest-first traversal using angular distance between
  known target gaze vectors;
- references: the existing 100-draw random average and fixed-uniform selector;
- uncertainty: 100,000 paired subject-bootstrap replicates.

Known target positions are legitimate inputs in guided calibration because the
system controls the displayed stimulus. They would not be available for passive
selection from unlabeled webcam frames.

## Aggregate results

| k | Mean offset | MAP offset | Mean target advantage vs random | 95% CI |
|---:|---:|---:|---:|---:|
| 1 | 3.5201° | **3.3331°** | 0.1351° | -0.3294–0.5730° |
| 3 | 3.1400° | **3.1119°** | -0.0980° | -0.4027–0.1750° |
| 5 | 3.0133° | **2.9992°** | -0.1308° | -0.3920–0.0779° |
| 9 | 2.8020° | **2.7991°** | -0.0317° | -0.1734–0.0910° |
| 16 | 2.7024° | **2.6997°** | 0.0099° | -0.0874–0.1104° |

The advantage column compares target diversity plus mean offset against the
paired random mean-offset reference; positive values favor target diversity.
MAP gives the same substantive conclusion.

At `k=16`, target diversity is better than fixed-uniform by 0.1059° for mean
offset (95% CI 0.0189–0.2037°) and 0.0996° for MAP (95% CI
0.0148–0.1915°). This only rejects the earlier fixed temporal heuristic; it
does not establish an improvement over the stochastic reference selector.

## Decision

- Keep random selection as the primary selector reference.
- Do not claim a target-diversity win.
- Retain target k-center as the diversity component for a later
  `quality_then_diversity` method.
- Add a validity/reliability signal before testing the combined selector.
- Keep MAP for one-shot comparisons and mean offset as the simplest default for
  larger budgets.

## Threats to validity

- The target distribution in the candidate pool may not correspond to an
  optimized real calibration grid.
- K-center selection emphasizes geometric extremes and does not measure blur,
  occlusion, landmark failure, head pose, or feature redundancy.
- Temporal order is approximated by minimum saved sample index.
- Results apply to a constant output-angle correction and may differ for an
  affine or feature-space calibrator.

## Reproduction

```bash
python3 scripts/run_task1_target_diversity.py \
  --predictions outputs/task0_lopo_baseline/all_predictions.csv \
  --mean-offset-results outputs/task1_mean_offset \
  --config configs/task1_target_diversity.toml \
  --output-dir outputs/task1_target_diversity
```

Outputs include the selected sample manifest, per-subject and aggregate metrics,
paired comparisons, and the resolved summary.
