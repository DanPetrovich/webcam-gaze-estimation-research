# GazeGene few-shot affine calibration

## Execution status

Completed on 2026-11-27. The two-draw real-data smoke took 9 s and the full
100-draw comparison took 26 s. The evaluation ran on Windows with Python 3.11.9 and NumPy
2.3.5. The smoke and full reports used the same inputs:

- validation prediction SHA-256 `fb63347dd9104e89d37a46b3bb581c26d9165c63b37f59265b5e5ba92646c36a`;
- test prediction SHA-256 `94adfa6d19c887f6c42d7ce37cff891b25a257543d363750e62d478a3f7e59ba`.

## Results

All results are subject-macro mean angular errors. The uncalibrated mean is
4.8534°. The rerun reproduced the earlier mean, MAP and Huber scores on the
same evaluation frames.

| Shots | Selected λ | Mean | MAP | Huber | Ridge-affine |
|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 6.9124° | 4.8113° | 6.9124° | 4.8260° |
| 3 | 10 | 5.6289° | 4.7166° | 5.0908° | 4.7239° |
| 5 | 10 | 5.2403° | 4.6498° | 4.7681° | 4.6768° |
| 9 | 10 | 4.9387° | 4.5833° | 4.5485° | 4.6482° |
| 16 | 1 | 4.7499° | 4.5221° | 4.4497° | 4.5066° |

At 16 shots, affine improves on the uncalibrated baseline by 0.3468° (paired
95% CI 0.1079–0.6030°) and on mean offset by 0.2433° (0.1289–0.3751°).
Its advantage over MAP is 0.0155° (−0.1721 to 0.2363°). Its advantage over
Huber is −0.0569° (−0.2231 to 0.1461°). At three shots, affine beats Huber
by 0.3669° (0.1601–0.5654°), while its comparison with MAP is inconclusive.
At 16 shots affine worsens two of ten identities against the uncalibrated
baseline and seven of ten against Huber after averaging each identity's draws.

| Test slice at 16 shots | Rows | Uncalibrated | Huber | Ridge-affine |
|---|---:|---:|---:|---:|
| <20° | 19,399 | 3.3089° | 3.0818° | 2.9634° |
| 20–<45° | 54,189 | 3.9957° | 3.6235° | 3.6466° |
| ≥45° | 52,412 | 5.9655° | 5.4847° | 5.5967° |

At ≥45° and 16 shots, the affine gain versus uncalibrated is 0.3688°
(0.0845–0.6817°). Its advantage over Huber is −0.1120°
(−0.2771 to 0.0860°). At nine shots extreme-angle Huber outperforms affine
by 0.1382° (0.0352–0.2524°). At both budgets the Huber point estimate is
lower in the extreme slice.

The hypothesis of an advantage over the strongest constant-offset references
is unresolved at 16 shots overall. The tested affine fit improves on leaving
the model uncalibrated, and the result does not support replacement of MAP or
Huber as the reference. Extreme-slice errors measure angular accuracy and
cannot establish whether the affine map repaired systematic radial bias.

## Hypothesis

An identity-regularized two-angle affine map may correct changes in scale and
cross-axis bias that a constant user offset leaves behind. The main comparison
is against Huber and MAP at 16 calibration examples. The fixed ≥45° gaze
slice tests whether the effect extends to extreme directions.

## Protocol

Use the frozen epoch-11 GazeGene MAE-B16 validation and test predictions.
The checkpoint SHA-256 is
`d24c6cee737cd12dbc66264538d28d8d4fd8a4e71caca69a24fd79d2e906a87b`.
Validation identities are 41–46 and test identities are 47–56. Fit each
personalization map separately for each subject and camera. Use frames 0–399
as calibration candidates, 400–599 as a guard region, and 600–1999 for the
final test. Draw the same `k=1,3,5,9,16` candidate indices as the preceding
constant-offset experiment with seed `20261118`. Use 100 paired random draws
per camera and budget.

The affine model maps predicted `[pitch, yaw]` to adjusted `[pitch, yaw]`
with a 2×2 matrix and intercept. Fit mean squared error in angular coordinates
scaled by 30° with ridge regularization toward the identity matrix and zero
intercept. Search `λ = 0.001, 0.01, 0.1, 1, 10, 100` per budget using 20
random calibration draws and only validation guard targets for scoring.
The Huber delta is fixed to the previous validation choice of 10° for
`k=1` and 3° for larger budgets. Refit the MAP prior on the validation
candidate windows. Test labels contribute only through the allowed `k)
calibration samples and final metric.

Evaluate exact spherical angular error on the same 126,000 late test rows for
each method. Average within camera, then across cameras within identity, then
across test identities. The fixed target-angle slices are <20°, 20–<45°,
and ≥45°. Bootstrap the ten paired identity scores with 100,000 replicates.
The slice and multi-budget intervals are descriptive with no multiplicity
adjustment. The 1-shot ridge fit is mathematically defined by regularization
but its slope terms have little data support. The parameterization uses
ordinary pitch/yaw coordinates and can be sensitive near the yaw wrap.

`scripts/run_task1_gazegene_affine.py` runs as a module with `--validation`,
`--test`, `--protocol`, `--config` and `--output` on Python 3.11+ with NumPy.
The research source revision is `7fc2b7eb1b089c01c5d6f06f7b677b166049a087`.
The run verified the checkpoint and research file hashes before evaluation.
Public artifacts contain aggregate metrics, provenance hashes and configuration.
