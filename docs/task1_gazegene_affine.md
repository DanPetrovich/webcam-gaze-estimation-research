# GazeGene few-shot affine calibration

## Execution status

The server evaluation is pending. The research code is pinned and the real-data
operation runs a reduced-draw smoke pass before the full comparison. No
scientific result is available while the self-hosted runner is offline.

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
The relay verifies the checkpoint and research file hashes before running.
Only aggregate metrics, provenance hashes and configuration leave the server.
