# Live webcam calibration protocol scaffold

## Current scope

The repository now contains a deterministic, normalized screen-stimulus plan
for the next independent validation stage. It intentionally does not open a
camera, record a face, run inference, or retain biometric data.

The initial protocol is a 3x3 grid with two repetitions, a 10% screen margin,
500 ms settle time, and 1,500 ms capture window per stimulus. The 18 stimuli
have a planned duration of 36 seconds before transitions or breaks.

This is a scaffold, not a completed live-webcam experiment. Timing, point count,
margin, order, and repetitions remain experimental variables and must be
reported with any result.

## Coordinate convention

- normalized screen origin: top left;
- normalized x: increases right, range `[0, 1]`;
- normalized y: increases down, range `[0, 1]`;
- pixel conversion: `(x * (width_px - 1), y * (height_px - 1))`;
- physical conversion: top-left millimetres using the measured active display
  width and height.

The plan remains normalized because actual pixel and physical dimensions are
device-specific runtime metadata. They must not be guessed from nominal laptop
model names.

## Required capture metadata

Before a real run, the capture layer must record beside the ignored local
artifact:

- consent/protocol identifier without the participant's name;
- camera identifier, resolution, nominal and measured FPS;
- active display pixel dimensions and measured physical dimensions;
- camera position relative to the display when defensibly measurable;
- stimulus manifest hash and timestamps;
- model/checkpoint hash, preprocessing version, and calibration policy;
- dropped frames, face/landmark failures, accepted frames, and latency stages;
- separate calibration and test sequence identifiers.

## Privacy and retention

- Do not commit frames, video, face crops, landmarks, or checkpoints.
- Do not start capture without explicit participant consent and a retention
  decision.
- Prefer derived predictions and aggregate metrics over retained raw video.
- If raw capture is temporarily necessary, store it only under an ignored local
  data directory with a documented deletion time.
- Never send capture artifacts through the GitHub relay.

## Experimental safeguards

- Calibration and test stimuli must be disjoint sequences even when positions
  repeat.
- Compare uncalibrated and calibrated predictions on exactly the same test
  frames.
- Report results per position and by target eccentricity, not only pooled.
- Measure calibration duration and accepted/rejected frames.
- Keep the current simple hybrid policy frozen for the first live run.
- Do not tune thresholds or region boundaries on the final test sequence.

## Build the manifest

```bash
PYTHONPATH=src python3 scripts/build_webcam_calibration_plan.py \
  --config configs/webcam_calibration_9point.toml \
  --output outputs/webcam_calibration/plan.json
```

The output contains stimulus coordinates and timing only.

## Remaining implementation boundary

The following components are deliberately not yet implemented because they
require actual device and consent context:

- full-screen stimulus presentation;
- camera acquisition and synchronized timestamps;
- face/landmark preprocessing and model checkpoint loading;
- camera-to-screen geometry or point-of-regard mapping;
- raw-data retention and deletion automation;
- real-time visualization and latency profiling.
