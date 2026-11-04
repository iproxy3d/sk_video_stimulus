# Calculated example data

| File | Meaning |
|---|---|
| `video_features.csv` | raw output for `input_25fps.mp4` |
| `neutral_features.csv` | generated raw output used to build the neutral profile; omitted because the compact JSON profile is sufficient |
| `normalized_features.csv` | calibrated example output |
| `test_features.csv` | calibrated output for `test_25fps.mp4` |

Rows before `ready=true` belong to the rolling-window warm-up. Normalized columns are empty during this interval by design.
