# Data Format Specification

## Supported Formats

### 1. JSON (prediction-data-couchbase format)

**File:** `logs/prediction-data-couchbase.json`

```json
{
  "prediction_data": [
    {
      "type": "FN",
      "uuid": "f8b4c9bd-3e45-482e-b22e-cb8687625d18",
      "watch_accelerometer_x": "-1.729,-1.729,-1.729,...",
      "watch_accelerometer_y": "-5.822,-5.822,-5.822,...",
      "watch_accelerometer_z": "7.465,7.465,7.465,...",
      "watch_gyroscope_x": "0.530,0.530,0.530,...",
      "watch_gyroscope_y": "-1.647,-1.647,-1.647,...",
      "watch_gyroscope_z": "-1.808,-1.808,-1.808,...",
      "probability": 0.4693,
      "label": "Fall"
    }
  ]
}
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| type | string | FN, FP, TP, TN (error classification) |
| uuid | string | Device/user identifier |
| watch_accelerometer_x | string | 128 comma-separated values (m/s²) |
| watch_accelerometer_y | string | 128 comma-separated values |
| watch_accelerometer_z | string | 128 comma-separated values |
| watch_gyroscope_x | string | 128 comma-separated values (rad/s) |
| watch_gyroscope_y | string | 128 comma-separated values |
| watch_gyroscope_z | string | 128 comma-separated values |
| probability | float | Original model prediction [0, 1] |
| label | string | Ground truth: "Fall" or "ADL" |

### 2. JSON Lines (.jsonl)

One record per line:
```
{"uuid": "...", "watch_accelerometer_x": "...", ...}
{"uuid": "...", "watch_accelerometer_x": "...", ...}
```

### 3. Parquet

**Schema:**
```python
{
    'uuid': str,
    'timestamp_ms': int64,
    'acc': object,   # np.ndarray (128, 3)
    'gyro': object,  # np.ndarray (128, 3)
    'label': str,
    'probability': float64,  # optional
}
```

## Data Requirements

### Window Size
- **128 samples** per window
- At 31.25 Hz = ~4.1 seconds

### Accelerometer
- Units: m/s²
- Axes: X, Y, Z (device frame)
- Range: typically ±40 m/s²

### Gyroscope
- Units: rad/s (convert from deg/s if needed)
- Axes: X, Y, Z (device frame)
- Range: typically ±10 rad/s

### Labels
- **"Fall"** or **"fall"** → ground_truth = 1
- **"ADL"** or anything else → ground_truth = 0

## Session Grouping

For alpha queue simulation, windows are grouped into sessions by:
1. Same UUID
2. Timestamp gap < 10 seconds

Windows within a session are processed sequentially through the alpha queue.

## Collecting New Data

### From Watch (Couchbase)

Data in Couchbase bucket `smart-fall-data`:
- Each document is a window prediction
- Contains raw sensor arrays and model output

Export using Couchbase CLI or SDK:
```python
from couchbase.cluster import Cluster

cluster = Cluster('couchbase://host')
bucket = cluster.bucket('smart-fall-data')
result = bucket.n1ql_query('SELECT * FROM data')
```

### From Watch (Direct Logging)

If collecting directly, ensure:
- 128 samples per window
- Include UUID for session tracking
- Include timestamp for Kalman dt calculation
- Label as "Fall" or "ADL"

## Validation

Check data with:
```python
from testing.data.loader import TestDataLoader

loader = TestDataLoader('data.json')
windows = loader.load_windows()
print(loader.get_stats())
# {'total_windows': N, 'fall_windows': M, ...}
```
