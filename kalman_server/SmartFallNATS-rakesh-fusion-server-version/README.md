# SmartFallNATS

**SmartFallNATS** is a real-time, end-to-end **wearable fall detection system** built on **NATS.io** for low-latency communication between a **Wear OS smartwatch** and a **server-side deep learning inference pipeline**.

This repository implements an experimental infrastructure for **personalized, multi-user fall detection**, emphasizing **efficient edge–cloud communication**, **binary data pipelines**, and **concurrent model serving**.

---

## Components

The system integrates the following components:

- **Wear OS Watch App**  
  Sensor data acquisition, local feedback, and inference request generation.

- **Android Phone App**  
  User onboarding, configuration, and system state management.

- **NATS.io Messaging Layer**  
  Subject-based routing with request–reply semantics for scalable, low-latency inference.

- **Server-Side Inference Pipeline**  
  A Kalman-filter–enhanced Transformer model for real-time fall probability estimation.

---

## System Architecture

### Wear OS Watch App

**Responsibilities**
- Collect accelerometer and gyroscope data
- Form fixed-size temporal windows (128 × 3)
- Encode sensor windows using a compact binary format (`InferenceBinary`)
- Publish inference requests via NATS.io
- Display real-time fall probability
- Accept manual *FELL* feedback for missed detections
- Persist inference results and sensor metadata locally (Couchbase Lite)
- Periodically synchronize data with cloud storage

---

### Android Phone App

**Responsibilities**
- Support user profile creation
- Manage system activation state (Activated / Deactivated)
- Coordinate watch-side configuration

---

## Binary Payload Format (SFN1)

Inference requests are transmitted as compact binary messages to minimize latency and overhead.

| MAGIC | VER | FLAGS | RSVD | TS | FS | UUID_LEN | UUID | ACC | GYRO |


**Key Properties**
- MAGIC: `SFN1`
- Accelerometer and gyroscope values stored as `int16`
- Scaling factor: `×1000`
- Fixed window size: `128 × 3`
- Big-endian encoding

**Design Rationale**
- Eliminates JSON parsing on the critical path
- Reduces garbage collection pressure on wearable devices
- Improves throughput and latency determinism under concurrent load

---

## Server: Kalman + Transformer Pipeline

**Entry point:**  
`KalmanFusionServer/server_kalman_transformer_nats.py`

**Processing Pipeline**
1. Decode binary payload
2. Apply Kalman filtering for orientation estimation (roll, pitch, yaw)
3. Construct feature vector:  
   `[SMV, ax, ay, az, roll, pitch, yaw]`
4. Normalize features using a trained `StandardScaler`
5. Perform Transformer-based inference
6. Return fall probability via NATS reply subject

**Concurrency Model**
- CPU-bound inference executes in thread pool executors
- NATS event loop remains asynchronous and non-blocking
- Supports concurrent inference streams across multiple UUIDs

---

## Running the Server

```bash
# Default config
python KalmanFusionServer/server.py

# Specific config
python KalmanFusionServer/server.py --config KalmanFusionServer/configs/s8_16_kalman_gyromag_norm.yaml
```

---

## Testing & Analysis Framework

Standalone offline testing platform for evaluating fall detection models using pre-collected data.

**Location:** `FusionTransformer/testing/` (requires FusionTransformer repo)

### Features

- **Exact server replication**: Same preprocessing pipeline as production
- **Multiple interfaces**: Streamlit web UI, CLI, Python API
- **Comprehensive metrics**: F1, precision, recall, ROC-AUC, PR-AUC, confusion matrix
- **Alpha queue simulation**: Replicates watch-side decision logic
- **Error analysis**: FN/FP breakdown, per-subject analysis
- **Interactive visualization**: Signal plots, probability distributions

### Quick Start

#### Web Interface (Streamlit)

```bash
# From FusionTransformer root
streamlit run testing/app/main.py --server.address 0.0.0.0 --server.port 8501

# Access via browser
# Local: http://localhost:8501
# Remote (VPN): http://<server-ip>:8501
```

#### CLI

```bash
# Run single config
python -m testing.cli run \
    --config KalmanFusionServer/configs/s8_16_kalman_gyromag_norm.yaml \
    --data KalmanFusionServer/logs/prediction-data-couchbase.json

# Compare multiple configs
python -m testing.cli compare \
    --configs "KalmanFusionServer/configs/*.yaml" \
    --data KalmanFusionServer/logs/prediction-data-couchbase.json

# Launch web app
python -m testing.cli app --port 8501
```

#### Python API

```python
from testing.services import InferenceService, AnalysisService
from testing.services.inference_service import DataLoadRequest, InferenceRequest

# Load data
svc = InferenceService()
data = svc.load_data(DataLoadRequest(path="data.json"))

# Run inference
predictions = svc.run_inference(InferenceRequest(
    windows=data.windows,
    config_path="config.yaml",
    threshold=0.5
))

# Analyze
analysis = AnalysisService().analyze(AnalysisRequest(
    predictions=predictions.predictions,
    windows=data.windows
))
print(f"F1: {analysis.metrics['f1']:.4f}")
```

### Web UI Dashboard

The Streamlit interface provides dropdown selectors for configuration:

| Selector | Options |
|----------|---------|
| **Preprocessing** | kalman_gyromag (7ch), kalman_yaw (7ch), raw_gyro (7ch), raw_gyromag (5ch) |
| **Normalization** | norm (acc_only), nonorm |
| **Stride** | s8_16 (8/16), s16_32 (16/32) |

Config files are built from selections: `{stride}_{preprocessing}_{normalization}.yaml`

**Analysis Options:**
- Metrics (F1, Precision, Recall, Accuracy, Specificity)
- Confusion Matrix
- ROC Curve
- PR Curve
- Threshold Sweep
- Error Analysis (FN/FP breakdown)

### Architecture

```
testing/
├── data/           # Data loading and schema
├── analysis/       # Metrics and error analysis
├── inference/      # Pipeline and alpha queue simulation
├── visualization/  # Plotly charts
├── services/       # Framework-agnostic business logic
├── app/            # Streamlit web app
│   └── pages/      # Dashboard, Compare, Errors, Signals
├── api/            # FastAPI endpoints (future)
└── cli.py          # Command-line interface
```

**Service Layer Design:**

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Streamlit  │  │   FastAPI   │  │     CLI     │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
              ┌─────────▼─────────┐
              │   Service Layer   │
              │ (InferenceService │
              │  AnalysisService) │
              └─────────┬─────────┘
                        │
       ┌────────────────┼────────────────┐
       │                │                │
┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
│    Data     │  │  Analysis   │  │  Inference  │
│   Loaders   │  │   Metrics   │  │  Pipeline   │
└─────────────┘  └─────────────┘  └─────────────┘
```

### Alpha Queue Simulation

Replicates watch-side decision logic:

1. Beta queue: 128 frames → 1 window prediction
2. Alpha queue: Collect 10 predictions
3. Average probability > 0.5 → FALL (flush queue)
4. Average probability ≤ 0.5 → ADL (retain last 5)

### Remote Access

**Via VPN:**
```bash
streamlit run testing/app/main.py --server.address 0.0.0.0
# Access: http://<server-ip>:8501
```

**Via SSH Tunnel:**
```bash
ssh -L 8501:localhost:8501 user@server
# Access: http://localhost:8501
```

### Dependencies

```
streamlit>=1.30.0
plotly>=5.18.0
pandas>=2.0.0
numpy
scikit-learn>=1.3.0
torch>=2.0.0
```

## NATS Configuration

NATS.io serves as the core communication backbone, enabling scalable and low-latency interaction between wearable clients and server-side inference services.

**NATS Server:**  
`nats://chocolatefrog@cssmartfall1.cose.txstate.edu:4224`

Note: NATS.io as service is already running on the cssmartfall1.cose.txstate.edu at port 4224.

**Kalman Transformer Subject:**  
`m.kalman_transformer.<uuid>`

## Subject Design

- Subjects act as logical namespaces for routing inference requests.

- <uuid> uniquely identifies a user or device stream.

- Model-specific prefixes (e.g., kalman_transformer, knowledge_distillation) allow multiple inference servers to coexist on the same NATS instance and port.

This subject hierarchy enables:

- Parallel deployment of multiple model variants.

- Clean separation of inference pipelines.

- Fine-grained control over routing and scalability.

---

## Messaging Semantics

- Request–reply pattern for synchronous inference.

- Each request includes a reply subject for returning predictions.

- Ensures deterministic request–response pairing per UUID.

---

## Transport & Reliability

- TCP with TLS enabled.

- Client-side auto-reconnect supported (watch and phone).

- Stateless server design allows safe reconnections and restarts.

---

## Working flow from WearOS App to the server

During real-time testing, the smartwatch collects data from both the accelerometer and the gyroscope. The accelerometer measures acceleration in meters per second squared(m/s^2), including the effect of gravity(g), while the gyroscope measures angular velocity in radians per second(rad/s).

If the magnitudes of both the accelerometer (exceeding 20.0f) and gyroscope (exceeding 5.0f) surpass predefined thresholds, then sensor's data are compressed to Binary-encoded payload and transmitted to the server for model inference.

We used NATS.io messaging framework to facilitates communication between the smartwatch and the server-based model. On the cloud, Kalman filtering preprocesses the sensor data ([[ax, ay, az], [gx, gy, gz]]) into features such as [SMV, ax, ay, az, roll, pitch, and yaw]. The processed data are then used for model inference, and the results are transmitted back to the smartwatch to determine fall prediction.

On the smartwatch, we used two queues to organize the sensor data, an Alpha queue of size 10, with each entry consisting of a Beta queue of size 128. Each Beta queue containing 128 samples is sent to the server for prediction, and the Alpha queue is updated by sliding one sample in the last Beta queue. Once the Alpha queue is full, the average of each prediction of a Beta queue is calculated to determine a fall/no fall event. If no fall event is detected, the latest 5 beta queue predictions in the Alpha queue are retained by discarding the oldest five values, the Alpha queue is then populated with new values from the Beta queue for the next prediction. Retaining some values from previous predictions helps prevent missed fall events (false negatives). In the event of a fall detection, the entire Alpha queue is discarded to avoid generating false positives after a true fall event.

---

## Research Context

This repository is part of the SmartFall research project at Texas State University, investigating:

- Personalized fall detection on wearables.

- Edge–cloud collaborative intelligence.

- Low-latency ML/DL inference under resource constraints.

- Concurrent, multi-user model serving using subject-based messaging.

- The system serves as a research platform for evaluating architectural trade-offs in real-time, wearable AI systems.

---

## Contact

SmartFall Research Team

Texas State University
