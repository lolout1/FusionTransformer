# Cross-Modal Knowledge Distillation for Fall Detection

Knowledge distillation from skeleton teacher to IMU student, with alignment-free methods robust to irregular timestamps.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Skeleton Data (32 joints × 3 xyz)                          │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────┐                                        │
│  │ SkeletonTransformer (Teacher)                            │
│  │  - JointEmbedding                                        │
│  │  - TransformerEncoder                                    │
│  │  - SqueezeExcitation                                     │
│  │  - TemporalAttentionPooling                              │
│  └────────┬────────┘                                        │
│           │ features_T (B, embed_dim)                       │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────┐                    │
│  │         KD LOSSES                   │                    │
│  │  - EmbeddingKDLoss (cosine)         │◄────┐              │
│  │  - GramKDLoss (structural)          │     │              │
│  │  - COMODOLoss (distribution)        │     │              │
│  └─────────────────────────────────────┘     │              │
│                                              │              │
│  IMU Data (6ch: acc + gyro)                  │              │
│         │                                    │              │
│         ▼                                    │              │
│  ┌─────────────────┐                         │              │
│  │ TimestampAwareStudent                     │              │
│  │  - EventTokenResampler ◄── Novel!         │              │
│  │  - TransformerEncoder                     │              │
│  │  - SqueezeExcitation                      │              │
│  │  - TemporalAttentionPooling               │              │
│  └────────┬────────┘                         │              │
│           │ features_S ──────────────────────┘              │
│           │                                                 │
│           ▼                                                 │
│      Binary Classifier (Fall/ADL)                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `skeleton_encoder.py` | SkeletonTransformer teacher model |
| `resampler.py` | EventTokenResampler + TimestampAwareStudent |
| `losses.py` | KD losses (Embedding, Gram, COMODO) |
| `trainer.py` | KDTrainer for joint training |
| `data_loader.py` | SmartFallMM data loading with timestamps |
| `stress_test.py` | Timestamp perturbation utilities |
| `test_resampler.py` | Unit tests and stress tests |
| `run_teacher_ablation.py` | Teacher architecture ablation |
| `analysis.py` | Dataset timestamp analysis |

## Quick Start

```bash
# 1. Analyze timestamp reliability
python kd/test_resampler.py --analyze-timestamps --data-root data

# 2. Run unit tests
python kd/test_resampler.py --unit-tests

# 3. Train teacher (skeleton only)
python kd/run_teacher_ablation.py --num-gpus 4 --parallel 2 --data-root data

# 4. Stress test student robustness
python kd/test_resampler.py --stress-tests --data-root data
```

## Novel Contribution: EventTokenResampler

Handles irregular IMU timestamps via learned time-anchored queries:

```python
# Problem: IMU timestamps are unreliable (drift, duplicates, gaps)
# Solution: Cross-attention from fixed queries to variable events

time_queries = linspace(0, 1, num_tokens)  # L fixed positions
tokens = cross_attention(
    query=time_queries,   # (B, L, D)
    key=events,           # (B, N, D) variable N
    value=events
)
# Output: (B, L, D) - fixed regardless of input length
```

Drift-invariant time features:
- `delta_t`: time since previous event
- `log(1 + delta_t)`: compressed gap representation
- `tau`: normalized position in [0, 1]

## Configuration

See `kd_config/` for example configurations:
- `teacher_skeleton.yaml`: Skeleton teacher training
- `student_kd.yaml`: Student with KD from teacher
- `ablation_kd.yaml`: KD loss ablation sweep
