# Novel Contributions

**Target Venue**: IMWUT/UbiComp (Wearable sensing, real-world deployment)

## 1. EventTokenResampler

**Problem**: IMU data has irregular timestamps (drift, duplicates, gaps) that break standard fixed-rate assumptions.

**Solution**: Learned cross-attention from time-anchored queries to variable-length events.

```python
# Key innovation: queries at fixed τ positions attend to events
time_queries = linspace(0, 1, L)  # L fixed query positions
tokens = cross_attention(queries, events)  # (B, N, D) → (B, L, D)
```

**Why novel**:
- Unlike interpolation: learns which events matter
- Unlike DTW: no monotonicity assumption, differentiable
- Unlike fixed-rate: handles variable N naturally

**Drift-invariant features**:
- `delta_t`: relative gaps (not absolute time)
- `log(1 + delta_t)`: compresses large gaps
- `tau`: normalized position [0,1]

## 2. Alignment-Free Cross-Modal KD

**Problem**: Skeleton has NO timestamps, IMU timestamps are unreliable. Cannot do frame-level alignment.

**Solution**: KD losses that match global/structural features, not temporal positions.

| Loss | What it matches | Why alignment-free |
|------|-----------------|-------------------|
| EmbeddingKD | Global pooled features | Single vector per sample |
| GramKD | Token correlation structure | L×L matrix, order-invariant |
| COMODO | Distribution over queue | No per-sample alignment |

**Why novel**:
- Most KD assumes aligned teacher-student outputs
- Our approach works even when temporal correspondence is impossible
- Robust to severe timestamp noise

## 3. Timestamp Robustness Evaluation

**Problem**: No standard way to evaluate model robustness to timestamp issues.

**Solution**: Systematic stress testing framework.

| Perturbation | Simulates |
|--------------|-----------|
| Jitter | Clock inaccuracy |
| Dropout | Sensor failures, packet loss |
| Burst | Buffered transmission |
| Warp | Clock drift |

**Metrics**: F1 degradation curves under increasing perturbation.

## Claims to Validate

| Claim | Experiment |
|-------|------------|
| EventTokenResampler handles irregular timestamps | Stress test vs fixed-rate baseline |
| Alignment-free KD works without temporal correspondence | Compare +KD vs -KD on student |
| Method is robust to real-world timestamp noise | Stress test degradation curves |

## Publication Story

> "Cross-modal KD from skeleton to IMU is challenging because (1) skeleton lacks timestamps for alignment, and (2) IMU timestamps are unreliable. We propose alignment-free KD using global embedding and Gram losses, combined with an EventTokenResampler that handles irregular IMU sampling via learned time-anchored cross-attention."

## Comparison to Related Work

| Approach | Handles Irregular Timestamps | Alignment-Free | Differentiable |
|----------|------------------------------|----------------|----------------|
| Interpolation | No | N/A | Yes |
| DTW | Partially | No | No |
| Neural ODE | Yes | N/A | Yes |
| **EventTokenResampler** | **Yes** | **Yes** | **Yes** |
