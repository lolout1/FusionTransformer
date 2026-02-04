# Novel Contributions

**Target Venue**: IMWUT/UbiComp (Wearable sensing, real-world deployment)

## 1. EventTokenResampler

**Problem**: Variable-length sensor sequences need to map to fixed-length representations for downstream models.

**Solution**: Learned cross-attention from position-anchored queries to variable-length events.

```python
# Key innovation: L learned queries attend to variable N events
queries = position_embed(linspace(0, 1, L))  # L query positions
tokens = cross_attention(queries, events)     # (B, N, D) → (B, L, D)
```

**Why novel**:
- Unlike interpolation: learns which events matter via attention
- Unlike DTW: no monotonicity assumption, fully differentiable
- Unlike fixed-rate: handles variable N naturally
- Unlike RNNs: parallel computation, explicit positional structure

**Key empirical finding**: Position-only mode (ignoring timestamps) outperforms timestamp-aware modes by 6.7% F1. This is because:
1. Real-world IMU timestamps are often unreliable (SmartFallMM: 35/100 reliability score)
2. Learned attention captures relevant structure without explicit time encoding
3. Position encoding provides sufficient ordering information

**Modes supported**:
- `position`: τ = i/N (recommended for unreliable timestamps)
- `timestamps`: delta_t, log(1+delta_t), normalized τ
- `cleaned`: timestamps with gap clipping

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

| Claim | Experiment | Status |
|-------|------------|--------|
| EventTokenResampler beats fixed-rate baseline | Stress test comparison | ✅ 62.0% vs 50.7% |
| Position-only outperforms timestamp-aware | Time mode ablation | ✅ 62.0% vs 55.3% |
| Model is robust to timestamp perturbations | Stress test (jitter, dropout, burst) | ✅ <1% degradation |
| Alignment-free KD works without temporal correspondence | Compare +KD vs -KD on student | Pending |

## Publication Story

> "We present EventTokenResampler, a learned cross-attention module that maps variable-length sensor sequences to fixed-length token representations. Unlike interpolation or DTW, our approach learns which events are informative via attention. Surprisingly, we find that position-only encoding (ignoring timestamps) significantly outperforms timestamp-aware variants on real-world IMU data, where timestamps suffer from drift, duplicates, and gaps. Combined with alignment-free KD losses that match global features rather than temporal positions, our method enables effective knowledge transfer from skeleton to IMU modality without requiring temporal alignment."

## Comparison to Related Work

| Approach | Variable Length | Learns Importance | Differentiable | Robust to Bad Timestamps |
|----------|-----------------|-------------------|----------------|--------------------------|
| Interpolation | No | No | Yes | No |
| DTW | Yes | No | No | No |
| Neural ODE | Yes | Partially | Yes | Partially |
| RNN/LSTM | Yes | Implicitly | Yes | Partially |
| **EventTokenResampler** | **Yes** | **Yes (attention)** | **Yes** | **Yes (position mode)** |

## Stress Test Results Summary

| Perturbation | Position Mode | Timestamp Mode | Degradation |
|--------------|---------------|----------------|-------------|
| Clean | 62.1% | 58.5% | +3.6% |
| Dropout 30% | 61.7% | 48.7% | +13.0% |
| Jitter 50ms | 62.1% | 58.6% | +3.5% |
| Burst 50% | 62.1% | 58.2% | +3.9% |

**Conclusion**: Position-only mode is more robust AND achieves higher accuracy.
