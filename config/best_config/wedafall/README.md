# WEDA-FALL Dataset Configuration

## Dataset Overview

| Parameter | Value |
|-----------|-------|
| Sensor | Consumer Fitbit (wrist-worn) |
| Sampling Rate | 50 Hz |
| Subjects | 14 (young adults) |
| LOSO Folds | 12 |
| Modalities | Accelerometer (3-axis), Gyroscope (3-axis) |

---

## Window Size Analysis

### Sample Counts by Window Size

| Window | Frames | Train Total | Train Falls | Train ADLs | Test Total | Test Falls | Test ADLs |
|--------|--------|-------------|-------------|------------|------------|------------|-----------|
| **2s** | 100 | 21,951 | 12,793 (58%) | 9,158 (42%) | 1,816 | 1,155 (64%) | 661 (36%) |
| **3s** | 150 | 19,422 | 11,030 (57%) | 8,392 (43%) | 1,571 | 1,022 (65%) | 549 (35%) |
| **4s** | 200 | 16,918 | 9,319 (55%) | 7,599 (45%) | 1,367 | 866 (63%) | 501 (37%) |
| **5s** | 250 | 14,789 | 7,894 (53%) | 6,895 (47%) | 749 | 418 (56%) | 331 (44%) |
| **6s** | 300 | 12,204 | 6,182 (51%) | 6,022 (49%) | 693 | 286 (41%) | 407 (59%) |

### Performance vs Window Size

| Window | Best F1 | Δ from Prev | Train Samples | Test Samples | F1 per 1K Train |
|--------|---------|-------------|---------------|--------------|-----------------|
| 2s | 87.58% | baseline | 21,951 | 1,816 | 3.99 |
| 3s | 90.06% | +2.48% | 19,422 | 1,571 | 4.64 |
| 4s | 92.25% | +2.19% | 16,918 | 1,367 | 5.45 |
| 5s | 94.58% | +2.33% | 14,789 | 749 | 6.40 |
| **6s** | **95.93%** | +1.35% | 12,204 | 693 | 7.86 |

### Key Observations

1. **Performance scales with window size**: +8.35% F1 gain from 2s to 6s
2. **Training data reduction**: 44% fewer samples at 6s vs 2s
3. **Test set shrinks significantly**: 62% reduction (1,816 → 693 samples)
4. **Class balance shifts**: 2s test is 64% falls; 6s test is 41% falls
5. **Diminishing returns**: 5s→6s gain (+1.35%) smaller than 4s→5s (+2.33%)

---

## Optimizing Transformers for Shorter Windows

The challenge: Achieve 6s-level performance (~96% F1) with 2-3s windows to enable:
- Faster inference (real-time deployment)
- More training/test samples (statistical robustness)
- Lower latency fall detection

### Approach 1: Multi-Scale Temporal Encoding

**Concept**: Capture patterns at multiple temporal resolutions simultaneously.

```
Input (B, T, C)
    │
    ├─► Conv1D(k=3)  → local patterns (50-150ms)
    ├─► Conv1D(k=7)  → medium patterns (150-350ms)
    ├─► Conv1D(k=15) → fall dynamics (300-750ms)
    │
    └─► Concat + Projection → Transformer
```

| Pros | Cons |
|------|------|
| Captures multi-scale features in single pass | Increased parameters (~30-50%) |
| No additional data required | May overfit on small datasets |
| Proven in vision (Inception) and audio | Hyperparameter tuning for kernel sizes |

**Implementation**: Already have `multikernel` encoder in `Models/gated_fusion.py`

---

### Approach 2: Frequency-Domain Augmentation

**Concept**: Add spectral features to capture fall signatures that span window boundaries.

```
Input: Raw IMU (B, T, 7)
    │
    ├─► Time-domain path (existing)
    │
    └─► FFT → magnitude spectrum → frequency features
        │
        └─► Concat with time-domain before classifier
```

| Pros | Cons |
|------|------|
| Falls have characteristic frequency signatures | FFT assumes stationarity |
| Captures periodic patterns (walking, etc.) | Adds preprocessing complexity |
| Works well for short windows | May not help for impact detection |

**Key frequencies for falls**: 0.5-3 Hz (body sway), 5-15 Hz (impact vibrations)

---

### Approach 3: Overlapping Window Ensemble

**Concept**: Use highly overlapping windows and aggregate predictions.

```
Signal: [--------------------6s--------------------]
         [--2s--][--2s--][--2s--][--2s--][--2s--]
              ↓      ↓      ↓      ↓      ↓
           pred1  pred2  pred3  pred4  pred5
                         ↓
                  Temporal voting/pooling
```

| Pros | Cons |
|------|------|
| Uses short windows, captures long context | 5x inference cost |
| Natural uncertainty quantification | Latency increases |
| Easy to implement | Requires calibrated probabilities |

**Voting strategies**: Max pooling, mean pooling, learned temporal attention

---

### Approach 4: Dilated Causal Convolutions (TCN-style)

**Concept**: Exponentially increasing receptive field without increasing parameters.

```
Layer 1: dilation=1  →  receptive field = 3
Layer 2: dilation=2  →  receptive field = 7
Layer 3: dilation=4  →  receptive field = 15
Layer 4: dilation=8  →  receptive field = 31
                              ↓
                    Effective RF = 6s with 2s input
```

| Pros | Cons |
|------|------|
| Large receptive field, few parameters | Causal constraint may hurt |
| Proven for time series (WaveNet, TCN) | Less flexible than attention |
| Parallelizable (unlike RNN) | Fixed receptive field pattern |

**Reference**: Bai et al., "TCN for Sequence Modeling" (2018)

---

### Approach 5: Relative Positional Attention

**Concept**: Replace absolute positions with relative distances for better generalization.

```
Standard: Q·K^T + positional_embedding
Relative: Q·K^T + relative_position_bias[i-j]
```

| Pros | Cons |
|------|------|
| Better length generalization | Slightly more complex |
| Captures "time since" relationships | May need more data |
| Used in Transformer-XL, DeBERTa | Implementation overhead |

**Benefit for falls**: "Impact happened X timesteps ago" is more relevant than "Impact at position Y"

---

### Approach 6: Contrastive Pre-training

**Concept**: Learn representations that distinguish fall vs ADL patterns before fine-tuning.

```
Stage 1: Self-supervised contrastive learning
         - Positive: augmented versions of same window
         - Negative: windows from different activities

Stage 2: Fine-tune classifier on labeled data
```

| Pros | Cons |
|------|------|
| Leverages unlabeled data | Two-stage training |
| Better representations | Requires augmentation design |
| State-of-art in vision, NLP | May not help with small datasets |

**Augmentation options**: Time warping, magnitude scaling, jittering, rotation

---

### Approach 7: Knowledge Distillation

**Concept**: Train small (2s) model to mimic large (6s) model predictions.

```
Teacher: 6s window model (95.93% F1)
    │
    └─► Soft labels (probability distributions)
            │
            ↓
Student: 2s window model
    │
    └─► Loss = α·CE(hard_labels) + (1-α)·KL(teacher_probs)
```

| Pros | Cons |
|------|------|
| Transfer knowledge without more data | Requires trained teacher |
| Proven technique (Hinton et al.) | Student ceiling = teacher |
| Can compress model size too | Temperature tuning needed |

---

### Approach 8: Learnable Temporal Aggregation

**Concept**: Let the model learn how to weight different time regions.

```
Current: Mean pooling or attention pooling over T dimension
Proposed: Multi-head temporal attention with learned queries

Q = learnable_queries (K queries)
K, V = encoder_output
Output = softmax(Q·K^T)·V  →  (B, K, D)
```

| Pros | Cons |
|------|------|
| Flexible aggregation | More parameters |
| Can focus on impact region | May overfit |
| Interpretable attention weights | Needs careful initialization |

---

## Recommended Investigation Priority

Based on implementation effort vs expected gain:

| Priority | Approach | Effort | Expected Gain | Rationale |
|----------|----------|--------|---------------|-----------|
| 1 | Multi-scale encoding | Low | Medium | Already have multikernel encoder |
| 2 | Overlapping ensemble | Low | High | No model changes, just inference |
| 3 | Dilated convolutions | Medium | Medium | Proven for time series |
| 4 | Knowledge distillation | Medium | High | 6s teacher already exists |
| 5 | Frequency features | Medium | Low-Medium | Domain-specific insight needed |
| 6 | Relative attention | High | Medium | Architecture change |
| 7 | Contrastive pre-training | High | Unknown | Research direction |

---

## Current Best Configurations

### Best Overall (6s window)
```yaml
# 95.93% F1 - kalman_global_gate_conv1d_linear_6s_d05_g32
model: Models.gated_fusion.GatedFusionTransformer
window_size: 300  # 6s at 50Hz
fusion_mode: concat_global_gate
acc_encoder: conv1d
ori_encoder: linear
embed_dim: 48
dropout: 0.5
gate_hidden_dim: 32
```

### Best for Statistical Robustness (5s window)
```yaml
# 94.58% F1 - 749 test samples vs 693 for 6s
window_size: 250  # 5s at 50Hz
# Otherwise same as above
```

### Best Short Window (4s)
```yaml
# 92.25% F1 - 1,367 test samples
window_size: 200  # 4s at 50Hz
```

---

## Fusion Mode Comparison

| Mode | Best F1 | Mean F1 | Experiments | Overhead |
|------|---------|---------|-------------|----------|
| **Global Gate** | **95.93%** | 89.98% | 135 | ~800 params |
| Concat | 94.61% | 90.68% | 50 | 0 |
| Timestep Gate | 92.06% | 87.24% | 27 | ~1.2K params |

Global gate provides +1.32% improvement over concat baseline with minimal parameter overhead.

---

## Files in This Directory

| File | Description |
|------|-------------|
| `kalman.yaml` | Best Kalman config (young subjects) |
| `kalman_with_elderly.yaml` | Kalman config including elderly subjects |
| `kalman_all_subjects.yaml` | Full LOSO with all subjects |
| `raw.yaml` | Raw gyroscope input (no Kalman fusion) |
| `acc_gyro_kalman.yaml` | AccGyroKalman architecture |

---

## Why LSTM Outperforms Transformer on Short Windows

### Verified Results

| Window | LSTM | Transformer | Winner |
|--------|------|-------------|--------|
| **2s** | **91.86%** | 87.58% | LSTM +4.28% |
| **4s** | **94.48%** | 92.25% | LSTM +2.23% |
| 5s | -- | 94.58% | -- |
| 6s | -- | **95.93%** | Transformer |

### Root Causes

| Issue | Explanation |
|-------|-------------|
| **No locality bias** | Transformers must learn "nearby = related" from data; LSTMs assume it |
| **Attention overkill** | 100×100 attention for 2s window when everything is already "nearby" |
| **Weak positional encoding** | Sinusoidal PE values too similar for adjacent positions |
| **Parameter inefficiency** | Same params, but LSTM uses them better for sequential data |

### When Transformers Excel

| Condition | Transformer Advantage |
|-----------|----------------------|
| Long sequences (>256) | Global attention captures long-range dependencies |
| Large datasets (>100K) | Can learn structure from scratch |
| Pre-training available | Transfer learning benefits |

### When Transformers Struggle

| Condition | Why |
|-----------|-----|
| Short sequences (<200) | Global attention is unnecessary |
| Limited data (<20K) | Can't learn what CNN/LSTM "know" innately |
| Strong local patterns | Inductive bias helps |

---

## Optimizing Transformers for Short Windows

### Strategy 1: Reduce Model Capacity

```yaml
# Current (optimized for 6s)
embed_dim: 48
num_layers: 2
num_heads: 4

# For 2s windows
embed_dim: 24
num_layers: 1
num_heads: 2
```

Less overfitting with matched capacity.

### Strategy 2: Deeper CNN Before Transformer

```
Current:  Input → Conv1D(k=8) → Transformer
Proposed: Input → Conv1D(k=3) → Conv1D(k=5) → Conv1D(k=7) → Transformer
```

CNN "teaches" transformer about local patterns.

### Strategy 3: LSTM-Transformer Hybrid

```
Input → LSTM (sequential bias) → Transformer (1 layer) → Classifier
```

Best of both: LSTM's inductive bias + Transformer's attention.

### Strategy 4: Convolutional Position Encoding

```python
# Instead of sinusoidal:
x = x + positional_embedding[positions]

# Use learned conv:
x = x + depthwise_conv1d(x, kernel_size=3)
```

### Strategy 5: Local Attention

```python
# Full attention (current)
attn[i,j] for all i,j in [0, T)

# Local attention (proposed)
attn[i,j] only if |i-j| <= window_size (e.g., 10)
```

---

## References

1. Vaswani et al., "Attention Is All You Need" (2017) - Transformer architecture
2. Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018) - TCN
3. Hinton et al., "Distilling the Knowledge in a Neural Network" (2015) - Knowledge distillation
4. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (2020) - SimCLR
5. Shaw et al., "Self-Attention with Relative Position Representations" (2018) - Relative attention
