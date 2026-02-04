# Skeleton→IMU Knowledge Distillation Plan

**Status**: In Progress
**Updated**: 2026-02-04

## Overview

Cross-modal knowledge distillation from skeleton teacher to IMU student for wearable fall detection. Focus on alignment-free methods robust to missing skeleton timestamps and irregular IMU sampling.

## Problem Statement

| Modality | Timestamps | Quality | Availability |
|----------|-----------|---------|--------------|
| Skeleton | None (sequential frames @ 30 FPS) | High | Training only |
| IMU | Irregular, unreliable | Variable | Training + Deployment |

**Challenge**: Cannot do frame-level alignment between modalities because:
1. Skeleton has no timestamps
2. IMU timestamps have drift, duplicates, gaps up to 558 seconds

## Architecture

### Teacher: SkeletonTransformer

```
Input: (B, T, 96) - 32 joints × 3 xyz
  │
  ▼ JointEmbedding (Linear + LayerNorm + SiLU)
  │
  ▼ PositionalEncoding (learnable, optional)
  │
  ▼ TransformerEncoder (N layers)
  │
  ▼ SqueezeExcitation (channel attention)
  │
  ▼ TemporalAttentionPooling
  │
Output: logits (B, 1), features (B, embed_dim)
```

### Student: TimestampAwareStudent

```
Input: events (B, N, 6), timestamps (B, N)
  │
  ▼ EventTokenResampler (variable N → fixed L tokens)
  │   - TimeFeatureEncoder: [delta_t, log_delta, tau]
  │   - Cross-attention: queries attend to events
  │
  ▼ TransformerEncoder (N layers)
  │
  ▼ SqueezeExcitation
  │
  ▼ TemporalAttentionPooling
  │
Output: logits (B, 1), features (B, embed_dim)
```

## KD Losses (Alignment-Free)

| Loss | Description | Requires Alignment |
|------|-------------|-------------------|
| **EmbeddingKD** | Cosine distance on pooled features | No |
| **GramKD** | MSE on L×L token similarity matrices | No |
| **COMODO** | Distribution matching via queue | No |

These losses match **global/structural** features, not temporal positions.

## Ablation Factors

### Teacher Ablation (run_teacher_ablation.py)

| Factor | Values | Purpose |
|--------|--------|---------|
| embed_dim | 96, 128, 192 | Model capacity |
| num_layers | 2, 3, 4 | Depth |
| num_heads | 4, 8 | Attention diversity |
| dropout | 0.3, 0.5 | Regularization |
| use_pos_enc | True, False | Positional encoding effect |

### Student Ablation (TODO)

| Factor | Values | Purpose |
|--------|--------|---------|
| num_tokens | 32, 64, 128 | Resampling resolution |
| time_features | with/without | Value of timestamp info |
| KD losses | combinations | Loss contribution |

## Data Split

| Split | Subjects | Purpose |
|-------|----------|---------|
| Train | 26 subjects | Model training |
| Val | [29, 30] | Early stopping |
| Test | [31, 32] | Final evaluation |

## Experimental Plan

### Phase 1: Teacher Training ✓
- [x] Implement SkeletonTransformer
- [x] Create teacher ablation script
- [x] Run ablation (in progress)

### Phase 2: Timestamp Analysis ✓
- [x] Implement timestamp reliability analysis
- [x] Create stress test utilities
- [ ] Generate timestamp quality report

### Phase 3: Student Training
- [x] Implement EventTokenResampler
- [x] Implement TimestampAwareStudent
- [ ] Train student without KD (baseline)
- [ ] Train student with KD
- [ ] Compare vs fixed-rate baseline

### Phase 4: Stress Testing
- [x] Implement perturbation functions (jitter, dropout, burst, warp)
- [ ] Run stress test comparison
- [ ] Generate robustness curves

### Phase 5: Full KD Pipeline
- [ ] Joint teacher-student training
- [ ] Loss weight tuning
- [ ] Final evaluation on test set

## Expected Results

| Model | Input | Test F1 | Notes |
|-------|-------|---------|-------|
| Teacher | Skeleton | ~90%+ | Upper bound |
| Student (no KD) | IMU | ~85% | Baseline |
| Student (with KD) | IMU | ~88%+ | Target |

## Open Questions

1. **Timestamp reliability**: Are IMU timestamps useful at all, or should we ignore them?
2. **Positional encoding**: Does it help for skeleton? Student?
3. **KD loss weights**: Optimal balance of embedding vs gram vs COMODO?
