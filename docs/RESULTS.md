# Experiment Results

**Updated**: 2026-02-04

## Teacher Ablation (Skeleton-Only)

### Best Configuration

| Config | Val F1 | Test F1 | Params |
|--------|--------|---------|--------|
| **e96_l2_h4_d3_bce** | 99.34% | 89.32% | 341K |
| e96_l2_h8_d3_bce | 97.99% | 85.71% | 341K |
| e96_l2_h4_d5_bce | 62.04% | 56.36% | 341K |

### Key Findings (Preliminary)

1. **Dropout**: 0.3 >> 0.5 (lower dropout works better)
2. **Heads**: 4 heads slightly better than 8
3. **Embed dim**: 96 sufficient for skeleton (96 input channels)

### Configuration Details

Best model: `e96_l2_h4_d3_bce`
- embed_dim: 96
- num_layers: 2
- num_heads: 4
- dropout: 0.3
- loss: BCE
- Best epoch: 38 (early stopped at 54)
- Training time: ~115 seconds

### Data Split

| Split | Subjects | Trials | Windows |
|-------|----------|--------|---------|
| Train | 26 | ~1,200 | ~18,000 |
| Val | [29, 30] | ~36 | ~250 |
| Test | [31, 32] | ~47 | ~260 |

## Stress Test Results

*Pending - run with:*
```bash
python kd/test_resampler.py --stress-tests --data-root data
```

## Student KD Results

*Pending - requires trained teacher*

## Timestamp Analysis

*Pending - run with:*
```bash
python kd/test_resampler.py --analyze-timestamps --data-root data
```
