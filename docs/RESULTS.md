# Experiment Results

**Updated**: 2026-02-04

## Robustness Fixes Applied (2026-02-04)

| Issue | File | Fix |
|-------|------|-----|
| Empty timestamp crash | `data_loader.py:355` | Guard with length check |
| Invalid trial filter | `data_loader.py:468` | Skip empty timestamp trials |
| imu_data None | `trainer.py:225` | Skip batch if missing |

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

## Stress Test Results (2026-02-04)

### Time Mode Comparison

Compared four approaches to handling timestamps in EventTokenResampler:

| Mode | Description |
|------|-------------|
| **position** | Ignore timestamps, use sequence position τ = i/N |
| timestamps | Use raw timestamps (delta_t, log_delta, tau) |
| cleaned | Clip gaps to [0.1ms, 100ms], handle duplicates |
| baseline | Fixed-rate interpolation (no resampler) |

### Results Table

| Condition | position | timestamps | cleaned | baseline |
|-----------|----------|------------|---------|----------|
| clean | **62.1%** | 58.5% | 57.1% | 49.6% |
| jitter_5ms | **62.1%** | 58.5% | 57.1% | 49.6% |
| jitter_10ms | **62.1%** | 58.5% | 57.1% | 49.6% |
| jitter_20ms | **62.1%** | 58.6% | 57.1% | 49.6% |
| jitter_50ms | **62.1%** | 58.6% | 57.1% | 49.6% |
| dropout_5pct | **61.9%** | 46.9% | 37.0% | 51.9% |
| dropout_10pct | **61.9%** | 48.8% | 35.7% | 53.6% |
| dropout_20pct | **61.6%** | 47.5% | 35.3% | 53.1% |
| dropout_30pct | **61.7%** | 48.7% | 37.4% | 54.9% |
| burst_25pct | **62.1%** | 59.0% | 57.1% | 49.6% |
| burst_50pct | **62.1%** | 58.2% | 56.8% | 49.6% |
| warp_10pct | **62.1%** | 58.6% | 57.1% | 49.6% |
| warp_20pct | **62.1%** | 58.6% | 57.4% | 49.6% |
| **Mean** | **62.0%** | 55.3% | 50.7% | 50.7% |

### Key Findings

1. **Position-only wins**: Ignoring timestamps entirely (62.0%) beats using them (55.3%)
2. **Timestamps hurt under dropout**: Raw timestamps drop to 46.9-48.8% with dropout, position stays at 61.6-61.9%
3. **Cleaning doesn't help**: "Cleaned" timestamps (50.7%) perform worse than raw (55.3%)
4. **Resampler > Baseline**: Position mode (62.0%) beats fixed-rate interpolation (50.7%)

### Implications

- SmartFallMM IMU timestamps are **unreliable** (score: 35/100)
- The novelty is in **learned cross-attention resampling**, not timestamp encoding
- Default mode should be `position` for this dataset
- Timestamp features may help on datasets with reliable timestamps

## Timestamp Reliability Analysis (2026-02-04)

Analyzed 20 random IMU files from SmartFallMM:

| Metric | Value | Expected (32 Hz) |
|--------|-------|------------------|
| Mean delta_t | 216 ms | 31 ms |
| Median delta_t | 37 ms | 31 ms |
| Std delta_t | 7568 ms | ~5 ms |
| Zero-delta (duplicates) | ~15% | 0% |
| Gaps >100ms | 1-3% | 0% |
| Max gap | 558 sec | ~100 ms |

**Reliability Score: 35/100** (UNRELIABLE)

Timestamps suffer from: clock drift, duplicate entries, large gaps, inconsistent sampling rate.

## Student KD Results

*Pending - requires trained teacher*
