# TODO List

**Updated**: 2026-02-04

## Immediate (This Week)

### Teacher Training
- [x] Complete teacher ablation (initial run)
- [x] Best so far: e96_l2_h4_d3_bce (89.32% test F1)
- [ ] Complete pos_enc ablation (72 configs)
- [ ] Select final best teacher config
- [ ] Save best teacher weights

### Timestamp Analysis
- [x] Run `python kd/test_resampler.py --analyze-timestamps`
- [x] Document timestamp reliability score: **35/100 (UNRELIABLE)**
- [x] Decision: **Ignore timestamps, use position-only mode**

### Unit Tests
- [x] Run `python kd/test_resampler.py --unit-tests`
- [x] All 10 tests pass
- [x] Edge cases verified

## Short-term (Next Week)

### Student Training (Position Mode)
- [ ] Train student without KD using position-only resampler
- [ ] Establish baseline F1 score
- [ ] Compare with fixed-rate interpolation

### Stress Tests
- [x] Run stress test comparison
- [x] Position mode: **62.0%** mean F1 (best)
- [x] Timestamp mode: 55.3% mean F1
- [x] Baseline: 50.7% mean F1
- [x] Position mode is most robust to perturbations

### KD Training
- [ ] Implement full KD training loop
- [ ] Test with frozen teacher
- [ ] Tune KD loss weights

## Medium-term (2-3 Weeks)

### Full Pipeline
- [ ] End-to-end KD training
- [ ] Ablation: which KD losses help most?
- [ ] Final test evaluation

### Multi-Dataset
- [ ] Adapt to UP-FALL dataset
- [ ] Adapt to WEDA-FALL dataset
- [ ] Cross-dataset evaluation

### Documentation
- [ ] Write method section
- [ ] Create architecture figures
- [ ] Document reproducibility steps

## Long-term (Paper Submission)

### Experiments
- [ ] Complete all ablations
- [ ] Statistical significance tests
- [ ] Comparison with baselines

### Writing
- [ ] Introduction
- [ ] Related work
- [ ] Method
- [ ] Experiments
- [ ] Conclusion

### Artifacts
- [ ] Clean code release
- [ ] Pre-trained weights
- [ ] README with reproduction steps

## Known Issues

1. **Timestamp reliability**: IMU timestamps may be too noisy to use
   - Mitigation: Test with synthetic uniform timestamps

2. **Teacher-student dimension mismatch**: Need projection layer
   - Status: Fixed in losses.py

3. **Data loading speed**: Timestamp parsing is slow
   - Status: Fixed with skeleton-only dataset for teacher

## Completed

- [x] SkeletonTransformer implementation
- [x] EventTokenResampler implementation
- [x] KD losses (Embedding, Gram, COMODO)
- [x] Teacher ablation script
- [x] Stress test utilities
- [x] Unit test framework
- [x] Data loader with timestamps
