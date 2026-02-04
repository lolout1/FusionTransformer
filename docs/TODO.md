# TODO List

**Updated**: 2026-02-04

## Immediate (This Week)

### Teacher Training
- [ ] Complete teacher ablation (72 configs running)
- [ ] Analyze results: embed_dim, layers, heads, pos_enc effects
- [ ] Select best teacher config
- [ ] Save best teacher weights

### Timestamp Analysis
- [ ] Run `python kd/test_resampler.py --analyze-timestamps`
- [ ] Document timestamp reliability score
- [ ] Decide: use timestamps or ignore them?

### Unit Tests
- [ ] Run `python kd/test_resampler.py --unit-tests`
- [ ] Verify all 10 tests pass
- [ ] Fix any failures

## Short-term (Next Week)

### Student Baseline
- [ ] Train student without KD on IMU data
- [ ] Compare TimestampAwareStudent vs FixedRateStudent
- [ ] Establish baseline F1 scores

### Stress Tests
- [ ] Run `python kd/test_resampler.py --stress-tests`
- [ ] Generate comparison table (resampler vs baseline)
- [ ] Plot robustness curves

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
