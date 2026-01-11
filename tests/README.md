# Tests Directory

This directory contains all test and validation scripts.

## Running Tests

Run tests from the **project root directory**:

```bash
# From project root
python tests/test_models.py
python tests/test_integration.py
python tests/test_validation_split.py
```

## Test Files

| Test | Description |
|------|-------------|
| `test_models.py` | Model architecture verification |
| `test_integration.py` | End-to-end integration tests |
| `test_validation_split.py` | Validation split tests |
| `test_imu_setup.py` | IMU setup verification |
| `test_imu_pipeline.py` | IMU data pipeline tests |
| `test_sensor_comparison_setup.py` | Sensor comparison setup |
| `test_sensor_config_validation.py` | Config validation |
| `test_dtw_validation.py` | DTW alignment validation |
| `test_stride_sync.py` | Stride synchronization tests |
| `test_channel_fix.py` | Channel configuration tests |
| `test_stats_fix.py` | Statistics calculation tests |
| `test_gravity_removal.py` | Gravity removal validation |
| `test_fold_grouper.py` | Cross-validation fold tests |

## Quick Verification

Run key tests before experiments:

```bash
python tests/test_models.py
python tests/test_imu_setup.py
python tests/test_validation_split.py
```
