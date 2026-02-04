# Configs

Format: `{stride}_{input}_{norm}.yaml`

## Inputs

| Input | Channels | Kalman | Format |
|-------|----------|--------|--------|
| kalman_gyromag | 7 | Yes | smv, acc, gyro_mag, roll, pitch |
| kalman_yaw | 7 | Yes | smv, acc, roll, pitch, yaw |
| raw_gyro | 7 | No | smv, acc, gx, gy, gz |
| raw_gyromag | 5 | No | smv, acc, gyro_mag |

## Generate Configs

```bash
python tools/generate_server_configs.py --stride s{fall}_{adl}
```
