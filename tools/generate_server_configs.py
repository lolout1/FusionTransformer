#!/usr/bin/env python3
"""Generate server configs for deployed model variants.

Creates YAML configs for all input × normalization combinations for a given stride.

Usage:
    python tools/generate_server_configs.py --stride s16_32
    python tools/generate_server_configs.py --stride s8_32 --output /path/to/configs/
    python tools/generate_server_configs.py --list-strides
"""

import argparse
from pathlib import Path

# Input configurations
INPUTS = {
    'kalman_gyromag': {
        'feature_mode': 'kalman_gyro_mag',
        'architecture': 'KalmanConv1dConv1d',
        'imu_channels': 7,
        'requires_kalman': True,
    },
    'kalman_yaw': {
        'feature_mode': 'kalman',
        'architecture': 'KalmanConv1dConv1d',
        'imu_channels': 7,
        'requires_kalman': True,
    },
    'raw_gyro': {
        'feature_mode': 'raw',
        'architecture': 'SingleStreamTransformerSE',
        'imu_channels': 7,
        'requires_kalman': False,
    },
    'raw_gyromag': {
        'feature_mode': 'raw_gyromag',
        'architecture': 'SingleStreamTransformerSE',
        'imu_channels': 5,
        'requires_kalman': False,
    },
}

NORMS = {
    'norm': {'mode': 'acc_only', 'needs_scaler': True},
    'nonorm': {'mode': 'none', 'needs_scaler': False},
}

CONFIG_TEMPLATE = '''# {name} - Stride {stride}
# Architecture: {architecture}
# Input: {input_desc} - {channels}ch
# Normalization: {norm_desc}

nats_url: "nats://localhost:4222"
subject_pattern: "m.kalman_transformer.*"
default_fs_hz: 31.25
window_size: 128

preprocessing:
  feature_mode: {feature_mode}
  normalization_mode: {norm_mode}
  scaler_path: {scaler_path}
  convert_gyro_to_rad: true

kalman:
  filter_type: linear
  Q_orientation: 0.005
  Q_rate: 0.01
  R_acc: 0.05
  R_gyro: 0.1

state:
  timeout_ms: 10000
  cache_ttl_ms: 60000
  enable_incremental: false

model:
  architecture: {architecture}
  weights_path: weights/{weights_file}
  device: cpu
  model_args:
    imu_frames: 128
    imu_channels: {channels}
    embed_dim: 48
    acc_ratio: 0.65
    num_heads: 4
    num_layers: 2
'''


def generate_config(stride: str, input_name: str, norm_name: str) -> str:
    """Generate config YAML content."""
    input_cfg = INPUTS[input_name]
    norm_cfg = NORMS[norm_name]

    name = f"{stride}_{input_name}_{norm_name}"
    weights_file = f"{name}.pth"

    scaler_path = "null"
    if norm_cfg['needs_scaler']:
        scaler_path = f"weights/scalers/{stride}_{input_name}_norm_scaler.pkl"

    return CONFIG_TEMPLATE.format(
        name=name,
        stride=stride,
        architecture=input_cfg['architecture'],
        input_desc=input_name.replace('_', ' ').title(),
        channels=input_cfg['imu_channels'],
        norm_desc='Acc normalization' if norm_name == 'norm' else 'No normalization',
        feature_mode=input_cfg['feature_mode'],
        norm_mode=norm_cfg['mode'],
        scaler_path=scaler_path,
        weights_file=weights_file,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate server configs")
    parser.add_argument('--stride', type=str, help="Stride config (e.g., s16_32)")
    parser.add_argument('--output', type=str,
                        default="kalman_server/SmartFallNATS-rakesh-fusion-server-version/KalmanFusionServer/configs",
                        help="Output directory")
    parser.add_argument('--list-strides', action='store_true', help="List available strides")
    args = parser.parse_args()

    if args.list_strides:
        print("Standard stride configurations:")
        print("  s8_16  - fall=8, adl=16 (high overlap)")
        print("  s8_32  - fall=8, adl=32 (best F1)")
        print("  s16_32 - fall=16, adl=32 (fewer windows)")
        return

    if not args.stride:
        parser.print_help()
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating configs for stride: {args.stride}")
    print(f"Output: {output_dir}\n")

    for input_name in INPUTS:
        for norm_name in NORMS:
            config_name = f"{args.stride}_{input_name}_{norm_name}.yaml"
            config_content = generate_config(args.stride, input_name, norm_name)

            config_path = output_dir / config_name
            config_path.write_text(config_content)
            print(f"  {config_name}")

    print(f"\nGenerated {len(INPUTS) * len(NORMS)} configs")


if __name__ == "__main__":
    main()
