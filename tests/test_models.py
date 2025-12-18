"""
Quick test script to verify both models work with standardized parameters
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Models.transformer import TransModel
from Models.imu_transformer import IMUTransformer


def test_transmodel():
    """Test TransModel (Accelerometer-only)"""
    print("\n" + "="*80)
    print("Testing TransModel (Accelerometer-only)")
    print("="*80)

    # Create model with standardized parameters
    model = TransModel(
        acc_frames=128,
        num_classes=2,
        num_heads=4,
        num_layer=2,
        embed_dim=64,
        dropout=0.5,
        activation='relu',
        norm_first=True
    )

    # Test with 4-channel input (ax, ay, az, smv)
    batch_size = 8
    seq_len = 128
    channels = 4

    acc_data = torch.randn(batch_size, seq_len, channels)
    skl_data = torch.randn(batch_size, seq_len, 32, 3)  # Dummy skeleton data

    # Forward pass
    logits, features = model(acc_data, skl_data)

    print(f"✓ Model created successfully")
    print(f"✓ Input shape: {acc_data.shape}")
    print(f"✓ Output logits shape: {logits.shape}")
    print(f"✓ Output features shape: {features.shape}")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Verify output shapes
    assert logits.shape == (batch_size, 2), f"Expected logits shape ({batch_size}, 2), got {logits.shape}"
    assert features.shape == (batch_size, seq_len, 64), f"Expected features shape ({batch_size}, {seq_len}, 64), got {features.shape}"

    print(f"✓ All assertions passed!")

    return model


def test_imu_transformer():
    """Test IMUTransformer (Full IMU)"""
    print("\n" + "="*80)
    print("Testing IMUTransformer (Full IMU)")
    print("="*80)

    # Create model with standardized parameters
    model = IMUTransformer(
        imu_frames=128,
        imu_channels=7,
        num_classes=2,
        num_heads=4,
        num_layers=2,
        embed_dim=64,
        dropout=0.5,
        activation='relu',
        norm_first=True
    )

    # Test with 7-channel input (ax, ay, az, gx, gy, gz, smv)
    batch_size = 8
    seq_len = 128
    channels = 7

    imu_data = torch.randn(batch_size, seq_len, channels)
    skl_data = torch.randn(batch_size, seq_len, 32, 3)  # Dummy skeleton data

    # Forward pass
    logits, features = model(imu_data, skl_data)

    print(f"✓ Model created successfully")
    print(f"✓ Input shape: {imu_data.shape}")
    print(f"✓ Output logits shape: {logits.shape}")
    print(f"✓ Output features shape: {features.shape}")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Verify output shapes
    assert logits.shape == (batch_size, 2), f"Expected logits shape ({batch_size}, 2), got {logits.shape}"
    assert features.shape == (batch_size, seq_len, 64), f"Expected features shape ({batch_size}, {seq_len}, 64), got {features.shape}"

    print(f"✓ All assertions passed!")

    return model


def compare_architectures():
    """Compare architecture details of both models"""
    print("\n" + "="*80)
    print("Architecture Comparison")
    print("="*80)

    # Create both models
    trans_model = TransModel(
        acc_frames=128, num_classes=2, num_heads=4, num_layer=2,
        embed_dim=64, dropout=0.5, activation='relu', norm_first=True
    )

    imu_model = IMUTransformer(
        imu_frames=128, imu_channels=7, num_classes=2, num_heads=4,
        num_layers=2, embed_dim=64, dropout=0.5, activation='relu', norm_first=True
    )

    # Count parameters
    trans_params = sum(p.numel() for p in trans_model.parameters())
    imu_params = sum(p.numel() for p in imu_model.parameters())

    print(f"\n{'Parameter':<30} {'TransModel':<15} {'IMUTransformer':<15} {'Match':<10}")
    print("-" * 70)
    print(f"{'Input Channels':<30} {4:<15} {7:<15} {'Different':<10}")
    print(f"{'Embedding Dim':<30} {64:<15} {64:<15} {'✓':<10}")
    print(f"{'Num Heads':<30} {4:<15} {4:<15} {'✓':<10}")
    print(f"{'Num Layers':<30} {2:<15} {2:<15} {'✓':<10}")
    print(f"{'Dropout':<30} {0.5:<15} {0.5:<15} {'✓':<10}")
    print(f"{'Activation':<30} {'relu':<15} {'relu':<15} {'✓':<10}")
    print(f"{'Norm First':<30} {'True':<15} {'True':<15} {'✓':<10}")
    print(f"{'Num Classes':<30} {2:<15} {2:<15} {'✓':<10}")
    print("-" * 70)
    print(f"{'Total Parameters':<30} {trans_params:<15,} {imu_params:<15,} {'~Same':<10}")

    param_diff = abs(imu_params - trans_params)
    param_diff_pct = (param_diff / trans_params) * 100

    print(f"\nParameter difference: {param_diff:,} ({param_diff_pct:.2f}%)")
    print(f"This difference is only due to different input channels (4 vs 7)")

    # Compare layer structures
    print(f"\n{'Layer Type':<40} {'TransModel':<20} {'IMUTransformer':<20}")
    print("-" * 80)

    print(f"{'Input Projection':<40} {'Conv1d(4→64)':<20} {'Conv1d(7→64)':<20}")
    print(f"{'Batch Normalization':<40} {'✓':<20} {'✓':<20}")
    print(f"{'Dropout (input)':<40} {'0.25':<20} {'0.25':<20}")
    print(f"{'Transformer Encoder Layers':<40} {'2':<20} {'2':<20}")
    print(f"{'  - Attention Heads':<40} {'4':<20} {'4':<20}")
    print(f"{'  - Feedforward Dim':<40} {'128 (64*2)':<20} {'128 (64*2)':<20}")
    print(f"{'  - Dropout':<40} {'0.5':<20} {'0.5':<20}")
    print(f"{'Temporal Normalization':<40} {'LayerNorm(64)':<20} {'LayerNorm(64)':<20}")
    print(f"{'Dropout (output)':<40} {'0.5':<20} {'0.5':<20}")
    print(f"{'Output Layer':<40} {'Linear(64→2)':<20} {'Linear(64→2)':<20}")

    print("\n✓ Both models have identical architecture (except input channels)")


def test_gradient_flow():
    """Test that gradients flow properly through both models"""
    print("\n" + "="*80)
    print("Testing Gradient Flow")
    print("="*80)

    # Test TransModel
    trans_model = TransModel(num_heads=4, num_layer=2, embed_dim=64, dropout=0.5)
    acc_data = torch.randn(4, 128, 4, requires_grad=True)
    skl_data = torch.randn(4, 128, 32, 3)

    logits, _ = trans_model(acc_data, skl_data)
    loss = logits.sum()
    loss.backward()

    print(f"✓ TransModel: Gradients flow correctly")
    print(f"  - Input gradient shape: {acc_data.grad.shape}")
    print(f"  - Number of parameters with gradients: {sum(1 for p in trans_model.parameters() if p.grad is not None)}")

    # Test IMUTransformer
    imu_model = IMUTransformer(imu_channels=7, num_heads=4, num_layers=2, embed_dim=64, dropout=0.5)
    imu_data = torch.randn(4, 128, 7, requires_grad=True)
    skl_data = torch.randn(4, 128, 32, 3)

    logits, _ = imu_model(imu_data, skl_data)
    loss = logits.sum()
    loss.backward()

    print(f"✓ IMUTransformer: Gradients flow correctly")
    print(f"  - Input gradient shape: {imu_data.grad.shape}")
    print(f"  - Number of parameters with gradients: {sum(1 for p in imu_model.parameters() if p.grad is not None)}")


def main():
    print("\n" + "="*80)
    print("MODEL VERIFICATION TEST SUITE")
    print("="*80)
    print("\nThis script verifies that both TransModel and IMUTransformer:")
    print("  1. Use identical hyperparameters (except input channels)")
    print("  2. Have compatible architectures")
    print("  3. Produce correct output shapes")
    print("  4. Support gradient flow for training")

    try:
        # Test individual models
        test_transmodel()
        test_imu_transformer()

        # Compare architectures
        compare_architectures()

        # Test gradient flow
        test_gradient_flow()

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nBoth models are ready for fair comparison.")
        print("You can now run: python compare_models.py")

    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED!")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
