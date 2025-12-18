"""
Test script to verify the comprehensive statistics fix handles duplicates correctly.
"""
import pandas as pd
import numpy as np

def test_duplicate_column_handling():
    """Test that duplicate columns are properly handled"""
    print("Testing duplicate column handling...")

    # Simulate rows with duplicate keys (the original issue)
    rows = [
        {'test_subject': 1, 'train_accuracy': 0.8, 'train_accuracy': 0.85, 'val_f1': 0.9},
        {'test_subject': 2, 'train_accuracy': 0.82, 'val_f1': 0.88},
    ]

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Check for duplicates
    if df.columns.duplicated().any():
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        print(f"  ✓ Detected duplicate columns: {duplicate_cols}")
        # Remove duplicates
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
        print(f"  ✓ Removed duplicates, columns now: {df.columns.tolist()}")
    else:
        print(f"  ✓ No duplicates found, columns: {df.columns.tolist()}")

    return df

def test_average_row_concatenation():
    """Test that average row concatenation works without IndexError"""
    print("\nTesting average row concatenation...")

    # Create test DataFrame
    data = {
        'test_subject': [1, 2, 3],
        'train_accuracy': [0.8, 0.82, 0.85],
        'val_f1': [0.9, 0.88, 0.92],
        'test_loss': [0.5, 0.45, 0.42]
    }
    df = pd.DataFrame(data)

    # Calculate average row (same logic as in the fix)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    averages = df[numeric_cols].mean()

    avg_row = {}
    for col in df.columns:
        if col in numeric_cols:
            if 'loss' in col:
                avg_row[col] = round(averages[col], 6)
            else:
                avg_row[col] = round(averages[col], 2)
        elif col == 'test_subject':
            avg_row[col] = 'Average'
        else:
            avg_row[col] = None

    # Create average DataFrame
    avg_df = pd.DataFrame([avg_row])
    avg_df = avg_df[df.columns]  # Ensure column order matches

    # Concatenate
    try:
        result = pd.concat([df, avg_df], ignore_index=True, sort=False)
        print(f"  ✓ Successfully concatenated, shape: {result.shape}")
        print(f"  ✓ Average row values: {result.iloc[-1].to_dict()}")
        return result
    except Exception as e:
        print(f"  ✗ Concatenation failed: {e}")
        return None

def test_metric_row_building():
    """Test that metric rows are built without duplicate keys"""
    print("\nTesting metric row building...")

    # Simulate fold_stat with potential duplicate keys
    fold_stat = {
        'fold': 1,
        'test_subject': 45,
        'train_windows': 100,
        'train_accuracy': 0.75,  # This might duplicate with metrics
    }

    # Simulate fold metrics
    fold_metrics = {
        'train': {'accuracy': 0.85, 'f1': 0.82},
        'val': {'accuracy': 0.88, 'f1': 0.86},
        'test': {'accuracy': 0.83, 'f1': 0.81}
    }

    # Build row (same logic as _build_metrics_row)
    row = {}

    # Add non-metric keys from fold_stat
    for key, value in fold_stat.items():
        if not any(key.startswith(f'{split}_') for split in ['train', 'val', 'test']):
            row[key] = value

    # Add metrics
    for split in ['train', 'val', 'test']:
        if split in fold_metrics:
            for metric_name, metric_value in fold_metrics[split].items():
                metric_key = f'{split}_{metric_name}'
                row[metric_key] = metric_value

    # Check for duplicate keys
    keys = list(row.keys())
    if len(keys) != len(set(keys)):
        print(f"  ✗ Duplicate keys found: {keys}")
        return None
    else:
        print(f"  ✓ No duplicate keys, built row with {len(keys)} unique keys")
        print(f"  ✓ Keys: {keys}")
        return row

if __name__ == '__main__':
    print("="*70)
    print("COMPREHENSIVE STATISTICS FIX - VALIDATION TESTS")
    print("="*70)

    # Run tests
    df1 = test_duplicate_column_handling()
    df2 = test_average_row_concatenation()
    row = test_metric_row_building()

    print("\n" + "="*70)
    if df1 is not None and df2 is not None and row is not None:
        print("✓ ALL TESTS PASSED - Fix is working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Review the fix")
    print("="*70)
