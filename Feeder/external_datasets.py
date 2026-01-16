"""
PyTorch Dataset wrappers for external fall detection datasets.

Provides a unified interface for UP-FALL and WEDA-FALL datasets
that is compatible with FusionTransformer's training loop.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional


class ExternalFallDataset(Dataset):
    """
    PyTorch Dataset for external fall detection datasets.

    Compatible with FusionTransformer's training loop (same interface as UTD_mm).

    Expects data in format:
        {
            'accelerometer': (N, T, C) numpy array,
            'labels': (N,) numpy array
        }

    Returns:
        data: dict with 'accelerometer' tensor (T, C)
        label: scalar tensor
        index: sample index
    """

    def __init__(
        self,
        data_dict: Dict[str, np.ndarray],
        include_smv: bool = False,  # SMV already in Kalman output
    ):
        """
        Initialize dataset.

        Args:
            data_dict: Dict with 'accelerometer' and 'labels' arrays
            include_smv: Add SMV channel (typically False if already included)
        """
        self.acc_data = data_dict['accelerometer']
        self.labels = data_dict['labels']
        self.include_smv = include_smv

        # Validate shapes
        if self.acc_data.ndim != 3:
            raise ValueError(f"accelerometer should be 3D (N,T,C), got {self.acc_data.ndim}D")
        if self.labels.ndim != 1:
            raise ValueError(f"labels should be 1D, got {self.labels.ndim}D")
        if self.acc_data.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"Sample count mismatch: acc={self.acc_data.shape[0]}, labels={self.labels.shape[0]}"
            )

        self.n_samples = self.acc_data.shape[0]
        self.seq_len = self.acc_data.shape[1]
        self.n_channels = self.acc_data.shape[2]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
        """
        Get a single sample.

        Returns:
            data: dict with 'accelerometer' tensor (T, C)
            label: scalar tensor (long)
            index: sample index
        """
        acc = torch.tensor(self.acc_data[index], dtype=torch.float32)

        if self.include_smv:
            # Compute SMV and prepend
            smv = torch.sqrt(torch.sum(acc[:, :3] ** 2, dim=1, keepdim=True))
            smv = smv - smv.mean()
            acc = torch.cat([smv, acc], dim=-1)

        data = {'accelerometer': acc}
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return data, label, index

    def get_class_distribution(self) -> Dict[int, int]:
        """Get count of samples per class."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced data."""
        dist = self.get_class_distribution()
        total = sum(dist.values())
        weights = [total / (len(dist) * dist.get(i, 1)) for i in range(max(dist.keys()) + 1)]
        return torch.tensor(weights, dtype=torch.float32)


class ExternalFallDatasetBuilder:
    """
    Builder class compatible with FusionTransformer's data preparation.

    Wraps external dataset loaders (UPFallLoader, WEDAFallLoader) to provide
    the same interface as DatasetBuilder from utils/loader.py.
    """

    def __init__(self, loader, arg):
        """
        Initialize builder.

        Args:
            loader: UPFallLoader or WEDAFallLoader instance
            arg: Namespace with config arguments
        """
        self.loader = loader
        self.arg = arg
        self.data = {}
        self.subjects = loader.subjects

    def make_dataset(self, subjects, fuse=False):
        """
        Compatibility method - data is loaded on demand.
        """
        pass

    def normalization(self):
        """
        Compatibility method - normalization handled in loader.
        """
        return self.data

    def split_by_subjects(
        self,
        train_subjects: list,
        val_subjects: list,
        test_subjects: list,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Split data by subjects (LOSO).

        Args:
            train_subjects: Subject IDs for training
            val_subjects: Subject IDs for validation
            test_subjects: Subject IDs for testing

        Returns:
            train_data, val_data, test_data dicts
        """
        # Prepare LOSO fold using the loader
        test_subject = test_subjects[0] if test_subjects else None

        if test_subject is None:
            raise ValueError("test_subjects must contain at least one subject")

        fold_data = self.loader.prepare_loso_fold(
            test_subject=test_subject,
            val_subjects=val_subjects,
            train_only_subjects=None  # Use config if needed
        )

        return fold_data['train'], fold_data['val'], fold_data['test']


def create_external_dataloaders(
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create PyTorch DataLoaders from split data.

    Args:
        train_data: Dict with 'accelerometer', 'labels'
        val_data: Dict with 'accelerometer', 'labels'
        test_data: Dict with 'accelerometer', 'labels'
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = ExternalFallDataset(train_data)
    val_dataset = ExternalFallDataset(val_data)
    test_dataset = ExternalFallDataset(test_data)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("External Dataset Wrapper Test")
    print("=" * 60)

    # Create dummy data
    N, T, C = 100, 128, 7
    dummy_data = {
        'accelerometer': np.random.randn(N, T, C).astype(np.float32),
        'labels': np.random.randint(0, 2, size=N).astype(np.int64),
    }

    # Create dataset
    dataset = ExternalFallDataset(dummy_data)
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    print(f"Class weights: {dataset.get_class_weights()}")

    # Test __getitem__
    data, label, idx = dataset[0]
    print(f"\nSample 0:")
    print(f"  acc shape: {data['accelerometer'].shape}")
    print(f"  label: {label}")
    print(f"  index: {idx}")

    # Test DataLoader
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    batch_data, batch_labels, batch_idx = next(iter(loader))
    print(f"\nBatch:")
    print(f"  acc shape: {batch_data['accelerometer'].shape}")
    print(f"  labels shape: {batch_labels.shape}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
