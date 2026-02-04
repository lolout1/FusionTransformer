"""
Knowledge Distillation trainer.

Extends the base Trainer class to support teacher-student training
with configurable KD losses.
"""

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kd.losses import CombinedKDLoss


class KDTrainer:
    """
    Knowledge Distillation trainer.

    Supports:
    - Skeleton-only teacher
    - Joint skeleton+IMU teacher
    - IMU-only student (for deployment)
    - Configurable KD losses (embedding, gram, COMODO)

    Integration with existing pipeline:
    - Uses same data loaders
    - Uses same evaluation metrics
    - Extends Trainer pattern from main.py
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: Optional[nn.Module] = None,
        config: Optional[Dict] = None,
        device: str = 'cuda',
    ):
        """
        Args:
            student_model: Student network (IMU-only)
            teacher_model: Teacher network (skeleton or joint) - optional
            config: KD configuration dict
            device: Device to train on
        """
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device) if teacher_model else None
        self.device = device

        # Freeze teacher if provided
        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

        # Parse config
        config = config or {}
        self.kd_config = config.get('kd', {})
        self.embed_dim = config.get('model_args', {}).get('embed_dim', 48)

        # Initialize loss
        self._init_loss()

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0

    def _init_loss(self):
        """Initialize combined KD loss from config."""
        kd = self.kd_config
        losses = kd.get('losses', {})

        # Get task loss
        loss_type = kd.get('task_loss', 'bce')
        if loss_type == 'focal':
            from utils.loss import BinaryFocalLoss
            task_loss = BinaryFocalLoss()
        else:
            task_loss = nn.BCEWithLogitsLoss()

        # Get teacher embed dim if teacher exists
        teacher_embed_dim = None
        if self.teacher is not None:
            teacher_args = kd.get('teacher_args', {})
            teacher_embed_dim = teacher_args.get('embed_dim', self.embed_dim)

        self.criterion = CombinedKDLoss(
            embed_dim=self.embed_dim,
            teacher_embed_dim=teacher_embed_dim,
            task_loss=task_loss,
            task_weight=kd.get('task_weight', 1.0),
            embedding_weight=losses.get('embedding', {}).get('weight', 1.0),
            embedding_enabled=losses.get('embedding', {}).get('enabled', True),
            gram_weight=losses.get('gram', {}).get('weight', 0.5),
            gram_enabled=losses.get('gram', {}).get('enabled', True),
            comodo_weight=losses.get('comodo', {}).get('weight', 0.5),
            comodo_enabled=losses.get('comodo', {}).get('enabled', True),
            comodo_queue_size=losses.get('comodo', {}).get('queue_size', 4096),
            comodo_tau_T=losses.get('comodo', {}).get('tau_T', 0.07),
            comodo_tau_S=losses.get('comodo', {}).get('tau_S', 0.1),
        ).to(self.device)

    def init_optimizer(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        total_epochs: int = 80,
    ):
        """Initialize optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def _unpack_batch(self, batch: Dict) -> Tuple:
        """
        Unpack batch from data loader into components.

        Supports both dict-based (new KD loader) and tuple-based (legacy) formats.
        """
        # Dict format from WindowedKDDataset
        if isinstance(batch, dict):
            # Handle both 'label' and 'labels' keys
            if 'labels' in batch:
                labels = batch['labels'].to(self.device)
            elif 'label' in batch:
                labels = batch['label'].to(self.device)
            else:
                raise KeyError("Batch missing 'label' or 'labels' key")

            # Combine acc + gyro for IMU input
            if 'acc_values' in batch and 'gyro_values' in batch:
                acc = batch['acc_values'].to(self.device)
                gyro = batch['gyro_values'].to(self.device)
                imu_data = torch.cat([acc, gyro], dim=-1)  # (B, T, 6)
                timestamps = batch.get('acc_timestamps', None)
                if timestamps is not None:
                    timestamps = timestamps.to(self.device)
            elif 'acc_values' in batch:
                imu_data = batch['acc_values'].to(self.device)
                timestamps = batch.get('acc_timestamps', None)
                if timestamps is not None:
                    timestamps = timestamps.to(self.device)
            else:
                imu_data = None
                timestamps = None

            # Skeleton data for teacher
            skeleton_data = batch.get('skeleton', None)
            if skeleton_data is not None:
                skeleton_data = skeleton_data.to(self.device)

            # Mask for valid positions
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(self.device)

            return imu_data, timestamps, skeleton_data, labels, mask

        # Legacy tuple format: (imu_data, skeleton_data, labels) or (imu_data, labels)
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                imu_data, skeleton_data, labels = batch
                skeleton_data = skeleton_data.to(self.device)
            else:
                imu_data, labels = batch
                skeleton_data = None

            imu_data = imu_data.to(self.device)
            labels = labels.to(self.device)

            # Create synthetic timestamps
            B, T, _ = imu_data.shape
            timestamps = torch.arange(T, device=self.device).float().unsqueeze(0).expand(B, -1)
            timestamps = timestamps / 30.0

            return imu_data, timestamps, skeleton_data, labels, None

        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        num_tokens: int = 64,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            num_tokens: Number of tokens for Gram KD

        Returns:
            metrics: dict with loss components and accuracy
        """
        self.student.train()
        if self.teacher is not None:
            self.teacher.eval()

        total_loss = 0.0
        loss_components = {}
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch
            imu_data, timestamps, skeleton_data, labels, mask = self._unpack_batch(batch)

            # Create timestamps if not provided
            if timestamps is None:
                B, T, _ = imu_data.shape
                timestamps = torch.arange(T, device=self.device).float().unsqueeze(0).expand(B, -1)
                timestamps = timestamps / 30.0

            # Student forward
            student_logits, student_embed = self.student(imu_data, timestamps, mask)

            # Get student tokens if Gram loss enabled
            if hasattr(self.student, 'get_tokens'):
                student_tokens = self.student.get_tokens(imu_data, timestamps, num_tokens)
            elif hasattr(self.student, 'resampler'):
                # Use resampler output directly
                student_tokens, _ = self.student.resampler(imu_data, timestamps, mask)
            else:
                student_tokens = None

            # Teacher forward (if available)
            teacher_embed = None
            teacher_tokens = None
            if self.teacher is not None and skeleton_data is not None:
                with torch.no_grad():
                    teacher_logits, teacher_embed = self.teacher(skeleton_data)
                    if hasattr(self.teacher, 'get_tokens'):
                        teacher_tokens = self.teacher.get_tokens(skeleton_data, num_tokens)

            # Compute loss
            loss, loss_dict = self.criterion(
                student_logits=student_logits,
                student_embed=student_embed,
                labels=labels,
                teacher_embed=teacher_embed,
                student_tokens=student_tokens,
                teacher_tokens=teacher_tokens,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v

            # Accuracy
            preds = (torch.sigmoid(student_logits.squeeze(-1)) > 0.5).float()
            correct += (preds == labels.float()).sum().item()
            total += labels.size(0)

        # Average metrics
        n_batches = len(train_loader)
        metrics = {
            'loss': total_loss / n_batches,
            'accuracy': correct / total if total > 0 else 0.0,
        }
        for k, v in loss_components.items():
            metrics[f'loss_{k}'] = v / n_batches

        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate on validation/test set.

        Args:
            val_loader: Validation data loader

        Returns:
            metrics: dict with loss, accuracy, precision, recall, f1
        """
        self.student.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in val_loader:
            # Unpack batch using same method as training
            imu_data, timestamps, skeleton_data, labels, mask = self._unpack_batch(batch)

            # Create timestamps if not provided
            if timestamps is None:
                B, T, _ = imu_data.shape
                timestamps = torch.arange(T, device=self.device).float().unsqueeze(0).expand(B, -1)
                timestamps = timestamps / 30.0

            # Forward
            logits, _ = self.student(imu_data, timestamps, mask)

            # Loss (task only, no KD during eval)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1), labels.float()
            )
            total_loss += loss.item()

            # Predictions
            preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        import numpy as np
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        tn = ((all_preds == 0) & (all_labels == 0)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / max(len(all_labels), 1)

        return {
            'loss': total_loss / max(len(val_loader), 1),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 80,
        patience: int = 15,
        save_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Maximum epochs
            patience: Early stopping patience
            save_dir: Directory to save checkpoints

        Returns:
            results: Training results including best metrics
        """
        if self.optimizer is None:
            self.init_optimizer(total_epochs=num_epochs)

        best_f1 = 0.0
        best_epoch = 0
        epochs_without_improvement = 0
        history = {'train': [], 'val': []}

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train'].append(train_metrics)

            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history['val'].append(val_metrics)

                # Check for improvement
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    best_epoch = epoch
                    epochs_without_improvement = 0

                    # Save best model
                    if save_dir:
                        save_dir = Path(save_dir)
                        save_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            self.student.state_dict(),
                            save_dir / 'best_student.pth'
                        )
                else:
                    epochs_without_improvement += 1

                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

                # Log
                print(f'Epoch {epoch}: train_loss={train_metrics["loss"]:.4f}, '
                      f'val_f1={val_metrics["f1"]:.4f}, best_f1={best_f1:.4f}')
            else:
                print(f'Epoch {epoch}: train_loss={train_metrics["loss"]:.4f}')

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        return {
            'best_f1': best_f1,
            'best_epoch': best_epoch,
            'history': history,
        }

    def save_checkpoint(self, path: Path):
        """Save full training state."""
        torch.save({
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'kd_config': self.kd_config,
        }, path)

    def load_checkpoint(self, path: Path):
        """Load training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)


def create_kd_trainer(config: Dict, device: str = 'cuda') -> KDTrainer:
    """
    Factory function to create KDTrainer from config.

    Args:
        config: Full experiment config
        device: Device to train on

    Returns:
        KDTrainer instance
    """
    from kd.resampler import TimestampAwareStudent
    from kd.skeleton_encoder import SkeletonTransformer, JointSkeletonIMUTeacher

    kd_config = config.get('kd', {})
    model_args = config.get('model_args', {})

    # Create student
    student_args = kd_config.get('student_args', {})
    student = TimestampAwareStudent(
        input_dim=student_args.get('input_dim', 6),
        embed_dim=model_args.get('embed_dim', 48),
        num_tokens=student_args.get('num_tokens', 64),
        num_heads=model_args.get('num_heads', 4),
        num_layers=model_args.get('num_layers', 2),
        dropout=model_args.get('dropout', 0.5),
    )

    # Create teacher
    teacher = None
    teacher_type = kd_config.get('teacher_type', 'skeleton')
    teacher_args = kd_config.get('teacher_args', {})

    if teacher_type == 'skeleton':
        teacher = SkeletonTransformer(
            embed_dim=teacher_args.get('embed_dim', 64),
            num_heads=teacher_args.get('num_heads', 4),
            num_layers=teacher_args.get('num_layers', 2),
        )
    elif teacher_type == 'joint':
        teacher = JointSkeletonIMUTeacher(
            embed_dim=teacher_args.get('embed_dim', 64),
            num_heads=teacher_args.get('num_heads', 4),
            num_layers=teacher_args.get('num_layers', 2),
        )

    # Load teacher weights if specified
    teacher_weights = kd_config.get('teacher_weights')
    if teacher_weights and teacher is not None:
        teacher.load_state_dict(torch.load(teacher_weights, map_location=device))

    return KDTrainer(
        student_model=student,
        teacher_model=teacher,
        config=config,
        device=device,
    )
