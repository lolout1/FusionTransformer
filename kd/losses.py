"""
Knowledge Distillation losses for skeleton→IMU transfer.

Tier 1 (Alignment-Free):
- EmbeddingKDLoss: Cosine distance between pooled embeddings
- GramKDLoss: MSE between token similarity matrices
- COMODOLoss: Distribution matching via embedding queue

Tier 2 (Soft Alignment):
- SoftDTWKDLoss: Differentiable DTW between token sequences
- SinkhornKDLoss: Optimal transport matching
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingKDLoss(nn.Module):
    """
    Global embedding KD via cosine similarity.

    Minimizes 1 - cos(g_T, g_S) where g_T and g_S are pooled embeddings.
    Completely alignment-free - only requires paired samples.
    """

    def __init__(self, normalize: bool = True, temperature: float = 1.0):
        super().__init__()
        self.normalize = normalize
        self.temperature = temperature

    def forward(
        self,
        student_embed: torch.Tensor,
        teacher_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_embed: (B, D) student pooled embedding
            teacher_embed: (B, D) teacher pooled embedding

        Returns:
            loss: scalar
        """
        if self.normalize:
            student_embed = F.normalize(student_embed, dim=-1)
            teacher_embed = F.normalize(teacher_embed, dim=-1)

        # Cosine similarity: (B,)
        cos_sim = (student_embed * teacher_embed).sum(dim=-1)

        # Loss = 1 - cos_sim, scaled by temperature
        loss = (1 - cos_sim) / self.temperature

        return loss.mean()


class GramKDLoss(nn.Module):
    """
    Relational KD via Gram matrix matching.

    Computes L×L Gram matrices (token similarities) for teacher and student,
    then minimizes MSE between them.

    This transfers structural relationships between tokens without requiring
    explicit alignment - the model learns to preserve relative similarities.
    """

    def __init__(self, normalize_tokens: bool = True, normalize_gram: bool = False):
        super().__init__()
        self.normalize_tokens = normalize_tokens
        self.normalize_gram = normalize_gram

    def forward(
        self,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_tokens: (B, L, D) student token sequence
            teacher_tokens: (B, L, D) teacher token sequence

        Returns:
            loss: scalar

        Note:
            Both tensors must have same L (number of tokens).
            Use adaptive pooling to match if needed.
        """
        assert student_tokens.shape[1] == teacher_tokens.shape[1], \
            f"Token counts must match: {student_tokens.shape[1]} vs {teacher_tokens.shape[1]}"

        # Normalize tokens if requested
        if self.normalize_tokens:
            student_tokens = F.normalize(student_tokens, dim=-1)
            teacher_tokens = F.normalize(teacher_tokens, dim=-1)

        # Compute Gram matrices: (B, L, L)
        G_S = torch.bmm(student_tokens, student_tokens.transpose(1, 2))
        G_T = torch.bmm(teacher_tokens, teacher_tokens.transpose(1, 2))

        # Optionally normalize Gram matrices
        if self.normalize_gram:
            G_S = G_S / G_S.shape[1]
            G_T = G_T / G_T.shape[1]

        # MSE loss
        loss = F.mse_loss(G_S, G_T)

        return loss


class COMODOLoss(nn.Module):
    """
    Cross-modal distribution distillation via embedding queue.

    Maintains a FIFO queue of teacher embeddings. For each sample:
    1. Compute teacher distribution: p_T = softmax(z_T @ Q^T / tau_T)
    2. Compute student distribution: p_S = softmax(z_S @ Q^T / tau_S)
    3. Minimize cross-entropy: -sum(p_T * log(p_S))

    This is completely alignment-free and label-free.
    """

    def __init__(
        self,
        embed_dim: int,
        queue_size: int = 4096,
        tau_T: float = 0.07,
        tau_S: float = 0.1,
        momentum: float = 0.0,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            queue_size: Number of embeddings in queue
            tau_T: Temperature for teacher distribution (lower = sharper)
            tau_S: Temperature for student distribution (higher = smoother for learning)
            momentum: If > 0, use momentum update for queue (0 = pure FIFO)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.queue_size = queue_size
        self.tau_T = tau_T
        self.tau_S = tau_S
        self.momentum = momentum

        # Register queue as buffer (not parameter)
        self.register_buffer('queue', torch.randn(queue_size, embed_dim))
        self.queue = F.normalize(self.queue, dim=-1)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # Track if queue is warmed up
        self.register_buffer('queue_filled', torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def update_queue(self, teacher_embed: torch.Tensor):
        """
        Update queue with new teacher embeddings (FIFO).

        Args:
            teacher_embed: (B, D) teacher embeddings to add
        """
        teacher_embed = F.normalize(teacher_embed.detach(), dim=-1)
        batch_size = teacher_embed.shape[0]

        ptr = int(self.queue_ptr)
        end_ptr = ptr + batch_size

        if end_ptr <= self.queue_size:
            self.queue[ptr:end_ptr] = teacher_embed
        else:
            # Wrap around
            first_part = self.queue_size - ptr
            self.queue[ptr:] = teacher_embed[:first_part]
            self.queue[:batch_size - first_part] = teacher_embed[first_part:]

        # Update pointer
        self.queue_ptr[0] = (end_ptr % self.queue_size)

        # Mark queue as filled once we've gone through it once
        if end_ptr >= self.queue_size:
            self.queue_filled[0] = True

    def forward(
        self,
        student_embed: torch.Tensor,
        teacher_embed: torch.Tensor,
        update_queue: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            student_embed: (B, D) student embedding
            teacher_embed: (B, D) teacher embedding
            update_queue: Whether to update queue after loss computation

        Returns:
            loss: scalar cross-entropy loss
        """
        # Normalize embeddings
        student_embed = F.normalize(student_embed, dim=-1)
        teacher_embed = F.normalize(teacher_embed, dim=-1)

        # Get normalized queue
        Q = F.normalize(self.queue, dim=-1)  # (K, D)

        # Compute logits against queue
        logits_T = teacher_embed @ Q.T / self.tau_T  # (B, K)
        logits_S = student_embed @ Q.T / self.tau_S  # (B, K)

        # Teacher distribution (soft targets)
        p_T = F.softmax(logits_T, dim=-1)

        # Student log-probabilities
        log_p_S = F.log_softmax(logits_S, dim=-1)

        # Cross-entropy loss: -sum(p_T * log(p_S))
        loss = -(p_T * log_p_S).sum(dim=-1).mean()

        # Update queue with teacher embeddings
        if update_queue:
            self.update_queue(teacher_embed)

        return loss

    def is_ready(self) -> bool:
        """Check if queue has been filled at least once."""
        return bool(self.queue_filled.item())


class CombinedKDLoss(nn.Module):
    """
    Combined KD loss with configurable components.

    Supports:
    - Task loss (BCE, Focal, etc.)
    - Embedding KD loss
    - Gram KD loss
    - COMODO loss

    Each component can be enabled/disabled via config.
    Handles dimension mismatch between teacher and student via learned projection.
    """

    def __init__(
        self,
        embed_dim: int,
        teacher_embed_dim: Optional[int] = None,
        task_loss: Optional[nn.Module] = None,
        task_weight: float = 1.0,
        embedding_weight: float = 1.0,
        embedding_enabled: bool = True,
        gram_weight: float = 0.5,
        gram_enabled: bool = True,
        comodo_weight: float = 0.5,
        comodo_enabled: bool = True,
        comodo_queue_size: int = 4096,
        comodo_tau_T: float = 0.07,
        comodo_tau_S: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.teacher_embed_dim = teacher_embed_dim or embed_dim

        # Projection layer if dimensions differ
        if self.teacher_embed_dim != embed_dim:
            self.teacher_proj = nn.Linear(self.teacher_embed_dim, embed_dim)
        else:
            self.teacher_proj = None

        # Task loss (default: BCE)
        if task_loss is None:
            task_loss = nn.BCEWithLogitsLoss()
        self.task_loss = task_loss
        self.task_weight = task_weight

        # KD components
        self.embedding_enabled = embedding_enabled
        self.embedding_weight = embedding_weight
        if embedding_enabled:
            self.embedding_loss = EmbeddingKDLoss()

        self.gram_enabled = gram_enabled
        self.gram_weight = gram_weight
        if gram_enabled:
            self.gram_loss = GramKDLoss()

        self.comodo_enabled = comodo_enabled
        self.comodo_weight = comodo_weight
        if comodo_enabled:
            self.comodo_loss = COMODOLoss(
                embed_dim=embed_dim,
                queue_size=comodo_queue_size,
                tau_T=comodo_tau_T,
                tau_S=comodo_tau_S,
            )

    def forward(
        self,
        student_logits: torch.Tensor,
        student_embed: torch.Tensor,
        labels: torch.Tensor,
        teacher_embed: Optional[torch.Tensor] = None,
        student_tokens: Optional[torch.Tensor] = None,
        teacher_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            student_logits: (B, 1) student predictions
            student_embed: (B, D) student pooled embedding
            labels: (B,) ground truth labels
            teacher_embed: (B, D) teacher embedding (optional, for KD)
            student_tokens: (B, L, D) student tokens (optional, for Gram)
            teacher_tokens: (B, L, D) teacher tokens (optional, for Gram)

        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # Task loss
        task_l = self.task_loss(student_logits.squeeze(-1), labels.float())
        loss_dict['task'] = task_l.item()
        total_loss = total_loss + self.task_weight * task_l

        # KD losses (only if teacher provided)
        if teacher_embed is not None:
            # Project teacher embedding if dimensions differ
            if self.teacher_proj is not None:
                teacher_embed_proj = self.teacher_proj(teacher_embed)
            else:
                teacher_embed_proj = teacher_embed

            # Embedding KD
            if self.embedding_enabled:
                emb_l = self.embedding_loss(student_embed, teacher_embed_proj)
                loss_dict['embedding'] = emb_l.item()
                total_loss = total_loss + self.embedding_weight * emb_l

            # COMODO (uses projected embeddings)
            if self.comodo_enabled:
                comodo_l = self.comodo_loss(student_embed, teacher_embed_proj)
                loss_dict['comodo'] = comodo_l.item()
                total_loss = total_loss + self.comodo_weight * comodo_l

        # Gram KD (requires tokens)
        if self.gram_enabled and student_tokens is not None and teacher_tokens is not None:
            # Project teacher tokens if needed
            if self.teacher_proj is not None and teacher_tokens.shape[-1] != student_tokens.shape[-1]:
                teacher_tokens_proj = self.teacher_proj(teacher_tokens)
            else:
                teacher_tokens_proj = teacher_tokens
            gram_l = self.gram_loss(student_tokens, teacher_tokens_proj)
            loss_dict['gram'] = gram_l.item()
            total_loss = total_loss + self.gram_weight * gram_l

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


# =============================================================================
# Tier 2 Losses (Soft Alignment) - Implement after Tier 1 is stable
# =============================================================================

class SoftDTWKDLoss(nn.Module):
    """
    Soft-DTW based token KD.

    Uses differentiable DTW to softly align teacher and student token sequences.
    More expensive than Gram but captures temporal structure.
    """

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_tokens: (B, L, D)
            teacher_tokens: (B, L, D)

        Returns:
            loss: scalar
        """
        # Compute pairwise distances
        # (B, L_S, L_T)
        dist = torch.cdist(student_tokens, teacher_tokens, p=2)

        # Soft-DTW forward pass
        # This is a simplified implementation - for production use tslearn or pytorch-soft-dtw
        B, L_S, L_T = dist.shape

        # Initialize DP table
        R = torch.full((B, L_S + 1, L_T + 1), float('inf'), device=dist.device)
        R[:, 0, 0] = 0

        for i in range(1, L_S + 1):
            for j in range(1, L_T + 1):
                r_prev = torch.stack([
                    R[:, i-1, j-1],
                    R[:, i-1, j],
                    R[:, i, j-1],
                ], dim=-1)  # (B, 3)

                # Soft-min
                softmin = -self.gamma * torch.logsumexp(-r_prev / self.gamma, dim=-1)
                R[:, i, j] = dist[:, i-1, j-1] + softmin

        return R[:, L_S, L_T].mean()


class SinkhornKDLoss(nn.Module):
    """
    Optimal Transport based token KD via Sinkhorn iterations.

    Finds soft assignment between teacher and student tokens that
    minimizes transport cost. More flexible than DTW (no monotonicity).
    """

    def __init__(self, reg: float = 0.1, n_iter: int = 10):
        super().__init__()
        self.reg = reg
        self.n_iter = n_iter

    def forward(
        self,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_tokens: (B, L_S, D)
            teacher_tokens: (B, L_T, D)

        Returns:
            loss: scalar OT distance
        """
        B, L_S, _ = student_tokens.shape
        _, L_T, _ = teacher_tokens.shape

        # Cost matrix: pairwise L2 distance
        C = torch.cdist(student_tokens, teacher_tokens, p=2)  # (B, L_S, L_T)

        # Uniform marginals
        a = torch.ones(B, L_S, device=C.device) / L_S
        b = torch.ones(B, L_T, device=C.device) / L_T

        # Sinkhorn iterations
        K = torch.exp(-C / self.reg)  # (B, L_S, L_T)
        u = torch.ones_like(a)

        for _ in range(self.n_iter):
            v = b / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
            u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)

        # Transport plan
        P = u.unsqueeze(-1) * K * v.unsqueeze(1)  # (B, L_S, L_T)

        # OT loss = <P, C>
        loss = (P * C).sum(dim=(1, 2)).mean()

        return loss
