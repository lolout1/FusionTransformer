import torch 
import torch.nn.functional as F
import torch.nn as nn

class DistillationLoss(nn.Module):
    '''
    Knowledge Distillation Loss
    '''
    def __init__(self, temperature=2.0, alpha=0.3):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Hard loss (cross entropy between student predictions and ground truth)
            soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.temperature**2)

            # Calculate the true label loss
            label_loss = self.criterion(student_logits, labels)

            # Weighted sum of the two losses
            loss = self.alpha * soft_targets_loss + (1-self.alpha) * label_loss
            return loss

