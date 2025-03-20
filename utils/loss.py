import torch 
import torch.nn.functional as F
import torch.nn as nn

class DistillationLoss(nn.Module):
    '''
    Knowledge Distillation Loss
    '''
    def __init__(self, temperature=2.0, alpha=0.75):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()
        self.embedding_loss = nn.CosineEmbeddingLoss()

    def forward(self, student_logits, teacher_logits, labels, teacher_features, student_features, target):
        # Hard loss (cross entropy between student predictions and ground truth)
            soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.temperature**2)

            ####### feature based ########
            # teacher_features = F.avg_pool1d(teacher_features, kernel_size=teacher_features.shape[-1], stride=1)
            # student_features = F.avg_pool1d(student_features, kernel_size=student_features.shape[-1], stride = 1)
            
            # flatten_student = torch.flatten(student_features, 1)
            # flatten_teacher = torch.flatten(teacher_features, 1)
            
            # cosine_loss = self.embedding_loss(flatten_teacher, flatten_student, target)
            # Calculate the true label loss
            label_loss = self.criterion(student_logits, labels)
            #loss = self.alpha * cosine_loss + ( 1 - self.alpha) * label_loss

            # Weighted sum of the two losses
            loss = self.alpha * soft_targets_loss + (1-self.alpha) * label_loss


            return loss


if __name__ == "__main__":
     loss = DistillationLoss()
     teacher_logits = torch.rand(size = (1, 2))
     student_logits = torch.rand(size = (1, 2))
     labels = torch.rand(size=(1,1))
     teacher_features = torch.rand(size= (1, 32, 16))
     student_features = torch.rand(size = (1, 32, 128))
     target = torch.ones(1)
     loss.forward(student_logits, teacher_logits, labels, teacher_features, student_features, target)


