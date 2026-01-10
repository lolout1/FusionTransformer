import torch 
import torch.nn.functional as F
import torch.nn as nn

class DistillationLoss(nn.Module):
    '''
    Knowledge Distillation Loss
    '''
    def __init__(self, temperature=4.5, alpha=.6, pos_weigths = None):
        # .4 for phone gyroscope
        # .6 for watch accelerometer
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        # self.criterion = nn.CrossEntropyLoss()
        #self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weigths)
        self.bce = BinaryFocalLoss(alpha=0.6)
        #self.embedding_loss = nn.CosineEmbeddingLoss()
        self.embeddin_loss = nn.KLDivLoss(reduction="none")
        self.epsilon = 1e-3
        

    def forward(self, student_logits, teacher_logits, labels, teacher_features, student_features, target):
        # Hard loss (cross entropy between student predictions and ground truth)
            #soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
            #soft_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            #soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.temperature**2)

            ####### feature based ########

            # #student_features = F.layer_norm(student_features, normalized_shape=(128,))
            # teacher_features = F.avg_pool1d(teacher_features, kernel_size=teacher_features.shape[-1], stride=1)
            # student_features = F.avg_pool1d(student_features, kernel_size=student_features.shape[-1], stride = 1)
            
            # #cosine loss
            # flatten_student = torch.flatten(student_features, 1)
            # flatten_teacher = torch.flatten(teacher_features, 1)
            
            #cosine_loss = self.embedding_loss(flatten_teacher, flatten_student, target)

            ## kldiv between teacher and student features#####
            soft_teacher = nn.functional.softmax(teacher_features/self.temperature, dim=-1)
            soft_prob = nn.functional.log_softmax(student_features/self.temperature, dim=-1)

            ## comparing only correct knowledge ## 
            teacher_pred = (torch.sigmoid(teacher_logits) > 0.5).int().squeeze(1)
            correct_mask = (teacher_pred ==labels).float()
            weights =(1.0 /1.5) * correct_mask + (.5/1.5)* (1 - correct_mask)
            # weights = weights.view(-1, 1, 1)
            cosine_loss = (weights.view(-1, 1,1) * self.embeddin_loss(soft_prob,soft_teacher)).mean()
            #print(cosine_loss)
            # Calculate the true label loss for multiclass
            #label_loss = self.criterion(student_logits, labels)
            ### for feature based#####
            
            # ### with one logit  
            label_loss = self.bce(student_logits.squeeze(1), labels.float())
            loss = self.alpha * cosine_loss + ( 1 - self.alpha) * label_loss
            # # Weighted sum of the two losses
            # loss = self.alpha * soft_targets_loss + (1-self.alpha) * label_loss

            return loss

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        """
        :param alpha: Weighting factor for positive class (float)
        :param gamma: Focusing parameter for modulating factor (1 - p_t)
        :param reduction: 'none' | 'mean' | 'sum'
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        :param logits: raw predictions (before sigmoid), shape [batch_size]
        :param targets: binary ground truth (0 or 1), shape [batch_size]
        """
        prob = torch.sigmoid(logits)
        targets = targets.float()
        
        pt = torch.where(targets == 1, prob, 1 - prob)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
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


