import torch 
import torch.nn.functional as F
import torch.nn as nn



class SemanticLoss(nn.Module):
    def __init__(self, T = 2, alpha = 0.7):
        super(SemanticLoss, self).__init__()
        self.T = T
        self.focal_loss = FocalLoss(alpha = 0.25, gamma=2)
        self.kd_loss = nn.KLDivLoss(reduction = 'batchmean', log_target = True).cuda()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.alpha = alpha
    
    def distillation_loss(self,logits, teacher_logits, labels):
       
        #Softmax of student prediciton
        pred_hard = F.softmax(logits, dim = 1)
        pred_soft = F.log_softmax(logits/self.T, dim = 1)

        #Softmax of teacher prediction
        teacher_soft = F.log_softmax(teacher_logits/self.T, dim = 1)

        #KLDivergence of this two
        kl_div = self.kd_loss(pred_soft, teacher_soft) * (self.alpha * self.T * self.T )
        # #cross entropy loss 
        # loss_y_label = F.cross_entropy(pred, labels) * (1.0 - alpha)

        #focal loss
        loss_y_label = self.cross_entropy(logits, labels) * (1 - self.alpha)
        distill_loss = kl_div + loss_y_label

        return distill_loss
    
    def angular_dist(self,student_pred, teacher_pred):

        # do I need to calculate gradients for the variables associated with student
        with torch.no_grad():
            td = (teacher_pred.unsqueeze(0) - teacher_pred.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
	
		#flatenning the prediction
        sd = (student_pred.unsqueeze(0) - student_pred.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
		# computing angular correlation between the norm_sd
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss
    
    def pdist(self,e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res
    
    def distance(self,student, teacher):
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss

    def forward(self,stud_logits, teacher_logits, labels):
        # gamma = 0.1
        # beta = 0.2
        # sigma = 1 - gamma - beta
        kd_loss = self.distillation_loss(logits = stud_logits, labels = labels, teacher_logits = teacher_logits)
        # y = F.log_softmax(stud_logits, dim = 1)
        # teacher_y = F.log_softmax(teacher_logits, dim = 1)
        # angular_loss = self.angular_dist(y, teacher_y)
        # dist_loss = self.distance(y, teacher_y)

        # loss = (sigma*kd_loss) + (beta*angular_loss) + (gamma*dist_loss)
        return kd_loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Compute the softmax probability
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

class MaskedBCE(nn.Module):
    def __init__(self,output_device, beta):
         super().__init__()
         self.slim_penalty = lambda var: torch.abs(var).sum().cuda()
         self.beta = beta
         self.criterion = nn.CrossEntropyLoss().cuda(self.output_device)
    
    def forward(self, masks, logits, targets):
        bce_loss = self.criterion(logits, targets)
        slim_loss = 0
        for mask in masks: 
            slim_loss += sum([self.slim_penalty(m) for m in mask])
        loss = bce_loss + (self.beta*slim_loss)

        return loss

class BCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.critetrion = nn.CrossEntropyLoss().cuda()
    
    def forward(self, masks, logits, targets):
        bce_loss = self.critetrion(logits, targets)
        return bce_loss

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, mask, y_pred, y_true):
        residual = torch.abs(y_true - torch.argmax(y_pred, dim = 1))
        small_res = 0.5 * residual ** 2
        large_res = self.delta * (residual - 0.5 * self.delta)
        loss = torch.where(residual < self.delta, small_res, large_res)
        return loss