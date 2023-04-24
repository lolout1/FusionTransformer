import torch 
import torch.nn.functional as F
import torch.nn as nn



class SemanticLoss(nn.Module):
    def __init__(self):
        super(SemanticLoss, self).__init__()
    
    def distillation_loss(self,pred, labels, teacher_pred, T, alpha):
        
        #Softmax of student prediciton 
        pred_soft = F.log_softmax(pred/T, dim = 1)

        #Softmax of teacher prediction
        teacher_soft = F.log_softmax(teacher_pred/T, dim = 1)

        #KLDivergence of this two
        kl_div = nn.KLDivLoss(reduction = 'batchmean', log_target = True)(pred_soft, teacher_soft) * ( alpha * T * T * 2.0)

        #cross entropy loss 
        loss_y_label = F.cross_entropy(pred, labels) * (1.0 - alpha)

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
    
    def forward(self,student_pred, labels, teacher_pred, T, alpha):
        kd_loss = self.distillation_loss(student_pred, labels, teacher_pred, T, alpha)
        angular_loss = self.angular_dist(student_pred, teacher_pred)

        loss = kd_loss + angular_loss

        return loss
    
