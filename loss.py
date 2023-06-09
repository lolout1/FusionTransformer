import torch 
import torch.nn.functional as F
import torch.nn as nn



class SemanticLoss(nn.Module):
    def __init__(self, T = 2, alpha = 0.7):
        super(SemanticLoss, self).__init__()
        self.T = T
        self.alpha = alpha
    
    def distillation_loss(self,pred, logits, labels, teacher_logits):
        
        #Softmax of student prediciton 
        pred_soft = F.log_softmax(logits/self.T, dim = 1)

        #Softmax of teacher prediction
        teacher_soft = F.log_softmax(teacher_logits/self.T, dim = 1)

        #KLDivergence of this two
        kl_div = nn.KLDivLoss(reduction = 'batchmean')(pred_soft, teacher_soft) * (self.T * self.T)

        #cross entropy loss 
        loss_y_label = F.cross_entropy(pred, labels) 

        distill_loss = (self.alpha * kl_div) + (loss_y_label * (1.0 - self.alpha))

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

    def forward(self,student_pred, stud_logits, labels, teacher_logits):
        gamma = 0.1
        beta = 0.2
        sigma = 1 - gamma - beta
        kd_loss = self.distillation_loss(pred = student_pred, logits = stud_logits, labels = labels, teacher_logits = teacher_logits)
        y = F.log_softmax(stud_logits, dim = 1)
        teacher_y = F.log_softmax(teacher_logits, dim = 1)
        angular_loss = self.angular_dist(y, teacher_y)
        dist_loss = self.distance(y, teacher_y)

        loss = (sigma*kd_loss) + (beta*angular_loss) + (gamma*dist_loss)
        return kd_loss
        # return kd_loss
       
    
