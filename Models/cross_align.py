import torch 
import torch.nn as nn 
import torch.nn.functional as F

class CrossModalAligner(nn.Module):
    '''
    Calculates cross aligned features between teacher and student 
    '''
    def __init__(self, feature_dim , num_heads = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim = feature_dim ,
                                                num_heads = num_heads, 
                                                batch_first= True
                                                )
        
    def forward(self, student_features : torch.Tensor, teacher_features : torch.Tensor) -> torch.Tensor:
            '''
            student_feature: Student Features
            teacher_feature: Teacher Features

            Output: 
                aligned features
            '''
            aligned_output , attention_weights = self.cross_attn(query = teacher_features,key = student_features, value = student_features)
            return aligned_output
        