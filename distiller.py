import argparse
import yaml
import traceback
import random 
import sys
import os
import time
from collections import OrderedDict

#environmental import

## visualization packages ##
import seaborn as sns 
import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import warnings
import json
import torch.nn.functional as F
from main import Trainer, str2bool, init_seed, import_class
from utils.loss import DistillationLoss
from utils.callbacks import EarlyStopping
from Models.cross_align import CrossModalAligner
from sklearn.metrics import f1_score, precision_score, recall_score, auc

TEMPERATURE = 3
ALPHA = 0.5


def get_args():

    parser = argparse.ArgumentParser(description = 'Distillation')
    parser.add_argument('--config' , default = './config/smartfallmm/distill.yaml')
    parser.add_argument('--dataset', type = str, default= 'utd' )
    #training
    parser.add_argument('--batch-size', type = int, default = 16, metavar = 'N',
                        help = 'input batch size for training (default: 8)')

    parser.add_argument('--test-batch-size', type = int, default = 8, 
                        metavar = 'N', help = 'input batch size')
    parser.add_argument('--val-batch-size', type = int, default = 8, 
                        metavar = 'N', help = 'input batch size')

    parser.add_argument('--num-epoch', type = int , default = 70, metavar = 'N', 
                        help = 'number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type = int, default = 0)

    #optim
    parser.add_argument('--optimizer', type = str, default = 'Adam')
    parser.add_argument('--base-lr', type = float, default = 0.001, metavar = 'LR', 
                        help = 'learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type = float , default=0.001)

    #data 
    # parser.add_argument('--train-subjects', nargs='+', type = int)
    parser.add_argument('--subjects', nargs='+', type = int)
    parser.add_argument('--dataset-args', default= None, type= str)

    #teacher model
    parser.add_argument('--teacher-model' ,default= None, help = 'Name of teacher model to load')
    parser.add_argument('--teacher-args', default= str, help = 'A dictionary for teacher args')
    parser.add_argument('--teacher-weight', type = str, default="/home/bgu9/LightHART/exps/smartfall_fall_wokd/teacher/skeleton_with_experimental/spTransformer", help= 'weight for teacher')

    #student model 
    parser.add_argument('--model', default = None , help = 'Name of the student model to load' )
    parser.add_argument('--model-args', default= str, help= 'A dictionary for student args')
    


    #model args
    parser.add_argument('--device', nargs='+', default=[2], type = int)
    parser.add_argument('--weights', type = str, help = 'Location of weight file')
    parser.add_argument('--model-saved-name', type = str,  default = "ttfstudent", help = 'Weigt name')

    #loss args
    parser.add_argument('--distill-loss', default='loss.BCE' , help = 'Name of loss function to use' )
    parser.add_argument('--distill-args', default ="{}", type = str,  help = 'A dictionary for loss')

    #student loss
    parser.add_argument('--loss', default='loss.BCE' , help = 'Name of loss function to use' )
    parser.add_argument('--loss-args', default ="{}", type = str,  help = 'A dictionary for loss')


    # parser.add_argument('--loss-args', default=str, help = 'A dictonary for loss args' )
    #dataloader 
    parser.add_argument('--feeder', default= None , help = 'Dataloader location')
    parser.add_argument('--train-feeder-args',default=str, help = 'A dict for dataloader args' )
    parser.add_argument('--val-feeder-args', default=str , help = 'A dict for validation data loader')
    parser.add_argument('--test_feeder_args',default=str, help= 'A dict for test data loader')
    parser.add_argument('--include-val', type = str2bool, default= True , help = 'If we will have the validation set or not')

    #initializaiton
    parser.add_argument('--seed', type =  int , default = 2 , help = 'random seed (default: 1)') 

    parser.add_argument('--log-interval', type = int , default = 10, metavar = 'N',
                        help = 'how many batches to wait before logging training status')


    parser.add_argument('--work-dir', type = str, default = 'exps/test', metavar = 'F', help = "Working Directory")
    parser.add_argument('--print-log',type=str2bool,default=True,help='print logging or not')
    
    parser.add_argument('--phase', type = str, default = 'train')
    
    parser.add_argument('--num-worker', type = int, default= 0)
    parser.add_argument('--result-file', type = str, help = 'Name of resutl file')
    
    return parser


class Distiller(Trainer):
    def __init__(self, arg):
        super().__init__(arg)
        self.teacher_model = self.load_model(arg.teacher_model, arg.teacher_args)
        self.cross_aligner  = self.load_aligner()
        self.early_stop = EarlyStopping(patience=15, min_delta=.001)

    
    def load_aligner(self):
        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        return CrossModalAligner(feature_dim=arg.model_args['embed_dim']).to(f'cuda:{self.output_device}' if use_cuda else 'cpu')


    def load_teacher_weights(self):
        '''
        Load weights to the load 
        '''
        model_weights = torch.load(f"{self.arg.teacher_weight}_{self.test_subject[0]}.pth", weights_only=False)
        if isinstance(model_weights, OrderedDict):
            self.teacher_model.load_state_dict(torch.load(f"{self.arg.teacher_weight}_{self.test_subject[0]}.pth"))
        else:
            self.teacher_model = model_weights

    def load_loss(self):
        self.mse = torch.nn.MSELoss()
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
        self.distillation_loss = DistillationLoss(pos_weigths=self.pos_weights,)
    


    def train(self, epoch):
        self.model.train()
        self.teacher_model.eval()
        self.record_time()
        loader = self.data_loader['train']
        timer = dict(dataloader = 0.001, model = 0.001, stats = 0.001)
        loss_value = []
        acc_value = []
        label_list = []
        pred_list = []
        accuracy = 0
        teacher_accuracy = 0
        cnt = 0
        train_loss = 0
        process = tqdm(loader, ncols = 80)
        use_cuda = torch.cuda.is_available()
        loss = 0
        
        for batch_idx, (inputs, targets, idx) in enumerate(process):

            with torch.no_grad():
                acc_data = inputs['accelerometer'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                skl_data = inputs['skeleton'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                targets = targets.to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
            timer['dataloader'] += self.split_time()

            self.optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits, teacher_features = self.teacher_model(acc_data.float(), skl_data.float())
            student_logits, student_features = self.model(acc_data.float(), skl_data.float(), epoch = epoch)
            #cross_aligned_features= self.cross_aligner(student_features = student_features, teacher_features = teacher_features)
            target = torch.ones(teacher_features.shape[0]).to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
            if epoch % 10 == 0 and batch_idx == 0 : 
                self.viz_feature(teacher_features=teacher_features, student_features=student_features, epoch = epoch)
            loss = self.distillation_loss(student_logits=student_logits,
                                          teacher_logits=teacher_logits, labels=targets, 
                                          teacher_features = teacher_features, 
                                          student_features = student_features, 
                                          target = target.float())
            loss.backward()
            self.optimizer.step()

            timer['model'] += self.split_time()
            with torch.no_grad():
                train_loss += loss.sum().item()
                preds = self.cal_prediction(student_logits)
                label_list.extend(targets.tolist())
                pred_list.extend(preds.tolist())
                
            cnt += len(targets) 
            timer['stats'] += self.split_time()
        train_loss /= cnt
        accuracy, f1, recall, precision, auc_score = self.cal_metrics(label_list, pred_list)

        self.train_loss_summary.append(train_loss)
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tTraining Loss: {:4f},  Acc: {:2f}%, F1 score: {:2f}%, Precision: {:2f}%, Recall: {:2f}%, AUC: {:2f}%  '.format(train_loss, accuracy, f1, precision, recall, auc_score)
        )
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        val_loss = self.eval(epoch, loader_name='val', result_file=self.arg.result_file)
        self.early_stop(val_loss=val_loss)
        self.val_loss_summary.append(val_loss)


        
    def start(self):
        if self.arg.phase == 'train':
                                            
                results = self.create_df()
                for i in range(len(self.arg.subjects[:-3])): 
                    self.train_loss_summary = []
                    self.val_loss_summary = []
                    self.best_loss = float('inf')
                    self.best_f1 = float('-inf')
                    self.best_recall = float('-inf')
                    self.best_precision = float('-inf')
                    self.best_accuracy = float('-inf')
                    test_subject = self.arg.subjects[i]
                    train_subjects = list(filter(lambda x : x not in [test_subject], self.arg.subjects))
                    self.val_subject = [35,46]
                    self.test_subject = [test_subject]
                    self.train_subjects = train_subjects
 
                    self.load_teacher_weights()
                    self.model = self.load_model(self.arg.model, self.arg.model_args)
                    if not self.load_data():
                        continue
                    self.load_loss()
                    self.load_optimizer(self.model.parameters())

                    self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
                    for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                        self.train(epoch)
                        if self.early_stop.early_stop:
                            self.early_stop = EarlyStopping(patience=15, min_delta=1e-6)
                            break
                    self.model = self.load_model(self.arg.model,self.arg.model_args)
                    self.cross_aligner = self.load_aligner()
                    self.load_weights()
                    self.model.eval()
                    self.print_log(f' ------------ Test Subject {self.test_subject[0]} -------')
                    self.eval(epoch = 0 , loader_name='test')
                    self.print_log(f'Test accuracy for : {self.test_accuracy}')
                    self.print_log(f'Test F-Score: {self.test_f1}')
                    self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                    subject_result = pd.DataFrame([{'test_subject' : str(self.test_subject[0]), 
                                                'accuracy':round(self.test_accuracy,2), 'f1_score':round(self.test_f1, 2), 'precision':round(self.test_precision, 2),
                                                'recall' : round(self.test_recall,2), 'auc': round(self.test_auc, 2) }])
                    results = pd.concat([results, subject_result], ignore_index=True)
                results = self.add_avg_df(results)
                results.to_csv(f'{self.arg.work_dir}/scores.csv')

        
        elif self.arg.phase == 'test' :
            if self.arg.weights is None: 
                raise ValueError('Please add --weights')
            y_pred, y_true, wrong_idx = self.eval(epoch=0, loader_name='test', result_file=self.arg.result_file)
            self.cm_viz(y_pred, y_true)   



if __name__ == "__main__":
    parser = get_args()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    trainer = Distiller(arg)
    trainer.start()

