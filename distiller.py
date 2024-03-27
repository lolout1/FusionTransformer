import argparse
import yaml
import traceback
import random 
import sys
import os
import time

#environmental import
import numpy as np 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import warnings
import json
import torch.nn.functional as F
# from torchsummary import summary

#local import 
from Feeder.augmentation import TSFilpper
from utils.dataprocessing import utd_processing , bmhad_processing,normalization
from main import Trainer, str2bool, init_seed, import_class



def get_args():

    parser = argparse.ArgumentParser(description = 'Distillation')
    parser.add_argument('--config' , default = './config/utd/distill.yaml')
    parser.add_argument('--dataset', type = str, default= 'utd' )
    #training
    parser.add_argument('--batch-size', type = int, default = 16, metavar = 'N',
                        help = 'input batch size for training (default: 8)')

    parser.add_argument('--test-batch-size', type = int, default = 8, 
                        metavar = 'N', help = 'input batch size for testing(default: 1000)')
    parser.add_argument('--val-batch-size', type = int, default = 8, 
                        metavar = 'N', help = 'input batch size for testing(default: 1000)')

    parser.add_argument('--num-epoch', type = int , default = 70, metavar = 'N', 
                        help = 'number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type = int, default = 0)

    #optim
    parser.add_argument('--optimizer', type = str, default = 'Adam')
    parser.add_argument('--base-lr', type = float, default = 0.001, metavar = 'LR', 
                        help = 'learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type = float , default=0.0004)

    #teacher model
    parser.add_argument('--teacher-model' ,default= None, help = 'Name of teacher model to load')
    parser.add_argument('--teacher-args', default= str, help = 'A dictionary for teacher args')
    parser.add_argument('--teacher-weight', type = str, help= 'weight for teacher')

    #student model 
    parser.add_argument('--student-model', default = None , help = 'Name of the student model to load' )
    parser.add_argument('--student-args', default= str, help= 'A dictionary for student args')
    


    #model args
    parser.add_argument('--device', nargs='+', default=[0], type = int)
    parser.add_argument('--weights', type = str, help = 'Location of weight file')
    parser.add_argument('--model-saved-name', type = str, help = 'Weigt name', default='test')

    #loss args
    parser.add_argument('--distill-loss', default='loss.BCE' , help = 'Name of loss function to use' )
    parser.add_argument('--distill-args', default ="{}", type = str,  help = 'A dictionary for loss')

    #student loss
    parser.add_argument('--student-loss', default='loss.BCE' , help = 'Name of loss function to use' )
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


   
    parser.add_argument('--work-dir', type = str, default = 'simple', metavar = 'F', help = "Working Directory")
    parser.add_argument('--print-log',type=str2bool,default=True,help='print logging or not')
    
    parser.add_argument('--phase', type = str, default = 'train')
    
    parser.add_argument('--num-worker', type = int, default= 0)
    parser.add_argument('--result-file', type = str, help = 'Name of resutl file')
    
    return parser


class Distiller(Trainer):
    def __init__(self, arg):
        self.arg = arg
        self.criterion = {}
        if self.arg.phase == "train":
            self.model = {}
            self.arg.model_args = arg.teacher_args
            # self.load_model('teacher', arg.teacher_model, arg.teacher_args)
            # self.load_weights(name='teacher', weight = self.arg.teacher_weight)
            self.model['teacher'] = torch.load(self.arg.teacher_weight)
            teacher_params = self.count_parameters(self.model['teacher'])
            print(f'# Teacher Parameters: {teacher_params}')
            self.load_model('student', arg.student_model, arg.student_args)
            self.load_loss(name = 'distill', loss = arg.distill_loss)
            self.load_loss(name = 'student', loss = arg.student_loss)



        self.include_val =arg.include_val
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        
        if self.arg.phase == 'test':
            use_cuda = torch.cuda.is_available()
            self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
            self.model = {}
            self.model['student'] = torch.load(self.arg.weights)
            self.load_loss(name='student', loss= arg.student_loss)
            self.arg.model_args = arg.student_args
            self.arg.model_args['spatial_embed'] =  self.arg.model_args['acc_embed']
            #self.load_weights(name = 'student', weight=self.arg.weights)
        
        self.load_data()
        self.load_optimizer()

        num_params = self.count_parameters(self.model['student'])
        self.print_log(f'# Student Parameters: {num_params}')
    
    def load_optimizer(self, name = 'student'):
        
        if self.arg.optimizer == "Adam" :
            self.optimizer = optim.Adam(
                self.model[name].parameters(), 
                lr = self.arg.base_lr,
                # weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model[name].parameters(), 
                lr = self.arg.base_lr, 
                weight_decay=self.arg.weight_decay
            )
        
        elif self.arg.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model[name].parameters(), 
                lr = self.arg.base_lr,
                weight_decay = self.arg.weight_decay
            )
        else :
           raise ValueError()


    def load_loss(self, name, loss):
        criterion= import_class(loss)
        #print(loss_args)
        #loss_args = yaml.safe_load(loss_args)
        self.criterion[name] = criterion()
    

    def load_model(self, name, model, model_args):
        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        Model = import_class(model)
        self.model[name] = Model(**model_args).to(f'cuda:{self.output_device}' if use_cuda else 'cpu')

    def load_weights(self, name, weight):
        self.model[name].load_state_dict(torch.load(weight))


    def train(self, epoch):
        self.model['student'].train()
        self.record_time()
        loader = self.data_loader['train']
        timer = dict(dataloader = 0.001, model = 0.001, stats = 0.001)
        loss_value = []
        acc_value = []
        accuracy = 0
        cnt = 0
        train_loss = 0
        process = tqdm(loader, ncols = 80)

        for batch_idx, (inputs, targets, idx) in enumerate(process):
            with torch.no_grad():
                acc_data = inputs['acc_data'].cuda(self.output_device) #print("Input batch: ",inputs)
                skl_data = inputs['skl_data'].cuda(self.output_device)
                targets = targets.cuda(self.output_device)
            
            timer['dataloader'] += self.split_time()

            self.optimizer.zero_grad()

            # Ascent Step
            #print("labels: ",targets)
            _, teacher_logits,predictions = self.model['teacher'](acc_data.float(), skl_data.float())
            _, student_logits, prediction = self.model['student'](acc_data.float(), skl_data.float())
            #logits = self.model(acc_data.float(), skl_data.float())
            #print("predictions: ",torch.argmax(predictions, 1) )
            # bce_loss = self.criterion(logits, targets)
            # slim_loss = 0
            # for mask in masks: 
            #     slim_loss += sum([self.slim_penalty(m) for m in mask])
            # loss = bce_loss + (0.3*slim_loss)
            loss = self.criterion['distill'](student_logits,teacher_logits, targets)
            loss.mean().backward()
            self.optimizer.step()

            timer['model'] += self.split_time()
            with torch.no_grad():
                train_loss += loss.sum().item()
                #accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                accuracy += (torch.argmax(F.log_softmax(student_logits,dim =1), 1) == targets).sum().item()
            cnt += len(targets) 
            timer['stats'] += self.split_time()
        train_loss /= cnt
        accuracy *= 100. / cnt
        loss_value.append(train_loss)
        acc_value.append(accuracy) 
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tTraining Loss: {:4f}. Training Acc: {:2f}%'.format(train_loss, accuracy)
        )
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        if not self.include_val:
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    state_dict = self.model.state_dict()
                    #weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                    torch.save(state_dict, self.arg.work_dir + '/' + self.arg.model_saved_name+ str(epoch)+ '.pt')
                    self.print_log('Weights Saved') 
        
        else: 
            self.eval(epoch, loader_name='val', result_file=self.arg.result_file)      

    def eval(self, epoch, loader_name = 'test', result_file = None):

        if result_file is not None : 
            f_r = open (result_file, 'w')
        
        self.model['student'].eval()

        self.print_log('Eval epoch: {}'.format(epoch+1))

        loss = 0
        cnt = 0
        accuracy = 0
        label_list = []
        pred_list = []
        
        process = tqdm(self.data_loader[loader_name], ncols=80)
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                label_list.extend(targets.tolist())
                #inputs = inputs.cuda(self.output_device)
                acc_data = inputs['acc_data'].cuda(self.output_device) #print("Input batch: ",inputs)
                skl_data = inputs['skl_data'].cuda(self.output_device)
                targets = targets.cuda(self.output_device)

                #_,logits,predictions = self.model(inputs.float())
                masks,logits,predictions = self.model['student'](acc_data.float(), skl_data.float())
                #logits = self.model(acc_data.float(), skl_data.float())
                # bce_loss = self.criterion(logits, targets)
                # slim_loss = 0
                # for mask in masks: 
                #     slim_loss += sum([self.slim_penalty(m) for m in mask])
                # batch_loss = bce_loss + (0.3*slim_loss)
                batch_loss = self.criterion['student'](masks, logits, targets)
                loss += batch_loss.sum().item()
                # accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                # pred_list.extend(torch.argmax(predictions ,1).tolist())
                accuracy += (torch.argmax(F.log_softmax(logits,dim =1), 1) == targets).sum().item()
                pred_list.extend(torch.argmax(F.log_softmax(logits,dim =1) ,1).tolist())
                cnt += len(targets)
            loss /= cnt
            accuracy *= 100./cnt
        
        if result_file is not None:
            predict = pred_list
            true = label_list

            for i, x in enumerate(predict):
                f_r.write(str(x) +  '==>' + str(true[i]) + '\n')
        
        self.print_log('{} Loss: {:4f}. {} Acc: {:2f}%'.format(loader_name.capitalize(),loss,loader_name.capitalize(), accuracy))
        if self.arg.phase == 'train':
            if accuracy > self.best_accuracy :
                    self.best_accuracy = accuracy
                    state_dict = self.model['student'].state_dict()
                    #weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                    torch.save(self.model['student'], self.arg.work_dir + '/' + self.arg.model_saved_name)
                    self.print_log('Weights Saved')
        else:
            return pred_list, label_list , []     


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

