import argparse
import yaml
import traceback
import random 
import sys
import os
import time

#local import
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
import json

def get_args():

    parser = argparse.ArgumentParser(description = 'Distillation')
    parser.add_argument('--config' , default = './config/utd/student.yaml')

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

    #model
    parser.add_argument('--model' ,default= None, help = 'Name of Model to load')

    #model args
    parser.add_argument('--device', nargs='+', default=[0], type = int)
    parser.add_argument('--model-args', default= str, help = 'A dictionary for model args')
    parser.add_argument('--weights', type = str, help = 'Location of weight file')
    parser.add_argument('--model-saved-name', type = str, help = 'Weigt name', default='test')

    #dataloader 
    parser.add_argument('--feeder', default= None , help = 'Dataloader location')
    parser.add_argument('--train-feeder-args',default=str, help = 'A dict for dataloader args' )
    parser.add_argument('--val-feeder-args', default=str , help = 'A dict for validation data loader')
    parser.add_argument('--test_feeder_args',default=str, help= 'A dict for test data loader')

    #initializaiton
    parser.add_argument('--seed', type =  int , default = 2 , help = 'random seed (default: 1)') 

    parser.add_argument('--log-interval', type = int , default = 10, metavar = 'N',
                        help = 'how many bathces to wait before logging training status')


   
    parser.add_argument('--work-dir', type = str, default = 'simple', metavar = 'F', help = "Working Directory")
    parser.add_argument('--print-log',type=str2bool,default=True,help='print logging or not')
    
    parser.add_argument('--phase', type = str, default = 'train')
    
    parser.add_argument('--num-worker', type = int, default= 0)
    parser.add_argument('--result-file', type = str, help = 'Name of resutl file')
    
    return parser

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = True
    # training speed is too slow if set to True
    torch.backends.cudnn.deterministic = False

    # on cuda 11 cudnn8, the default algorithm is very slow
    # unlike on cuda 10, the default works well
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str , _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

class Trainer():
    
    def __init__(self, arg):
        self.arg = arg
        # self.save_arg()
        self.load_model()
        self.load_optimizer()
        self.load_data()
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

        if self.arg.phase == 'test':
            self.load_weights(self.arg.weights)

    
    def load_model(self):
        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args).to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
        self.criterion= nn.CrossEntropyLoss().cuda(self.output_device)
    
    def load_weights(self, weights):
        self.model.load_state_dict(torch.load(self.arg.weights))
    
    def load_optimizer(self):
        
        if self.arg.optimizer == "Adam" :
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr = self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr = self.arg.base_lr, 
                weight_decay=self.arg.weight_decay
            )
        
        else :
           raise ValueError()
        
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
            
            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.val_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
        else:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker)
    


    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, string, print_time = True):
        print(string)
        if arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(string, file = f)

    def train(self, epoch):
        self.model.train()
        self.record_time()
        loader = self.data_loader['train']
        timer = dict(dataloader = 0.001, model = 0.001, stats = 0.001)
        loss_value = []
        acc_value = []
        accuracy = 0
        cnt = 0
        train_loss = 0
        process = tqdm(loader, ncols = 80)

        for batch_idx, (inputs, targets) in enumerate(process):
            with torch.no_grad():
                acc_data = inputs['acc_data'].cuda(self.output_device) #print("Input batch: ",inputs)
                skl_data = inputs['skl_data'].cuda(self.output_device)
                targets = targets.cuda(self.output_device)
            
            timer['dataloader'] += self.split_time()

            self.optimizer.zero_grad()

            # Ascent Step
            #print("labels: ",targets)
            out, logits,predictions = self.model(acc_data.float(), skl_data.float())
            #print("predictions: ",torch.argmax(predictions, 1) )
            loss = self.criterion(logits, targets)
            loss.mean().backward()
            self.optimizer.step()

            timer['model'] += self.split_time()
            with torch.no_grad():
                train_loss += loss.sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
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

        #Still need to work with this one
        # if save_model:
        #     state_dict = self.model.state_dict()
        #     #weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
        #     torch.save(state_dict, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')
    
    def eval(self, epoch, loader_name = 'test', result_file = None):

        if result_file is not None : 
            f_r = open (result_file, 'w')
        
        self.model.eval()

        self.print_log('Eval epoch: {}'.format(epoch+1))

        loss = 0
        cnt = 0
        accuracy = 0
        label_list = []
        pred_list = []
        
        process = tqdm(self.data_loader[loader_name], ncols=80)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(process):
                label_list.extend(targets.tolist())
                #inputs = inputs.cuda(self.output_device)
                acc_data = inputs['acc_data'].cuda(self.output_device) #print("Input batch: ",inputs)
                skl_data = inputs['skl_data'].cuda(self.output_device)
                targets = targets.cuda(self.output_device)

                #_,logits,predictions = self.model(inputs.float())
                _,logits,predictions = self.model(acc_data.float(), skl_data.float())
                batch_loss = self.criterion(logits, targets)
                loss += batch_loss.sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                pred_list.extend(torch.argmax(predictions ,1).tolist())
                cnt += len(targets)
            loss /= cnt
            accuracy *= 100./cnt
        
        if result_file is not None:
            predict = pred_list
            true = label_list

            for i, x in enumerate(predict):
                f_r.write(str(x) +  '==>' + str(true[i]) + '\n')
        
        self.print_log('\tValidation Loss: {:4f}. Validaiton Acc: {:2f}%'.format(loss, accuracy))

        if self.arg.phase == 'train':
            if accuracy > self.best_accuracy :
                self.best_accuracy = accuracy
                state_dict = self.model.state_dict()
                #weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                torch.save(state_dict, self.arg.work_dir + '/' + self.arg.model_saved_name+ '.pt')
                self.print_log('Weights Saved')        

    def start(self):
        if self.arg.phase == 'train':
            self.best_accuracy  = 0
            self.print_log('Parameters: \n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            num_params = count_parameters(self.model)
            self.print_log(f'# Parameters: {num_params}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train(epoch)
                self.eval(epoch, loader_name='val', result_file=self.arg.result_file)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'seed: {self.arg.seed}')
        
        elif self.arg.phase == 'test' :
            if self.arg.weights is None: 
                raise ValueError('Please appoint --weights')
            self.eval(epoch=0, loader_name='test', result_file=self.arg.result_file)

    # def save_arg(self):
    #     #save arg
    #     arg_dict = vars(self.arg)


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
    trainer = Trainer(arg)
    trainer.start()