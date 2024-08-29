import argparse
import yaml
import traceback
import random 
import sys
import os
import time

#environmental import
import numpy as np 
import pandas as pd
from einops import rearrange
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import warnings
import json
import torch.nn.functional as F
#from torchsummary import summary
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, accuracy_score

#local import 
from Feeder.augmentation import TSFilpper
from utils.dataprocessing import utd_processing , bmhad_processing, czu_processing, sf_processing,normalization
from utils.dataset_loader import load_inertial_data, load_skeleton_data

# #args = [2, 4, 5, 7,8, 10, 11,14, 15, 16, 17,19, 21, 23, 26]
# args = [i for i in range(30, 37)]
# TEST_SUBJECT = [37]

def get_args():

    parser = argparse.ArgumentParser(description = 'Distillation')
    parser.add_argument('--config' , default = './config/utd/student.yaml')
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

    #model
    parser.add_argument('--model' ,default= None, help = 'Name of Model to load')

    #model args
    parser.add_argument('--device', nargs='+', default=[0], type = int)

    parser.add_argument('--model-args', default= str, help = 'A dictionary for model args')
    parser.add_argument('--weights', type = str, help = 'Location of weight file')
    parser.add_argument('--model-saved-name', type = str, help = 'Weigt name', default='test')

    #loss args
    parser.add_argument('--loss', default='loss.BCE' , help = 'Name of loss function to use' )
    parser.add_argument('--loss-args', default ="{}", type = str,  help = 'A dictionary for loss')
    # parser.add_argument('--loss-args', default=str, help = 'A dictonary for loss args' )

    #dataloader 
    parser.add_argument('--subjects', nargs='+', type=int)
    # parser.add_argument('--test-subjects', nargs='+', type = int)
    parser.add_argument('--feeder', default= None , help = 'Dataloader location')
    parser.add_argument('--train-feeder-args',default=str, help = 'A dict for dataloader args' )
    parser.add_argument('--val-feeder-args', default=str , help = 'A dict for validation data loader')
    parser.add_argument('--test_feeder_args',default=str, help= 'A dict for test data loader')
    parser.add_argument('--include-val', type = str2bool, default= True , help = 'If we will have the validation set or not')

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
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else: 
            use_cuda = torch.cuda.is_available()
            self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
            self.model = torch.load(self.arg.weights)
        self.load_loss()
        
        # self.load_data()
        self.include_val = arg.include_val

        # if self.arg.phase == 'test':
        #     # self.load_weights(self.arg.weights)
        #     self.model.load_state_dict(torch.load(self.arg.weights))
        
        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size : {num_params/ (1024 ** 2):.2f} MB')

    def count_parameters(self, model):
            # return sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_size = 0
        for param in model.parameters():
            total_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.nelement() * buffer.element_size()
        return total_size
            

    
    def load_model(self, model, model_args):
        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        Model = import_class(model)
        model = Model(**model_args).to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
        return model 
    
    def load_loss(self):
        #criterion= import_class(self.arg.loss)
        self.criterion = torch.nn.CrossEntropyLoss()
        #loss_args = yaml.safe_load(arg.loss_args)
        #self.criterion = criterion()
    
    def load_weights(self, weights):
        self.model.load_state_dict(torch.load(self.arg.weights))
    
    def load_optimizer(self):
        
        if self.arg.optimizer == "Adam" :
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr = self.arg.base_lr,
                # weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr = self.arg.base_lr, 
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr = self.arg.base_lr,
                weight_decay = self.arg.weight_decay
            )
        
        else :
           raise ValueError()
    
    def distribution_viz( self,labels, work_dir, mode):
        values, count = np.unique(labels, return_counts = True)
        plt.bar(x = values,data = count, height = count)
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.savefig( work_dir + '/' + '{}_Label_Distribution'.format(mode.capitalize()))
        plt.close()
        
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        ## need to change it to dynamic import 
        self.data_loader = dict()
        if self.arg.phase == 'train':
            if self.arg.dataset == 'utd':
                train_data =  utd_processing(mode = self.arg.phase, 
                                             acc_window_size = self.arg.model_args['acc_frames'], 
                                             #skl_window_size = self.arg.model_args['spatial_embed'], 
                                             num_windows = 15)

                #self.distribution_viz(train_data['labels'], self.arg.work_dir, 'train')
                norm_train, acc_scaler, skl_scaler =  normalization(data = train_data, mode = 'fit')
                val_data =  utd_processing(mode = 'val', 
                                             acc_window_size = self.arg.model_args['acc_frames'], 
                                             #skl_window_size = self.arg.model_args['spatial_embed'], 
                                             num_windows = 15)
                norm_val, _, _ =  normalization(data = val_data,#acc_scaler=acc_scaler,
                                               #skl_scaler=skl_scaler,
                                                 mode = 'fit')
                self.distribution_viz(val_data['labels'], self.arg.work_dir, 'val')

                test_data =  utd_processing(mode = 'test', 
                                            acc_window_size = self.arg.model_args['acc_frames'], 
                                             #skl_window_size = self.arg.model_args['spatial_embed'], 
                                             num_windows = 15)
                norm_test, _, _ =  normalization(data = test_data, mode = 'fit' )
                self.distribution_viz(test_data['labels'], self.arg.work_dir, 'test')
                self.data_loader['test'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.test_feeder_args, dataset = norm_test),
                    batch_size=self.arg.test_batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker)
            #elif self.arg.dataset == 'dmft':
                
            elif self.arg.dataset == 'bmhad':
                #train_data = bmhad_processing(mode = arg.phase)
                train_data = np.load('data/berkley_mhad/bhmad_uniformdis_skl50_train.npz')
                norm_train, acc_scaler, skl_scaler = normalization(data= train_data, mode = 'fit' )
                self.distribution_viz(train_data['labels'], self.arg.work_dir, 'train')
                #val_data  = bmhad_processing(mode = 'val')
                val_data = np.load('data/berkley_mhad/bhmad_uniformdis_skl50_val.npz')
                norm_val, _, _ = normalization(data = val_data , mode = 'transform')
                self.distribution_viz(val_data['labels'], self.arg.work_dir, 'val')

                test_data = np.load('data/berkley_mhad/bhmad_uniformdis_skl50_test.npz')
                norm_test, _, _ =  normalization(data = test_data, mode = 'fit' )

                self.data_loader['test'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.test_feeder_args, dataset = norm_test),
                    batch_size=self.arg.test_batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker)
            
            elif self.arg.dataset == 'czu':
                train_data = czu_processing(mode='train', acc_window_size = self.arg.model_args['acc_frames'],
                                            skl_window_size=self.arg.model_args['spatial_embed'],
                                            num_windows=15  )
                
                #norm_train, acc_scaler, skl_scaler =  normalization(data=train_data, mode = 'fit')

                val_data = czu_processing(mode='test', acc_window_size=self.arg.model_args['acc_frames'], 
                                          skl_window_size=self.arg.model_args['spatial_embed'], 
                                          num_windows=15)
                #norm_val, acc_scaler, skl_scaler =  normalization(data=val_data, mode = 'fit')
                
                #norm_test = norm_val
                norm_test = val_data

                self.data_loader['test'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.test_feeder_args, dataset = norm_test),
                    batch_size=self.arg.test_batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker)

            elif self.arg.dataset == 'smartfallmm':

                train_data = sf_processing(subjects= self.train_subjects,
                                            acc_window_size= self.arg.model_args['acc_frames'],
                                            skl_window_size=self.arg.model_args['mocap_frames'], 
                                            #mode = 'train',
                                            num_windows = 20)
                
                norm_train, acc_scaler, skl_scaler =  normalization(data=train_data, mode = 'fit')

                val_data = sf_processing(subjects = self.test_subject, 
                                          acc_window_size=self.arg.model_args['acc_frames'],
                                          skl_window_size=self.arg.model_args['mocap_frames'], 
                                          #mode = 'test',
                                          num_windows=20)
                norm_val, acc_scaler, skl_scaler =  normalization(data=val_data, mode = 'fit')
                
                norm_test = norm_val

                self.data_loader['test'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.test_feeder_args, dataset = norm_test),
                    batch_size=self.arg.test_batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker)

            else: 
                norm_train = torch.load('data/UTD_MAAD/utd_train.pt')
                norm_val = torch.load('data/UTD_MAAD/utd_test.pt')


            # self.acc_scaler = acc_scaler
            # self.skl_scaler = skl_scaler
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args,
                               dataset = norm_train, 
                               #dataset = train_data,
                               transform =None),
                #dataset = norm_train,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
            
            
            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.val_feeder_args,
                               dataset = norm_val
                               #dataset = val_data
                               ),
                #dataset = norm_val,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
        else:
            if self.arg.dataset == 'utd':
                self.test_data =  utd_processing(mode = 'test', 
                                            acc_window_size = self.arg.model_args['acc_frames'], 
                                             #skl_window_size = self.arg.model_args['spatial_embed'], 
                                             num_windows = 15)
                #self.distribution_viz(test_data['labels'], self.arg.work_dir, 'test')
            elif self.arg.dataset == 'czu':
                self.test_data =  utd_processing(mode = 'test', 
                                acc_window_size = self.arg.model_args['acc_frames'], 
                                    skl_window_size = self.arg.model_args['spatial_embed'], 
                                    num_windows = 15)
            elif self.arg.dataset == 'smartfallmm':
                self.test_data = sf_processing(subjects = self.arg.test_subjects, 
                                          acc_window_size=self.arg.model_args['acc_frames'],
                                          skl_window_size=self.arg.model_args['mocap_frames'], 
                                          #mode = 'test',
                                          num_windows=20)
            else:
                self.test_data = np.load('data/berkley_mhad/bhmad_uniformdis_skl50_test.npz')
            #self.distribution_viz(self.test_data['labels'], self.arg.work_dir, 'test')
            norm_test, _, _ =  normalization(data = self.test_data, mode = 'fit')
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args, dataset = norm_test),
                batch_size=self.arg.test_batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
            #self.distribution_viz(test_data['labels'], self.arg.work_dir, 'test')


    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, string, print_time = True):
        print(string)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(string, file = f)
    def loss_viz(self, train_loss : List[float], val_loss: List[float]) :
        epochs = range(len(train_loss))
        plt.plot(epochs, train_loss,'b', label = "Training Loss")
        plt.plot(epochs, val_loss, 'r', label = "Validation Loss")
        plt.title('Train Vs Val Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(self.arg.work_dir+'/'+'trainvsval.png')
        plt.close()
    
    def cm_viz(self, y_pred : List[int], y_true : List[int]): 
        cm = confusion_matrix(y_true, y_pred)
        # plot the confusion matrix
        plt.figure(figsize=(10,6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.xticks(np.unique(y_true))
        plt.yticks(np.unique(y_true))
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix")
        plt.savefig(self.arg.work_dir + '/' + 'Confusion Matrix')
        plt.close()
    
    def create_df(self, columns = ['test_subject', 'train_subjects', 'accuracy', 'f1_score']):
        df = pd.DataFrame(columns=columns)
        return df
    def wrong_pred_viz(self, wrong_idx: torch.Tensor):
        wrong_label = self.test_data['labels'][wrong_idx]
        wrong_label = wrong_label
        labels, mis_count = np.unique(wrong_label, return_counts =True)
        wrong_idx = np.array(wrong_idx)
        plt.figure(figsize=(10,10))
        for i in labels:
            act_idx = wrong_idx[np.where(wrong_label == i)[0]]
            count = 0
            for j in range(5):
                count += 1
                # plt.subplot(i+1,j+1, count)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                trial_data = self.test_data['acc_data'][act_idx[j]]
                plt.plot(trial_data)
                plt.xlabel(i)     
                plt.savefig(self.arg.work_dir + '/' +'wrong_predictions'+'/'+ 'Wrong_Pred' +str(i)+str(j))
                plt.close()



    def train(self, epoch):
        self.model.train()
        self.record_time()
        loader = self.data_loader['train']
        timer = dict(dataloader = 0.001, model = 0.001, stats = 0.001)
        acc_value = []
        accuracy = 0
        cnt = 0
        train_loss = 0

        process = tqdm(loader, ncols = 80)

        for batch_idx, (inputs, targets, idx) in enumerate(process):
        # for batch_idx, [acc_data, skl_data, targets] in enumerate(process):
 
            with torch.no_grad():
                acc_data = inputs['acc_data'].cuda(self.output_device) #print("Input batch: ",inputs)
                skl_data = inputs['skl_data'].cuda(self.output_device)
                targets = targets.cuda(self.output_device)

            
            timer['dataloader'] += self.split_time()

            self.optimizer.zero_grad()
            #masks, logits,predictions = self.model(acc_data.float(), skl_data.float())
            logits= self.model(acc_data.float(), skl_data.float())

            #logits = self.model(acc_data.float(), skl_data.float())
            #print("predictions: ",torch.argmax(predictions, 1) )
            # bce_loss = self.criterion(logits, targets)
            # slim_loss = 0
            # for mask in masks: 
            #     slim_loss += sum([self.slim_penalty(m) for m in mask])
            # loss = bce_loss + (0.3*slim_loss)
            #print(torch.argmax(F.log_softmax(logits,dim =1), 1))
            #print(targets)
            #loss = self.criterion(masks, logits, targets)
            loss = self.criterion(logits, targets)
            loss.mean().backward()
            self.optimizer.step()

            timer['model'] += self.split_time()
            with torch.no_grad():
                train_loss += loss.mean().item()
                #accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                accuracy += (torch.argmax(F.log_softmax(logits,dim =1), 1) == targets).sum().item()
                
            cnt += len(targets) 
            timer['stats'] += self.split_time()
        
        train_loss /= cnt
        accuracy *= 100. / cnt

        self.train_loss_summary.append(train_loss)
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
                    torch.save(state_dict, self.arg.work_dir + '/' + self.arg.model_saved_name+ '.pt')
                    self.print_log('Weights Saved') 
        
        else: 
            val_loss = self.eval(epoch, loader_name='val', result_file=self.arg.result_file)
            self.val_loss_summary.append(val_loss)
        
        #test 


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
        wrong_idx = []
        
        process = tqdm(self.data_loader[loader_name], ncols=80)
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
            # for batch_idx, [acc_data, skl_data, targets] in enumerate(process):
                acc_data = inputs['acc_data'].cuda(self.output_device) #print("Input batch: ",inputs)
                skl_data = inputs['skl_data'].cuda(self.output_device)
                targets = targets.cuda(self.output_device)

                # masks,logits,predictions = self.model(acc_data.float(), skl_data.float())
                logits= self.model(acc_data.float(), skl_data.float())
                #batch_loss = self.criterion(masks, logits, targets)
                batch_loss = self.criterion(logits, targets)
                #batch_loss = self.criterion(logits, targets)
                loss += batch_loss.sum().item()
                # accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                # pred_list.extend(torch.argmax(predictions ,1).tolist())
                accuracy += (torch.argmax(F.log_softmax(logits,dim =1), 1) == targets).sum().item()
                # wrong_pred = idx[torch.argmax(F.log_softmax(logits,dim =1), 1) == targets]
                # wrong_idx.extend(wrong_pred.tolist())
                label_list.extend(targets.tolist())
                pred_list.extend(torch.argmax(F.log_softmax(logits,dim =1) ,1).tolist())
                # print(len(pred_list))
                cnt += len(targets)
            loss /= cnt
            target = np.array(label_list)
            y_pred = np.array(pred_list)
            f1 = f1_score(target, y_pred, average='macro') * 100
            accuracy *= 100./cnt
        # accuracy = accuracy_score(label_list, pred_list) * 100
        if result_file is not None: 
            predict = pred_list
            true = label_list

            for i, x in enumerate(predict):
                f_r.write(str(x) +  '==>' + str(true[i]) + '\n')
        
        self.print_log('{} Loss: {:4f}. {} Acc: {:2f}% f1: {:2f}'.format(loader_name.capitalize(),loss,loader_name.capitalize(), accuracy, f1))
        if self.arg.phase == 'train':
            if accuracy > self.best_accuracy :
                    self.best_accuracy = accuracy
                    self.best_f1 = f1
                    #state_dict = self.model.state_dict()
                    #weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                    # torch.save(self.model, f'{self.arg.work_dir}/{self.arg.model_saved_name}_{self.test_subject[0]}.pth')
                    torch.save(self.model, f'{self.arg.work_dir}/{self.arg.model_saved_name}.pth')
                    self.print_log('Weights Saved')
        else: 
            return pred_list, label_list, wrong_idx
        return loss       

    def start(self):
        #summary(self.model,[(model_args['acc_frames'],3), (model_args['mocap_frames'], model_args['num_joints'],3)] , dtypes=[torch.float, torch.float] )
        if self.arg.phase == 'train':
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_accuracy  = float('-inf')

                self.best_f1 = float('inf')
                self.print_log('Parameters: \n{}\n'.format(str(vars(self.arg))))
            
            # results = self.create_df()
            # for test_subject in self.arg.subjects : 
                # train_subjects = list(filter(lambda x : x != test_subject, self.arg.subjects))
                # self.test_subject = [test_subject]
                # self.train_subjects = train_subjects
                self.model = self.load_model(self.arg.model, self.arg.model_args)
                self.print_log(f'Model Parameters: {self.count_parameters(self.model)}')
                self.load_data()
                self.load_optimizer()

                self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    self.train(epoch)
                # self.print_log(f'Train Subjects : {self.train_subjects}')
                # self.print_log(f' ------------ Test Subject {self.test_subject[0]} -------')
                self.print_log(f'Best accuracy for : {self.best_accuracy}')
                self.print_log(f'Best F-Score: {self.best_f1}')
                #self.print_log(f'Epoch number: {self.best_acc_epoch}')
                self.print_log(f'Model name: {self.arg.work_dir}')
                #self.print_log(f'Model total number of params: {num_params}')
                self.print_log(f'Weight decay: {self.arg.weight_decay}')
                self.print_log(f'Base LR: {self.arg.base_lr}')
                self.print_log(f'Batch Size: {self.arg.batch_size}')
                self.print_log(f'seed: {self.arg.seed}')
                self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                # subject_result = pd.Series({'test_subject' : str(self.test_subject), 'train_subjects' :str(self.train_subjects), 
                #                                'accuracy':round(self.best_accuracy,2), 'f1_score':round(self.best_f1, 2)})
                # results.loc[len(results)] = subject_result
                self.best_accuracy = 0
                self.best_f1 = 0
            # results.to_csv(f'{self.arg.work_dir}/scores.csv')
                
            # self.model = self.load_model(self.arg.model, self.arg.model_args)
            # self.load_loss()
            # self.load_optimizer()
            # self.model.load_state_dict(torch.load(self.arg.weights))
            # val_loss = self.eval(0, loader_name='test', result_file=self.arg.result_file)

        
        elif self.arg.phase == 'test' :
            if self.arg.weights is None: 
                raise ValueError('Please add --weights')
            y_pred, y_true, wrong_idx = self.eval(epoch=0, loader_name='test', result_file=self.arg.result_file)
            self.cm_viz(y_pred, y_true)
            #self.wrong_pred_viz(wrong_idx=wrong_idx)

            

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
