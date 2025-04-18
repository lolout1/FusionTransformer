'''
Script to train the models
'''
import traceback
from typing import List
import random 
import sys
import os
import time
import datetime
import shutil
import argparse
import yaml
from copy import deepcopy
from collections import Counter
#environmental import
import numpy as np 
import pandas as pd
import torch

#visualization packages 
import matplotlib.pyplot as plt 
import seaborn as sns 

import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

#local import 
from utils.dataset import prepare_smartfallmm, split_by_subjects
from utils.callbacks import EarlyStopping
from utils.loss import BinaryFocalLoss

def get_args():
    '''
    Function to build Argument Parser
    '''

    parser = argparse.ArgumentParser(description = 'Distillation')
    parser.add_argument('--config' , default = './config/smartfallmm/teacher.yaml')
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
    parser.add_argument('--weight-decay', type = float , default=0.001)

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
    
    #dataset args 
    parser.add_argument('--dataset-args', default=str, help = 'Arguements for dataset')

    #dataloader 
    parser.add_argument('--subjects', nargs='+', type=int)
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
    '''
    Fuction to parse boolean from text
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
def init_seed(seed):
    '''
    Initial seed for reproducabilty of the resutls
    '''
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
    '''
    Imports a class dynamically 

    '''
    mod_str , _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

THRESHOLD = 0.5

class Trainer():
    
    def __init__(self, arg):
        self.arg = arg
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_loss = float('inf')
        self.test_accuracy = 0 
        self.test_f1 = 0
        self.test_accuracy = 0
        self.test_auc = 0 
        self.test_precision = 0
        self.test_recall = 0 
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        self.optimizer = None
        self.norm_train = None
        self.norm_val = None
        self.norm_test = None
        self.data_loader = dict()
        self.early_stop = EarlyStopping(patience=15, min_delta=.001)
        #self.intertial_modality = (lambda x: next((modality for modality in x if modality != 'skeleton'), None))(arg.dataset_args['modalities'])
        self.inertial_modality = [modality  for modality in arg.dataset_args['modalities'] if modality != 'skeleton']
        self.fuse = len(self.inertial_modality) > 1 
        if os.path.exists(self.arg.work_dir):
            self.arg.work_dir = f"{self.arg.work_dir}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(self.arg.work_dir) 
        self.model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}'                    
        self.save_config(arg.config, arg.work_dir)
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else: 
            use_cuda = torch.cuda.is_available()
            self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
            self.model = torch.load(self.arg.weights)
        # self.load_loss()
        
        self.include_val = arg.include_val
        
        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size : {num_params/ (1024 ** 2):.2f} MB')
    
    def add_avg_df(self, results):
        averages = results.mean(numeric_only=True, axis = 0)
        average_row = {col: round(averages[col],2) if col in averages else 'Average' for col in results.columns}
        results = pd.concat([results, pd.DataFrame([average_row])], ignore_index=True)
        return results
    
    def save_config(self,src_path : str, desc_path : str) -> None: 
        '''
        Function to save configaration file
        ''' 
        print(f'{desc_path}/{src_path.rpartition("/")[-1]}') 
        shutil.copy(src_path, f'{desc_path}/{src_path.rpartition("/")[-1]}')
    
    def cal_weights(self):
        label_count = Counter(self.norm_train['labels'])
        self.pos_weights = torch.Tensor([label_count[0] / label_count[1]])
        self.pos_weights = self.pos_weights.to(f'cuda:{self.output_device}' 
                            if torch.cuda.is_available() else 'cpu')

    def count_parameters(self, model):
        '''
        Function to count the trainable parameters
        '''
        total_size = 0
        for param in model.parameters():
            total_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.nelement() * buffer.element_size()
        return total_size
    
    def has_empty_value(self, *lists):
        return any(len(lst) == 0 for lst in lists)
    def load_model(self, model, model_args):
        '''
        Function to load model 
        '''
        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        Model = import_class(model)
        model = Model(**model_args).to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
        return model 
    
    def load_loss(self):
        '''
        Loading loss function for the models training
        '''
        #self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
        # alpha = 1/self.pos_weights.item()
        #self.criterion = BinaryFocalLoss(alpha = 0.75)
    
    def load_weights(self):
        '''
        Load weights to the load 
        '''

        self.model.load_state_dict(torch.load(f'{self.model_path}_{self.test_subject[0]}.pth'))
    
    def load_optimizer(self, parameters) -> None:
        '''
        Loads Optimizers
        '''          
        if self.arg.optimizer.lower() == "adam" :
            self.optimizer = optim.Adam(
                parameters, 
                lr = self.arg.base_lr,
                # weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                parameters, 
                lr = self.arg.base_lr, 
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                parameters, 
                lr = self.arg.base_lr,
                #weight_decay = self.arg.weight_decay
            )
        
        else :
           raise ValueError()
    
    def distribution_viz( self,labels : np.array, work_dir : str, mode : str) -> None:
        '''
        Visualizes the training, validation/ test data set distribution
        '''
        values, count = np.unique(labels, return_counts = True)
        plt.bar(x = values,data = count, height = count)
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.savefig( work_dir + '/' + '{}_Label_Distribution'.format(mode.capitalize()))
        plt.close()
        
    def load_data(self):
        '''
        Loads different datasets
        '''
        Feeder = import_class(self.arg.feeder)
   

        if self.arg.phase == 'train':
            # if self.arg.dataset == 'smartfallmm':

            # dataset class for futher processing
            builder = prepare_smartfallmm(self.arg)

            self.norm_train = split_by_subjects(builder, self.train_subjects, self.fuse) 
            self.norm_val = split_by_subjects(builder , self.val_subject, self.fuse)          # print(norm_val['skeleton'].shape)
            if self.has_empty_value(list(self.norm_val.values())):
                return False

            #validation dataset
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args,
                               dataset = self.norm_train),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
            
            self.cal_weights()

            self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')
            
            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.val_feeder_args,
                               dataset = self.norm_val
                               ),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
            #self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, 'val')
            self.norm_test = split_by_subjects(builder , self.test_subject, self.fuse)
            if self.has_empty_value(list(self.norm_test.values())):
                    return  False
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args, dataset = self.norm_test),
                batch_size=self.arg.test_batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
            self.distribution_viz(self.norm_test['labels'], self.arg.work_dir, f'test_{self.test_subject[0]}')
            return True

    def record_time(self):
        '''
        Function to record time
        '''
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        '''
        Split time 
        '''
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, string : str, print_time = True) -> None:
        '''
        Prints log to a file
        '''
        print(string)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(string, file = f)
    def loss_viz(self, train_loss : List[float], val_loss: List[float]) :
        '''
        Visualizes the val and train loss curve togethers
        '''
        epochs = range(len(train_loss))
        plt.plot(epochs, train_loss,'b', label = "Training Loss")
        plt.plot(epochs, val_loss, 'r', label = "Validation Loss")
        plt.title(f'Train Vs Val Loss for {self.test_subject[0]}')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(self.arg.work_dir+'/'+f'trainvsval_{self.test_subject[0]}.png')
        plt.close()
    
    def cm_viz(self, y_pred : List[int], y_true : List[int]): 
        '''
        Visualizes the confusion matrix
        '''
        cm = confusion_matrix(y_true, y_pred)
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
    
    def create_df(self, columns = ['test_subject', 'accuracy', 'f1_score', 'precision', 'recall', 'auc']) -> pd.DataFrame:
        '''
        Initiats a new dataframe
        '''
        df = pd.DataFrame(columns=columns)
        return df
    
    def wrong_pred_viz(self, wrong_idx: torch.Tensor):
        '''
        Visualizes and stores the wrong predicitons

        '''

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

    def cal_prediction(self, logits):
        return (torch.sigmoid(logits)>THRESHOLD).int().squeeze(1)
        #return torch.argmax(F.log_softmax(logits,dim =1), 1)

    def cal_metrics(self, targets, predictions): 
        targets = np.array(targets)
        predictions = np.array(predictions)
        f1 = f1_score(targets, predictions)
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        auc_score = roc_auc_score(targets, predictions)
        accuracy = accuracy_score(targets, predictions)
        return accuracy*100, f1*100, recall*100, precision*100, auc_score*100

    def train(self, epoch):
        '''
        Trains the model for multiple epoch
        '''
        use_cuda = torch.cuda.is_available()
        self.model.train()
        self.record_time()
        loader = self.data_loader['train']
        timer = dict(dataloader = 0.001, model = 0.001, stats = 0.001)
        acc_value = []
        label_list = []
        pred_list = []
        accuracy = 0
        cnt = 0
        train_loss = 0

        process = tqdm(loader, ncols = 80)

        for batch_idx, (inputs, targets, idx) in enumerate(process):
            with torch.no_grad():
                acc_data = inputs['accelerometer'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                skl_data = inputs['skeleton'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                targets = targets.to(f'cuda:{self.output_device}' if use_cuda else 'cpu')

            
            timer['dataloader'] += self.split_time()

            self.optimizer.zero_grad()
            logits, _= self.model(acc_data.float(), skl_data.float())
            loss = self.criterion(logits.squeeze(1), targets.float())
            loss.backward()
            self.optimizer.step()

            timer['model'] += self.split_time()
            with torch.no_grad():
                train_loss += loss.mean().item()
                #accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                #print(torch.argmax(F.log_softmax(logits,dim =1), 1))
                preds = self.cal_prediction(logits)
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
        self.val_loss_summary.append(val_loss)
        self.early_stop(val_loss)

    
    def eval(self, epoch, loader_name = 'val', result_file = None):
        '''
        Evaluates the models performance
        '''
        use_cuda = torch.cuda.is_available()
        if result_file is not None : 
            f_r = open (result_file, 'w', encoding='utf-8')
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
                acc_data = inputs['accelerometer'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                skl_data = inputs['skeleton'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                targets = targets.to(f'cuda:{self.output_device}' if use_cuda else 'cpu')

                logits, _ = self.model(acc_data.float(), skl_data.float())
                batch_loss = self.criterion(logits.squeeze(1), targets.float())
                loss += batch_loss.sum().item()
                preds = self.cal_prediction(logits)
                label_list.extend(targets.tolist())
                pred_list.extend(preds.tolist())
                cnt += len(targets)
            loss /= cnt
            accuracy, f1, recall, precision, auc_score = self.cal_metrics(label_list, pred_list)

        if result_file is not None: 
            predict = pred_list
            true = label_list

            for i, x in enumerate(predict):
                f_r.write(str(x) +  '==>' + str(true[i]) + '\n')
        self.print_log('{} Loss: {:4f}. {} Acc: {:2f}% F1 score: {:2f}%, Precision: {:2f}%, Recall: {:2f}%, AUC: {:2f}%'.format(loader_name.capitalize(),loss,loader_name.capitalize(), accuracy, f1, precision, recall, auc_score))
        if loader_name == 'val':
            if loss < self.best_loss :
                    self.best_loss = loss
                    torch.save(deepcopy(self.model.state_dict()), f'{self.model_path}_{self.test_subject[0]}.pth')
                    self.print_log('Weights Saved')
        else: 
            self.test_accuracy = accuracy
            self.test_f1 = f1
            self.test_recall = recall
            self.test_precision = precision
            self.test_recall = recall
            self.test_auc = auc_score
        return loss       

    def start(self):

        '''
        Function to start the the training 

        '''

        if self.arg.phase == 'train':

                self.best_accuracy  = float('-inf')

                self.best_f1 = float('inf')
                self.print_log('Parameters: \n{}\n'.format(str(vars(self.arg))))
            
                results = self.create_df()
                for i in range(len(self.arg.subjects[:-3])): 
                    self.train_loss_summary = []
                    self.val_loss_summary = []
                    self.best_loss = float('inf')
                    test_subject = self.arg.subjects[i]
                    train_subjects = list(filter(lambda x : x not in [test_subject], self.arg.subjects))
                    self.val_subject = [38,46]
                    self.test_subject = [test_subject]
                    self.train_subjects = train_subjects
                    self.model = self.load_model(self.arg.model, self.arg.model_args)
                    self.print_log(f'Model Parameters: {self.count_parameters(self.model)}')
                    if not self.load_data():
                        continue

                    self.load_optimizer(self.model.parameters())
                    self.load_loss()
                    self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
                    for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                        self.train(epoch)
                        if self.early_stop.early_stop:
                            self.early_stop = EarlyStopping(min_delta=.001)
                            break
                    self.load_model(self.arg.model,self.arg.model_args)
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
    
    def viz_feature(self, teacher_features, student_features, epoch):
        teacher_features = torch.flatten(teacher_features, start_dim=1)
        student_features = torch.flatten(student_features, start_dim= 1)
        plt.figure(figsize=(8,4))
        for i in range(8):
            plt.subplot(2,4,i+1)
            sns.kdeplot(teacher_features[i, :].cpu().detach().numpy(), bw_adjust=0.5, color = 'blue', label = 'Teacher KDE')
            
            plt.subplot(2,4,i+1)
            sns.kdeplot(student_features[i, :].cpu().detach().numpy(), bw_adjust=0.5, color = 'red', label = 'Student KDE')
            plt.legend()
            plt.savefig(f'{self.arg.work_dir}/Feature_KDE_{epoch}.png')
        plt.close()
                    


if __name__ == "__main__":
    parser = get_args()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r', encoding= 'utf-8') as f:
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
