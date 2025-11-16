'''
Script to train the models
'''
import traceback
from typing import List, Dict
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
    parser.add_argument('--validation-subjects', nargs='+', type=int, default=[38,46],
                        help='Subjects reserved for validation (excluded from training)')
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
        self.fold_metrics = []
        self.epoch_logs = []
        self.best_val_metrics = None
        self.current_fold_metrics = {'train': None, 'val': None, 'test': None}
        self.current_fold_index = None
        self.current_test_subject = None
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
        self.use_skeleton = arg.dataset_args.get('use_skeleton', True)
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
        if results.empty:
            return results
        averages = results.mean(numeric_only=True, axis=0)
        average_row = {}
        for col in results.columns:
            if col == 'test_subject':
                average_row[col] = 'Average'
            elif col in averages:
                precision = 6 if 'loss' in col else 2
                average_row[col] = round(averages[col], precision)
            else:
                average_row[col] = None
        results = pd.concat([results, pd.DataFrame([average_row])], ignore_index=True)
        return results

    def _format_metrics(self, loss, accuracy, f1, precision, recall, auc_score):
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'auc': float(auc_score)
        }

    def _round_metrics(self, metrics):
        if not metrics:
            return {}
        return {
            key: round(value, 6 if key == 'loss' else 2)
            for key, value in metrics.items()
        }

    def _record_epoch_metrics(self, phase, epoch, metrics):
        if not metrics:
            return
        entry = {
            'fold': (self.current_fold_index + 1) if self.current_fold_index is not None else None,
            'test_subject': str(self.current_test_subject) if self.current_test_subject is not None else None,
            'phase': phase,
            'epoch': epoch
        }
        entry.update(metrics)
        self.epoch_logs.append(entry)

    def _init_fold_tracking(self, fold_index, test_subject):
        self.current_fold_index = fold_index
        self.current_test_subject = test_subject
        self.current_fold_metrics = {'train': None, 'val': None, 'test': None}
        self.best_val_metrics = None
    
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

    def _get_inertial_key(self, inputs: Dict[str, torch.Tensor]) -> str:
        """
        Determine which inertial modality key is present in the current batch.
        Falls back to 'accelerometer' when combined IMU data is stored there.
        """
        for modality in self.inertial_modality:
            if modality in inputs:
                return modality
        if 'accelerometer' in inputs:
            return 'accelerometer'
        available = ', '.join(inputs.keys())
        raise KeyError(f'No inertial modality found in batch inputs: {available}')
    
    def has_empty_value(self, lists):
        """Check if any array/list in the collection is empty"""
        return any(len(lst) == 0 for lst in lists)
    def load_model(self, model, model_args):
        '''
        Function to load model
        '''
        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device

        # Determine device
        if use_cuda:
            device = f'cuda:{self.output_device}'
        else:
            device = 'cpu'

        Model = import_class(model)
        model = Model(**model_args).to(device)
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
            self.norm_val = split_by_subjects(builder , self.val_subject, self.fuse)

            if self.has_empty_value(list(self.norm_val.values())):
                self.print_log(f'Warning: Validation data is empty for subjects {self.val_subject}. Skipping iteration.')
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
    
    def create_df(self, columns=None) -> pd.DataFrame:
        '''
        Initiats a new dataframe
        '''
        if columns is None:
            columns = [
                'test_subject',
                'train_loss', 'train_accuracy', 'train_f1_score', 'train_precision', 'train_recall', 'train_auc',
                'val_loss', 'val_accuracy', 'val_f1_score', 'val_precision', 'val_recall', 'val_auc',
                'test_loss', 'test_accuracy', 'test_f1_score', 'test_precision', 'test_recall', 'test_auc'
            ]
        df = pd.DataFrame(columns=columns)
        return df
    
    def wrong_pred_viz(self, wrong_idx: torch.Tensor):
        '''
        Visualizes the wrong predictions
        '''
        wrong_acc_data = []
        for i in range(len(label_list)):
            if label_list[i] == 1 and pred_list[i] == 1:
                wrong_acc_data.append(acc_data[i])

        plt.figure(figsize=(24,16))
        for i in range(min(16, len(wrong_acc_data))):
            plt.subplot(4,4,i+1)
            plt.plot(acc_data[i])
            plt.title(f'Right Prediction {i+1}')
            plt.xlabel('Time')
            plt.ylabel('Acceleration')
        plt.savefig(f'{self.arg.work_dir}/Wrong_Predictions.png')
        plt.close()

    def cal_prediction(self, logits):
        return (torch.sigmoid(logits)>THRESHOLD).int().squeeze(1)
        #return torch.argmax(F.log_softmax(logits,dim =1), 1)

    def cal_metrics(self, targets, predictions): 
        targets = np.array(targets)
        predictions = np.array(predictions)
        f1 = f1_score(targets, predictions, zero_division=0)
        precision = precision_score(targets, predictions, zero_division=0)
        recall = recall_score(targets, predictions, zero_division=0)
        if np.unique(targets).size < 2 or np.unique(predictions).size < 2:
            auc_score = 0.5
        else:
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
        label_list = []
        pred_list = []
        cnt = 0
        loss_accum = 0.0

        process = tqdm(loader, ncols = 80)

        for batch_idx, (inputs, targets, idx) in enumerate(process):
            with torch.no_grad():
                acc_key = self._get_inertial_key(inputs)
                acc_data = inputs[acc_key].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                skl_tensor = None
                if self.use_skeleton and 'skeleton' in inputs:
                    skl_tensor = inputs['skeleton'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                targets = targets.to(f'cuda:{self.output_device}' if use_cuda else 'cpu')

            
            timer['dataloader'] += self.split_time()

            self.optimizer.zero_grad()
            logits, _= self.model(acc_data.float(), skl_tensor.float() if skl_tensor is not None else None)
            batch_loss = self.criterion(logits.squeeze(1), targets.float())
            batch_loss.backward()
            self.optimizer.step()

            timer['model'] += self.split_time()
            with torch.no_grad():
                loss_accum += batch_loss.item()
                preds = self.cal_prediction(logits)
                label_list.extend(targets.tolist())
                pred_list.extend(preds.tolist())

            cnt += len(targets)
            timer['stats'] += self.split_time()

        train_loss = loss_accum / cnt if cnt else 0.0
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
        train_metrics = self._format_metrics(train_loss, accuracy, f1, precision, recall, auc_score)
        self.current_fold_metrics['train'] = train_metrics
        self._record_epoch_metrics('train', epoch + 1, train_metrics)
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

        loss_accum = 0.0
        cnt = 0
        label_list = []
        pred_list = []
        wrong_idx = []

        process = tqdm(self.data_loader[loader_name], ncols=80)
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                acc_key = self._get_inertial_key(inputs)
                acc_data = inputs[acc_key].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                skl_tensor = None
                if self.use_skeleton and 'skeleton' in inputs:
                    skl_tensor = inputs['skeleton'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                targets = targets.to(f'cuda:{self.output_device}' if use_cuda else 'cpu')

                logits, _ = self.model(acc_data.float(), skl_tensor.float() if skl_tensor is not None else None)
                batch_loss = self.criterion(logits.squeeze(1), targets.float())
                loss_accum += batch_loss.item()
                preds = self.cal_prediction(logits)
                label_list.extend(targets.tolist())
                pred_list.extend(preds.tolist())
                cnt += len(targets)
            avg_loss = loss_accum / cnt if cnt else 0.0
            accuracy, f1, recall, precision, auc_score = self.cal_metrics(label_list, pred_list)

        if result_file is not None:
            predict = pred_list
            true = label_list

            for i, x in enumerate(predict):
                f_r.write(str(x) +  '==>' + str(true[i]) + '\n')
        metrics = self._format_metrics(avg_loss, accuracy, f1, precision, recall, auc_score)
        epoch_label = epoch + 1 if loader_name != 'test' else 'best'
        self._record_epoch_metrics(loader_name, epoch_label, metrics)
        self.print_log('{} Loss: {:4f}. {} Acc: {:2f}% F1 score: {:2f}%, Precision: {:2f}%, Recall: {:2f}%, AUC: {:2f}%'.format(loader_name.capitalize(), avg_loss, loader_name.capitalize(), accuracy, f1, precision, recall, auc_score))
        if loader_name == 'val':
            self.current_fold_metrics['val'] = metrics
            if avg_loss < self.best_loss :
                    self.best_loss = avg_loss
                    self.best_val_metrics = metrics
                    torch.save(deepcopy(self.model.state_dict()), f'{self.model_path}_{self.test_subject[0]}.pth')
                    self.print_log('Weights Saved')
        elif loader_name == 'test':
            self.current_fold_metrics['test'] = metrics
            self.test_accuracy = metrics['accuracy']
            self.test_f1 = metrics['f1_score']
            self.test_recall = metrics['recall']
            self.test_precision = metrics['precision']
            self.test_auc = metrics['auc']
        return avg_loss

    def start(self):

        '''
        Function to start the the training 

        '''

        if self.arg.phase == 'train':

                self.best_accuracy  = float('-inf')

                self.best_f1 = float('inf')
                self.print_log('Parameters: \n{}\n'.format(str(vars(self.arg))))
            
                results = self.create_df()
                # Filter out validation subjects from training/testing subjects
                available_subjects = [s for s in self.arg.subjects if s not in self.arg.validation_subjects]
                self.print_log(f'Total subjects: {len(self.arg.subjects)}')
                self.print_log(f'Validation subjects (held out): {self.arg.validation_subjects}')
                self.print_log(f'Available subjects for train/test: {available_subjects}')
                self.print_log(f'Number of train/test iterations: {len(available_subjects)}\n')

                for i in range(len(available_subjects)):
                    self.train_loss_summary = []
                    self.val_loss_summary = []
                    self.best_loss = float('inf')
                    test_subject = available_subjects[i]
                    self._init_fold_tracking(i, test_subject)
                    # Exclude both test subject AND validation subjects from training
                    train_subjects = [s for s in available_subjects if s != test_subject]
                    self.val_subject = self.arg.validation_subjects
                    self.test_subject = [test_subject]
                    self.train_subjects = train_subjects

                    self.print_log(f'\n{"="*60}')
                    self.print_log(f'Iteration {i+1}/{len(available_subjects)}')
                    self.print_log(f'Test subject: {test_subject}')
                    self.print_log(f'Validation subjects: {self.val_subject}')
                    self.print_log(f'Training subjects: {train_subjects}')
                    self.print_log(f'{"="*60}\n')

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
                    train_metrics = deepcopy(self.current_fold_metrics['train'])
                    val_metrics = deepcopy(self.best_val_metrics if self.best_val_metrics else self.current_fold_metrics['val'])
                    self.load_model(self.arg.model,self.arg.model_args)
                    self.load_weights()
                    self.model.eval()
                    self.print_log(f' ------------ Test Subject {self.test_subject[0]} -------')
                    self.eval(epoch = 0 , loader_name='test')
                    self.print_log(f'Test accuracy for : {self.test_accuracy}')
                    self.print_log(f'Test F-Score: {self.test_f1}')
                    self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                    test_metrics = deepcopy(self.current_fold_metrics['test'])
                    if not all([train_metrics, val_metrics, test_metrics]):
                        self.print_log(f'Warning: Missing metrics for subject {self.test_subject[0]}, skipping summary row.')
                        continue
                    fold_record = {
                        'test_subject': str(self.test_subject[0]),
                        'train': train_metrics,
                        'val': val_metrics,
                        'test': test_metrics
                    }
                    self.fold_metrics.append(fold_record)
                    row = {'test_subject': str(self.test_subject[0])}
                    for split_name, metrics in [('train', train_metrics), ('val', val_metrics), ('test', test_metrics)]:
                        rounded_metrics = self._round_metrics(metrics)
                        for metric_name, metric_value in rounded_metrics.items():
                            row[f'{split_name}_{metric_name}'] = metric_value
                    subject_result = pd.DataFrame([row])
                    results = pd.concat([results, subject_result], ignore_index=True)
                if not results.empty:
                    results = self.add_avg_df(results)
                    results.to_csv(f'{self.arg.work_dir}/scores.csv', index=False)
                if self.epoch_logs:
                    log_df = pd.DataFrame(self.epoch_logs)
                    log_columns = ['fold', 'test_subject', 'phase', 'epoch', 'loss', 'accuracy', 'f1_score', 'precision', 'recall', 'auc']
                    log_df = log_df.reindex(columns=log_columns)
                    log_df.to_csv(f'{self.arg.work_dir}/training_log.csv', index=False)
    
    def viz_feature(self, teacher_features, student_features, epoch):
        teacher_features = torch.flatten(teacher_features, start_dim=1)
        student_features = torch.flatten(student_features, start_dim= 1)
        plt.figure(figsize=(12,6))
        for i in range(8):
            plt.subplot(2,4,i+1)
            sns.kdeplot(teacher_features[i, :].cpu().detach().numpy(), bw_adjust=0.5, color = 'blue', label = 'T')
            
            plt.subplot(2,4,i+1)
            sns.kdeplot(student_features[i, :].cpu().detach().numpy(), bw_adjust=0.5, color = 'red', label = 'S')
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
