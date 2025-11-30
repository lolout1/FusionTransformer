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
from utils.metrics_report import (save_enhanced_results, generate_text_report,
                                  create_scores_csv_compatible)

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
    parser.add_argument('--device', nargs='+', default=[0], type=device_type)

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
    parser.add_argument('--validation-subjects', nargs='+', type=int, default=[38,44],
                        help='Subjects reserved for validation (excluded from training). Optimized for 60% ADLs in both acc-only and acc+gyro.')
    parser.add_argument('--train-only-subjects', nargs='+', type=int, default=[],
                        help='Subjects permanently fixed in training, excluded from LOSO test iterations. '
                             'Use [29,32,35,39] for IMU models - these have poor gyroscope data quality.')
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

def device_type(v):
    '''
    Parse device argument: accepts integer GPU ID or "cpu"
    '''
    if v.lower() == 'cpu':
        return 'cpu'
    try:
        return int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f'Device must be "cpu" or an integer GPU ID, got: {v}')
    
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
        # Dataset statistics tracking for comprehensive CSV
        self.dataset_statistics = []
        self.builder = None  # Will store DatasetBuilder reference to access statistics
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
    
    def _get_device_str(self):
        '''Returns the device string for tensor allocation'''
        if self.output_device == 'cpu':
            return 'cpu'
        if torch.cuda.is_available():
            return f'cuda:{self.output_device}'
        return 'cpu'

    def cal_weights(self):
        label_count = Counter(self.norm_train['labels'])
        self.pos_weights = torch.Tensor([label_count[0] / label_count[1]])
        self.pos_weights = self.pos_weights.to(self._get_device_str())

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

    def _log_dataset_split(self) -> None:
        """Log the number of windows available per split at the beginning of a fold, including class distribution."""
        from collections import Counter

        def _count_windows(split: Dict[str, np.ndarray]) -> int:
            if not split or 'labels' not in split or split['labels'] is None:
                return 0
            return len(split['labels'])

        def _count_classes(split: Dict[str, np.ndarray]) -> tuple:
            """Count fall (label=1) and ADL (label=0) windows in a split."""
            if not split or 'labels' not in split or split['labels'] is None:
                return 0, 0
            labels = split['labels']
            label_counts = Counter(labels)
            adl = int(label_counts.get(0, 0))
            fall = int(label_counts.get(1, 0))
            return fall, adl

        fold_id = (self.current_fold_index + 1) if self.current_fold_index is not None else 'N/A'
        train_windows = _count_windows(self.norm_train)
        val_windows = _count_windows(self.norm_val)
        test_windows = _count_windows(self.norm_test)

        train_fall, train_adl = _count_classes(self.norm_train)
        val_fall, val_adl = _count_classes(self.norm_val)
        test_fall, test_adl = _count_classes(self.norm_test)

        train_subjects = len(self.train_subjects) if self.train_subjects else 0
        val_subjects = len(self.val_subject) if self.val_subject else 0
        test_subject = self.test_subject[0] if self.test_subject else 'unknown'

        # Log with class distribution
        self.print_log(
            f'[Fold {fold_id}] Dataset windows -> '
            f'train: {train_windows} ({train_subjects} subjects, fall={train_fall}, adl={train_adl}), '
            f'val: {val_windows} ({val_subjects} subjects, fall={val_fall}, adl={val_adl}), '
            f'test: {test_windows} (subject {test_subject}, fall={test_fall}, adl={test_adl})'
        )

        # Store dataset statistics for later export to comprehensive CSV
        dataset_stats = {
            'fold': fold_id,
            'test_subject': test_subject,
            'train_subjects': ','.join(map(str, self.train_subjects)) if self.train_subjects else '',
            'val_subject': ','.join(map(str, self.val_subject)) if self.val_subject else '',
            'train_windows': train_windows,
            'train_fall': train_fall,
            'train_adl': train_adl,
            'train_subjects_count': train_subjects,
            'val_windows': val_windows,
            'val_fall': val_fall,
            'val_adl': val_adl,
            'val_subjects_count': val_subjects,
            'test_windows': test_windows,
            'test_fall': test_fall,
            'test_adl': test_adl
        }

        # Store in instance variable for later CSV export
        if not hasattr(self, 'dataset_statistics'):
            self.dataset_statistics = []
        self.dataset_statistics.append(dataset_stats)
    
    def has_empty_value(self, lists):
        """Check if any array/list in the collection is empty"""
        return any(len(lst) == 0 for lst in lists)

    def has_inertial_data(self, dataset_dict):
        """
        Ensure at least one inertial modality (accelerometer/gyroscope) is present with samples.
        """
        if not dataset_dict:
            return False
        for key in ('accelerometer', 'gyroscope'):
            if key in dataset_dict and len(dataset_dict[key]) > 0:
                return True
        return False

    def validate_split(self, dataset_dict, split_name):
        """
        Validate a split before creating a DataLoader to avoid crashes when data is missing.
        """
        if not dataset_dict:
            self.print_log(f'Warning: {split_name} data is empty. Skipping iteration.')
            return False
        if not self.has_inertial_data(dataset_dict):
            self.print_log(f'Warning: {split_name} data is missing accelerometer/gyroscope streams. Skipping iteration.')
            return False
        if self.has_empty_value(list(dataset_dict.values())):
            self.print_log(f'Warning: {split_name} data contains empty arrays. Skipping iteration.')
            return False
        return True
    def load_model(self, model, model_args):
        '''
        Function to load model
        '''
        use_cuda = torch.cuda.is_available()
        raw_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device

        if raw_device == 'cpu':
            self.output_device = 'cpu'
            device = 'cpu'
        elif use_cuda:
            self.output_device = raw_device
            device = f'cuda:{self.output_device}'
        else:
            self.output_device = 'cpu'
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
            self.builder = prepare_smartfallmm(self.arg)

            # Load all splits (don't print validation yet - wait until all splits are loaded)
            self.norm_train = split_by_subjects(self.builder, self.train_subjects, self.fuse, print_validation=False)
            if not self.validate_split(self.norm_train, 'train'):
                return False
            self.norm_val = split_by_subjects(self.builder , self.val_subject, self.fuse, print_validation=False)
            if not self.validate_split(self.norm_val, f'validation subjects {self.val_subject}'):
                return False

            # Get SMV control flags from dataset_args (default: True for backward compatibility)
            include_smv = self.arg.dataset_args.get('include_smv', True)
            include_gyro_mag = self.arg.dataset_args.get('include_gyro_mag', True)

            #validation dataset
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args,
                               dataset=self.norm_train,
                               include_smv=include_smv,
                               include_gyro_mag=include_gyro_mag),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)

            self.cal_weights()

            self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')

            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.val_feeder_args,
                               dataset=self.norm_val,
                               include_smv=include_smv,
                               include_gyro_mag=include_gyro_mag),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
            #self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, 'val')
            self.norm_test = split_by_subjects(self.builder , self.test_subject, self.fuse, print_validation=False)
            test_split_name = f'test subject {self.test_subject[0]}' if self.test_subject else 'test'
            if not self.validate_split(self.norm_test, test_split_name):
                    return  False
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args,
                               dataset=self.norm_test,
                               include_smv=include_smv,
                               include_gyro_mag=include_gyro_mag),
                batch_size=self.arg.test_batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
            self.distribution_viz(self.norm_test['labels'], self.arg.work_dir, f'test_{self.test_subject[0]}')

            # Print comprehensive validation and skip summaries after all splits are loaded
            self.builder.print_validation_summary()
            self.builder.print_skip_summary()

            self._log_dataset_split()
            return True

    def record_time(self):
        '''
        Function to record time
        '''
        self.cur_time = time.time()
        return self.cur_time

    def _sanitize_row_dict(self, row_dict: dict) -> dict:
        """
        Remove duplicate keys from a dictionary by keeping only the last value.
        This prevents pandas from creating DataFrames with duplicate columns.

        Args:
            row_dict: Dictionary potentially containing duplicate keys

        Returns:
            Dictionary with unique keys only
        """
        # Convert to dict to ensure uniqueness (dict preserves insertion order in Python 3.7+)
        return dict(row_dict)

    def _build_metrics_row(self, fold_stat: dict, fold_index: int) -> dict:
        """
        Build a comprehensive metrics row combining fold statistics, model metrics,
        and builder statistics.

        Args:
            fold_stat: Base fold statistics dictionary
            fold_index: Index of the current fold

        Returns:
            Complete row dictionary with all metrics
        """
        # Start with a fresh dictionary to avoid duplicate keys
        row = {}

        # Add dataset split information (only non-metric keys)
        for key, value in fold_stat.items():
            # Skip any metric keys that might be in fold_stat to prevent duplicates
            if not any(key.startswith(f'{split}_') for split in ['train', 'val', 'test']):
                row[key] = value

        # Add model metrics if available
        if fold_index < len(self.fold_metrics):
            fold_metric = self.fold_metrics[fold_index]
            for split in ['train', 'val', 'test']:
                if split in fold_metric and fold_metric[split]:
                    metrics = self._round_metrics(fold_metric[split])
                    for metric_name, metric_value in metrics.items():
                        metric_key = f'{split}_{metric_name}'
                        row[metric_key] = metric_value

        # Add builder-level statistics
        if self.builder:
            skip_stats = self.builder.skip_stats

            # Add skip statistics
            row['total_trials_attempted'] = skip_stats.get('total_trials', 0)
            row['valid_trials_processed'] = skip_stats.get('valid_trials', 0)

            if skip_stats.get('total_trials', 0) > 0:
                row['skip_rate'] = round(
                    (skip_stats['total_trials'] - skip_stats['valid_trials']) / skip_stats['total_trials'] * 100,
                    2
                )
            else:
                row['skip_rate'] = 0.0

            row['skipped_missing'] = skip_stats.get('skipped_missing_modality', 0)
            row['skipped_mismatch'] = skip_stats.get('skipped_length_mismatch', 0)
            row['skipped_short'] = skip_stats.get('skipped_too_short', 0)
            row['skipped_dtw'] = skip_stats.get('skipped_dtw_length_mismatch', 0)
            row['skipped_preprocessing'] = skip_stats.get('skipped_preprocessing_error', 0)

            # Motion filtering statistics
            motion_enabled = self.arg.dataset_args.get('enable_motion_filtering', False)
            row['motion_enabled'] = motion_enabled
            if motion_enabled:
                row['motion_total'] = skip_stats.get('motion_total_windows', 0)
                row['motion_passed'] = skip_stats.get('motion_passed_windows', 0)
                row['motion_rejected'] = skip_stats.get('motion_rejected_windows', 0)
                row['motion_rejection_rate'] = round(
                    skip_stats.get('motion_rejection_rate', 0.0) * 100,
                    2
                )
            else:
                row['motion_total'] = None
                row['motion_passed'] = None
                row['motion_rejected'] = None
                row['motion_rejection_rate'] = None

        return row

    def _add_average_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add an average row to the DataFrame for numeric columns.

        Args:
            df: DataFrame to add average row to

        Returns:
            DataFrame with average row appended
        """
        if len(df) <= 1:
            return df

        # Calculate averages for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        averages = df[numeric_cols].mean()

        # Build average row with proper column alignment
        avg_row = {}
        for col in df.columns:
            # Special handling for identifier columns (always set to 'Average')
            if col in ['test_subject', 'val_subject', 'fold']:
                avg_row[col] = 'Average' if col == 'test_subject' else None
            elif col in numeric_cols:
                # Round based on column type
                if 'loss' in col:
                    avg_row[col] = round(averages[col], 6)
                else:
                    avg_row[col] = round(averages[col], 2)
            else:
                avg_row[col] = None

        # Create new DataFrame with average row and concatenate
        avg_df = pd.DataFrame([avg_row])

        # Ensure column order matches
        avg_df = avg_df[df.columns]

        # Use robust concatenation
        result = pd.concat([df, avg_df], ignore_index=True, sort=False)

        return result

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder DataFrame columns for better readability.

        Args:
            df: DataFrame to reorder

        Returns:
            DataFrame with reordered columns
        """
        # Define priority columns
        priority_cols = [
            'fold', 'test_subject', 'train_subjects', 'val_subject',
            'train_windows', 'train_fall', 'train_adl', 'train_subjects_count',
            'val_windows', 'val_fall', 'val_adl', 'val_subjects_count',
            'test_windows', 'test_fall', 'test_adl'
        ]

        # Categorize columns
        existing_priority = [col for col in priority_cols if col in df.columns]
        metric_cols = [col for col in df.columns if any(
            col.startswith(f'{split}_') for split in ['train', 'val', 'test']
        ) and col not in existing_priority]
        other_cols = [col for col in df.columns
                     if col not in existing_priority and col not in metric_cols]

        # Reorder
        ordered_cols = existing_priority + metric_cols + other_cols

        return df[ordered_cols]

    def save_comprehensive_statistics(self) -> None:
        """
        Save comprehensive dataset statistics including class splits, motion filtering,
        and per-subject breakdowns to a CSV file.

        This method creates a robust CSV export with proper handling of:
        - Duplicate column prevention
        - Metric aggregation from multiple sources
        - Average row calculation
        - Column ordering for readability
        """
        if not self.dataset_statistics or not self.builder:
            self.print_log("No dataset statistics available for comprehensive CSV export")
            return

        try:
            # Build comprehensive rows
            comprehensive_rows = []
            for i, fold_stat in enumerate(self.dataset_statistics):
                row = self._build_metrics_row(fold_stat, i)
                comprehensive_rows.append(row)

            # Create DataFrame
            if not comprehensive_rows:
                self.print_log("No comprehensive statistics to save")
                return

            df = pd.DataFrame(comprehensive_rows)

            # Verify no duplicate columns exist
            if df.columns.duplicated().any():
                duplicate_cols = df.columns[df.columns.duplicated()].tolist()
                self.print_log(f"Warning: Duplicate columns detected and removed: {duplicate_cols}")
                # Keep only the last occurrence of each column
                df = df.loc[:, ~df.columns.duplicated(keep='last')]

            # Reorder columns for readability
            df = self._reorder_columns(df)

            # Add average row
            df = self._add_average_row(df)

            # Save to CSV
            output_path = f'{self.arg.work_dir}/scores_comprehensive.csv'
            df.to_csv(output_path, index=False)
            self.print_log(f'Comprehensive statistics saved to: {output_path}')

        except Exception as e:
            self.print_log(f"Error saving comprehensive statistics: {str(e)}")
            import traceback
            self.print_log(traceback.format_exc())

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

        device_str = self._get_device_str()
        for batch_idx, (inputs, targets, idx) in enumerate(process):
            with torch.no_grad():
                acc_key = self._get_inertial_key(inputs)
                acc_data = inputs[acc_key].to(device_str)
                skl_tensor = None
                if self.use_skeleton and 'skeleton' in inputs:
                    skl_tensor = inputs['skeleton'].to(device_str)
                targets = targets.to(device_str)

            
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
        device_str = self._get_device_str()
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                acc_key = self._get_inertial_key(inputs)
                acc_data = inputs[acc_key].to(device_str)
                skl_tensor = None
                if self.use_skeleton and 'skeleton' in inputs:
                    skl_tensor = inputs['skeleton'].to(device_str)
                targets = targets.to(device_str)

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

                # Auto-select optimal validation subjects and train-only subjects based on config
                from utils.val_split_selector import (
                    get_optimal_validation_subjects, get_validation_split_info,
                    validate_imu_validation_subjects, get_train_only_subjects,
                    POOR_GYRO_SUBJECTS
                )
                optimal_val_subjects = get_optimal_validation_subjects(self.arg.dataset_args)
                optimal_train_only = get_train_only_subjects(self.arg.dataset_args)

                # Override default validation subjects with optimal selection
                # This ensures proper ADL ratios for motion-filtered vs non-filtered experiments
                if self.arg.validation_subjects == [38, 44]:  # Default value
                    self.arg.validation_subjects = optimal_val_subjects
                    self.print_log(f'Auto-selected validation subjects: {optimal_val_subjects}')
                    self.print_log(f'  → {get_validation_split_info(optimal_val_subjects)}')
                elif self.arg.validation_subjects != optimal_val_subjects:
                    self.print_log(f'Using custom validation subjects: {self.arg.validation_subjects}')
                    self.print_log(f'  (Optimal would be: {optimal_val_subjects})')

                # Auto-select train-only subjects for IMU models if not explicitly set
                # These subjects have poor gyro data and should not be used for testing
                current_train_only = getattr(self.arg, 'train_only_subjects', []) or []
                if not current_train_only and optimal_train_only:
                    self.arg.train_only_subjects = optimal_train_only
                    self.print_log(f'Auto-selected train-only subjects: {optimal_train_only}')
                    self.print_log(f'  → These subjects have poor gyroscope data quality')
                elif current_train_only:
                    self.print_log(f'Using configured train-only subjects: {current_train_only}')

                # Validate that validation subjects don't include poor gyro subjects for IMU models
                is_valid, problematic = validate_imu_validation_subjects(
                    self.arg.validation_subjects, self.arg.dataset_args, print_warning=True
                )
                if not is_valid:
                    self.print_log(f'WARNING: Validation subjects {problematic} have poor gyro data!')
                    self.print_log(f'  Train-only subjects (not for val/test): {POOR_GYRO_SUBJECTS}')

                results = self.create_df()

                # Get train-only subjects (permanently fixed in training, never tested)
                # These subjects have poor data quality for certain modalities (e.g., corrupt gyro timestamps)
                train_only_subjects = getattr(self.arg, 'train_only_subjects', []) or []

                # Filter out validation subjects AND train-only subjects from test candidates
                # Train-only subjects are always in training, never become test subjects
                test_candidates = [s for s in self.arg.subjects
                                   if s not in self.arg.validation_subjects
                                   and s not in train_only_subjects]

                self.print_log(f'Total subjects: {len(self.arg.subjects)}')
                self.print_log(f'Validation subjects (held out): {self.arg.validation_subjects}')
                if train_only_subjects:
                    self.print_log(f'Train-only subjects (never tested): {train_only_subjects}')
                self.print_log(f'Test candidates for LOSO: {test_candidates}')
                self.print_log(f'Number of LOSO iterations: {len(test_candidates)}\n')

                for i in range(len(test_candidates)):
                    self.train_loss_summary = []
                    self.val_loss_summary = []
                    self.best_loss = float('inf')
                    test_subject = test_candidates[i]
                    self._init_fold_tracking(i, test_subject)

                    # Training includes: all test candidates except current test subject + train-only subjects
                    # This ensures train-only subjects always contribute to training
                    train_subjects = [s for s in test_candidates if s != test_subject] + train_only_subjects
                    self.val_subject = self.arg.validation_subjects
                    self.test_subject = [test_subject]
                    self.train_subjects = train_subjects

                    self.print_log(f'\n{"="*60}')
                    self.print_log(f'Iteration {i+1}/{len(test_candidates)}')
                    self.print_log(f'Test subject: {test_subject}')
                    self.print_log(f'Validation subjects: {self.val_subject}')
                    self.print_log(f'Training subjects ({len(train_subjects)}): {sorted(train_subjects)}')
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

                    # Enhanced reporting: Save detailed per-fold analysis
                    if self.fold_metrics:
                        model_name = self.arg.model.split('.')[-1] if '.' in self.arg.model else self.arg.model
                        save_enhanced_results(
                            fold_metrics=self.fold_metrics,
                            output_dir=self.arg.work_dir,
                            model_name=model_name
                        )
                        # Print summary to console
                        summary_text = generate_text_report(self.fold_metrics, model_name)
                        self.print_log("\n" + summary_text)

                if self.epoch_logs:
                    log_df = pd.DataFrame(self.epoch_logs)
                    log_columns = ['fold', 'test_subject', 'phase', 'epoch', 'loss', 'accuracy', 'f1_score', 'precision', 'recall', 'auc']
                    log_df = log_df.reindex(columns=log_columns)
                    log_df.to_csv(f'{self.arg.work_dir}/training_log.csv', index=False)

                # Save comprehensive statistics including class splits and motion filtering
                self.save_comprehensive_statistics()
    
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
