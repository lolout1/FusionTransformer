'''
Dataset Builder
'''
import os
from typing import List, Dict
import numpy as np

from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

from utils.processor.base import Processor

def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    '''Function to fitter noise '''
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data, axis=0) 

class DatasetBuilder:
    '''
    Builds a numpy file for the data and labels and 

    Args: 
        Dataset: a dataset class containing all matched files
    '''
    def __init__(self , dataset: object, mode: str, max_length: int, task = 'fd', **kwargs) -> None:
        assert mode in ['avg_pool' , 'sliding_window'], f'Unsupported processing method {mode}'
        self.dataset = dataset
        self.data : Dict[str, List[np.array]] = {}
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task

    
    def make_dataset(self, subjects : List[int]): 
        '''
        Reads all the files and makes a numpy  array with all data
        '''
        self.data = {}
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:        
                if self.task == 'fd': 
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                self.data['labels'] = self.data.get('labels',[])
                self.data['labels'].append(label)
                for modality, file_path in trial.files.items():
                    #here we need the processor class 
                    keys = self.kwargs.get('keys', None)
                    key = None
                    if keys:
                        key = keys[modality.lower()]
                    processor = Processor(file_path, self.mode, self.max_length, key = key)
                    try: 
                        unimodal_data = butterworth_filter(processor.process(), cutoff=1.0, fs=20)
                        self.data[modality] = self.data.get(modality, [])

                        self.data[modality].append(unimodal_data)

                    except Exception as e : 
                        print(e)
                        os.remove(file_path)
            
        for key in self.data:
            self.data[key] = np.stack(self.data[key], axis=0)

    
    def normalization(self) -> np.ndarray:
        '''
        Function to normalize  the data
        '''

        for key ,value  in self.data.items():        
            if key != 'labels':
                num_samples, length = value.shape[:2]
                norm_data = StandardScaler().fit_transform(value.reshape(num_samples*length, -1))
                self.data[key] = norm_data.reshape(num_samples, length, -1)

        return self.data
             