'''
Dataset Builder
'''
import os
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from numpy.linalg import norm
from dtaidistance import dtw
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler 
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from utils.processor.base import Processor



def csvloader(file_path: str, **kwargs):
    '''
    Loads csv data
    '''
    file_data = pd.read_csv(file_path, index_col=False, header = 0).dropna().bfill()
    # num_col = file_data.shape[1]
    # num_extra_col = num_col % 
    # cols_to_select = num_col - num_extra_col
    if 'skeleton' in file_path: 
        cols = 96
    else: 
        cols  = 3
    activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
    return activity_data

def matloader(file_path: str, **kwargs):
    '''
    Loads MatLab files 
    '''
    key = kwargs.get('key',None)
    assert key in ['d_iner' , 'd_skel'] , f'Unsupported {key} for matlab file'
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {
    'csv' : csvloader, 
    'mat' : matloader
}

def avg_pool(sequence : np.array, window_size : int = 5, stride :int =1, 
             max_length : int = 512 , shape : int = None) -> np.ndarray:

    '''
    Executes average pooling to smoothen out the data

    '''
    shape = sequence.shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis = 0).transpose(0,2, 1)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    stride =  ((sequence.shape[2]//max_length)+1 if max_length < sequence.shape[2] else 1)
    sequence = F.avg_pool1d(sequence,kernel_size=window_size, stride=stride)
    sequence = sequence.squeeze(0).numpy().transpose(1,0)
    sequence = sequence.reshape(-1, *shape[1:])
    return sequence


def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, 
                       input_shape: np.array) -> np.ndarray:
    '''
    Pools and pads the sequence to uniform length

    Args:
        sequence : data 
        max_sequence_length(int) : the fixed length of data
        input_shape: shape of the data
    Return: 
        new_sequence: data after padding
    '''
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled_sequence = avg_pool(sequence=sequence, max_length = max_sequence_length, shape = input_shape)
    new_sequence = np.zeros(shape, sequence.dtype)
    new_sequence[:len(pooled_sequence)] = pooled_sequence
    return new_sequence

def sliding_window(data: Dict[str, np.ndarray], clearing_time_index: int, max_time: int,
                   sub_window_size: int, stride_size: int, label: int,
                   reference_key: str = 'skeleton') -> Dict[str, np.ndarray]:
    '''
    Sliding Window
    '''
    if reference_key not in data:
        # Fallback to any available modality (except labels)
        available_keys = [key for key in data.keys() if key != 'labels']
        if not available_keys:
            raise ValueError('No modality available for sliding window generation')
        reference_key = available_keys[0]

    assert clearing_time_index >= sub_window_size - 1 , "Clearing value needs to be greater or equal to (window size - 1)"
    start = clearing_time_index - sub_window_size + 1 
    reference_length = data[reference_key].shape[0]
    if max_time is None or max_time > reference_length:
        max_time = reference_length
    if reference_length < sub_window_size:
        raise ValueError(f"Reference modality '{reference_key}' shorter than window size ({reference_length} < {sub_window_size})")
    if max_time >= reference_length - sub_window_size:
        max_time = reference_length - sub_window_size + 1
        max_time = max(max_time, 1)
        # 2510 // 100 - 1 25 #25999 1000 24000 = 24900

    sub_windows  = (
        start + 
        np.expand_dims(np.arange(sub_window_size), 0) + 
        np.expand_dims(np.arange(max_time, step = stride_size), 0).T
    )

    for key in list(data.keys()):
        if key == 'labels':
            continue
        data[key] = data[key][sub_windows]
    data['labels'] =  np.repeat(label, len(data[reference_key]))
    return data

def quaternion_to_euler(q):
    rot = Rotation.from_quat(q)
    return rot.as_euler('xyz', degrees=True)

def fuse_inertial_data(data, window_size): 
    q = np.array([1, 0, 0, 0], dtype=np.float64)
    quaterions  = []
    length = len(data['accelerometer'])
    madgwick = Madgwick()
    for i in range(length): 
        transformed_windows = []
        for j in range(window_size): 
            gyro_data = data['gyroscope'][i][j,:]
            acc_data = data['accelerometer'][i][j,:]
            q  = madgwick.updateIMU(q, acc=acc_data, gyr=gyro_data)
            euler_angels = quaternion_to_euler(q)
            transformed_windows.append(euler_angels)
        quaterions.append(np.array(transformed_windows))
    data['fused'] = quaterions
    return data 

    

def selective_sliding_window(data: np.ndarray, window_size: int , peaks : list, label : int, fuse : bool) -> np.array: 

    windowed_data = defaultdict(np.ndarray)
    for modality, modality_data in data.items():
        windows = []
        for peak in peaks:
            start = max(0, peak - window_size)
            end = min(len(modality_data), start + window_size)
            # difference = length - (end-start)
            # if difference != 0 : 
            #     if start == 0 : 
            #         end = end + difference
            #     elif 
            if modality_data[start:end, :].shape[0] < window_size:
                continue
            windows.append(modality_data[start:end, :])
        windowed_data[modality] = windows
    if fuse and set(("accelerometer" , "gyroscope")).issubset(windowed_data): 
        windowed_data  = fuse_inertial_data(windowed_data, window_size)
    windowed_data['labels'] = np.repeat(label, len(windows))
    return windowed_data


def filter_data_by_ids(data : np.ndarray, ids : List[int]):
    '''
    Index the different modalities with only selected ids

    Arguements: 
        data : data dictionary with skeleton and inertial data
        skeleton_ids: skeleton data selected ids
        inertial_ids: inertial data selected ids
    Return : 
        changed data with selected ids
    '''
    return data[ids, :]




def filter_repeated_ids(path : List[Tuple[int, int]]) -> Tuple[set, set]:
    '''
    Filtering indices those match with mutliple other indices
    Arguements: 
        path : Tuple of indices defining the DTW path
    
    Return : 
        set of tuples containing the unique indices

    '''
    seen_first = set()
    seen_second = set()

    for (first , second) in path : 

        if first not in seen_first and second not in  seen_second: 
            seen_first.add(first)
            seen_second.add(second)
    
    return seen_first, seen_second

def align_sequence(data : Dict[str, np.ndarray] ) -> Dict[str, np.ndarray]: 
    '''
    Matching the skeleton and phone data using dynamic time warping 
    Args: 
        dataset: Dictionary containing skeleton and accelerometer data

    '''
    if 'skeleton' not in data:
        return data
    joint_id = 9
    #skeleton_before_dtw =  data['skeleton'][idx][:, (joint_id -1) * 3 : joint_id * 3 ]
    #seperating left wrist joint data
    dynamic_keys = sorted([key for key in data.keys() if key != "skeleton"])
    
    skeleton_joint_data = data['skeleton'][:, (joint_id -1) * 3 : joint_id * 3 ]
    inertial_data = data[dynamic_keys[0]]
    if len(dynamic_keys) > 1: 
        gyroscope_data = data[dynamic_keys[1]]
        min_len = min(inertial_data.shape[0], gyroscope_data.shape[0])
        inertial_data = inertial_data[:min_len, :]
        data[dynamic_keys[1]] = gyroscope_data[:min_len, :]

   # calcuating frobenis norm of skeleton and intertial data 
    skeleton_frob_norm = norm(skeleton_joint_data, axis = 1)
    interial_frob_norm = norm(inertial_data, axis = 1)
    
    # calculating dtw of the two sequence
    # path =  dtw.warping_path(
    #     skeleton_frob_norm.flatten(), 
    #     interial_frob_norm.flatten()
    # )

    distance, path  = fastdtw(interial_frob_norm[:, np.newaxis], skeleton_frob_norm[:, np.newaxis],dist = euclidean)


    interial_ids ,skeleton_idx ,= filter_repeated_ids(path)
    data['skeleton'] = filter_data_by_ids(data['skeleton'], list(skeleton_idx))
    for key in dynamic_keys: 
        data[key]= filter_data_by_ids(data[key],list(interial_ids))
    #skeleton_after_dtw = data['skeleton'][idx][:, (joint_id -1) * 3 : joint_id * 3 ]
    #plt.plot( np.arange(skeleton_before_dtw.shape[0]),skeleton_before_dtw[..., 0], '--r',
             #np.arange(skeleton_after_dtw.shape[0]), skeleton_after_dtw[..., 0], '--g')
    
    # plt.savefig(f'exps/comparision/comparision_before_after_dtw_{idx}.jpg')
    # plt.close()
    return data


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
        self.data = defaultdict(list)
        #self.processed_data : Dict[str, List[np.array]] = {'labels':[]}
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = None
        self.diff = []
        # Add configurable filtering option
        self.enable_filtering = kwargs.get('enable_filtering', True)
        self.filter_cutoff = kwargs.get('filter_cutoff', 5.5)
        self.filter_fs = kwargs.get('filter_fs', 25)
        self.use_skeleton = kwargs.get('use_skeleton', True)
        self.align_with_skeleton = kwargs.get('enable_skeleton_alignment', self.use_skeleton)

    def _get_reference_key(self, data: Dict[str, np.ndarray]) -> str:
        if self.use_skeleton and 'skeleton' in data:
            return 'skeleton'
        for candidate in ('accelerometer', 'gyroscope', 'fused'):
            if candidate in data:
                return candidate
        for key in data.keys():
            if key != 'labels':
                return key
        raise ValueError('No valid modality found for sliding window')

    def _synchronize_modalities(self, trial_data: Dict[str, np.ndarray]) -> int:
        """Trim all modalities to the minimum available length."""
        lengths = [value.shape[0] for key, value in trial_data.items() if key != 'labels']
        if not lengths:
            raise ValueError('Trial data contains no modalities to process')
        min_length = min(lengths)
        if min_length < self.max_length:
            raise ValueError(f'Minimum modality length {min_length} shorter than required window {self.max_length}')
        for key in list(trial_data.keys()):
            if key == 'labels':
                continue
            trial_data[key] = trial_data[key][:min_length]
        return min_length
    def load_file(self, file_path):
        '''
        
        '''
        loader = self._import_loader(file_path)
        data = loader(file_path, **self.kwargs)
        #self.set_input_shape(data)
        return data
    

    def _import_loader(self, file_path:str) -> np.array :
        '''
        Reads file and loads data from
         
        '''

        file_type = file_path.split('.')[-1]

        assert file_type in ['csv', 'mat'], f'Unsupported file type {file_type}'

        return LOADER_MAP[file_type]
    
    def process(self, data, label):
        '''
        function implementation to process data
        '''

        if self.mode == 'avg_pool':
            data = pad_sequence_numpy(sequence=data, max_sequence_length=self.max_length,
                                      input_shape=data.shape)
        
        else: 
            # sqrt_sum = np.sqrt(np.sum(data['accelerometer']**2, axis = 1))
            # if label == 1: 
            #     #phone height = 25, distance = 200
            #     #meta height = 1 distaince = 10 
            #     peaks , _ = find_peaks(sqrt_sum, height=15, distance=10)
                
            # else: 
            #     #phone height = 15, distance = 500
            #     peaks , _ = find_peaks(sqrt_sum, height=15, distance=15)

            # data = selective_sliding_window(data, window_size= self.max_length,peaks = peaks, label = label, fuse = self.fuse)
            reference_key = self._get_reference_key(data)
            data = sliding_window(
                data,
                self.max_length - 1,
                data[reference_key].shape[0],
                self.max_length,
                32,
                label,
                reference_key=reference_key
            )
        return data

    def _add_trial_data(self, trial_data):

        for modality, modality_data in trial_data.items():
            self.data[modality].append(modality_data)
    
    def _len_check(self, d):
        return all(len(v) > 1 for v in d.values())

    def get_size_diff(self, trial_data):
        return trial_data['accelerometer'].shape[0]  - trial_data['skeleton'].shape[0]

    def store_trial_diff(self, difference):
        self.diff.append(difference)
    
    def viz_trial_diff(self):
        value_range = range(min(self.diff) , max(self.diff)+2)
        # plt.hist(self.diff, bins = value_range, edgecolor = 'black', alpha = 0.7)
        # plt.xlabel("Differences")
        # plt.ylabel("Frequency")
        print(len(self.diff))
        counter = Counter(self.diff)

        #   Extract values for plotting


        plt.hist(self.diff, bins=range(min(self.diff), max(self.diff) + 2, 200), edgecolor='black', alpha=0.7)

        # Labels and title
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of Numbers")
        plt.savefig("Distribution.png")

    def select_subwindow_pandas(self, unimodal_data):
        n = len(unimodal_data)
        magnitude = np.linalg.norm(unimodal_data, axis = 1)
        df = pd.DataFrame({"values":magnitude})
        #250
        df["variance"] = df["values"].rolling(window=125).var()

        # Get index of highest variance
        max_idx = df["variance"].idxmax()

        # Get segment
        final_start = max(0, max_idx-100)
        final_end = min(n, max_idx + 100)
        return unimodal_data[final_start:final_end, :]
        #high_var_segment = df["values"].iloc[max_idx : max_idx + 200].values

    def make_dataset(self, subjects : List[int], fuse : bool): 
        '''
        Reads all the files and makes a numpy  array with all data
        '''
        self.data = defaultdict(list)
        self.fuse = fuse
        self.processed_data : Dict[str, List[np.array]] = {'labels':[]}
        count = 0 
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:       
                if self.task == 'fd': 
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                #self.data['labels'] = self.data.get('labels',[])
                trial_data = defaultdict(np.ndarray)
                
                for modality, file_path in trial.files.items():
                    #here we need the processor class 
                    keys = self.kwargs.get('keys', None)
                    key = None
                    if keys:
                        key = keys[modality.lower()]
                    #processor = Processor(file_path, self.mode, self.max_length,label, key = key)
                    try:
                        executed  = True
                        unimodal_data = self.load_file(file_path)
                        # Apply configurable Butterworth filtering to inertial data
                        if modality in ['accelerometer' , 'gyroscope'] and self.enable_filtering:
                            unimodal_data = butterworth_filter(unimodal_data, cutoff=self.filter_cutoff, fs=self.filter_fs)
                        trial_data[modality] = unimodal_data
                        if modality in ['accelerometer', 'gyroscope'] and unimodal_data.shape[0]>250:
                            trial_data[modality] = self.select_subwindow_pandas(unimodal_data)                            
                        # if modality == 'skeleton':
                        #     print(unimodal_data.shape)

                    except Exception as e :
                        executed = False
                        print(e)
                # trial_difference = self.get_size_diff(trial_data)
                # self.store_trial_diff(trial_difference)
                
                if executed : 
                    if self.align_with_skeleton and 'skeleton' in trial_data:
                        trial_data = align_sequence(trial_data)
                        # os.remove(file_path)
                    try:
                        self._synchronize_modalities(trial_data)
                    except ValueError as err:
                        print(f"Skipping trial due to modality length issue: {err}")
                        continue
                    try:
                        trial_data = self.process(trial_data, label)
                    except ValueError as err:
                        print(f"Skipping trial due to preprocessing error: {err}")
                        continue
                    #print(trial_data['skeleton'][0].shape)
                    if self._len_check(trial_data):
                        self._add_trial_data(trial_data)
                # for modality, file_path in trial_data.files.items():
                #     window_stack = self.process(trial_data[modality])
                #     if len(window_stack) != 0 : 
                #         trial_data[modality] = window_stack
                #trial_data['labels'].append(np.repeat(label,len(window_stack)))
                
                # for modality, file_path in trial.files.items():
                #     processor = Processor(file_path, self.mode, self.max_length, label,  key = key)
                #     processor.set_input_shape(self.data[modality][count-1])
                #     window_stack = processor.process(self.data[modality][count-1])
                #     if len(window_stack) != 0 :
                #         self.processed_data[modality] = self.processed_data.get(modality, [])
                #         self.processed_data[modality].append(window_stack)
                # #if processor.input_shape[0] >= self.max_length:
                #         self.processed_data['labels'].append(np.repeat(label,len(window_stack)))

                    #print(self.data['skeleton'][1].shape)
                #print(count)
                #count +=1
        #self.viz_trial_diff()
        for key in self.data:
            #print(key)
            #print(len(self.processed_data[key]))
            self.data[key] = np.concatenate(self.data[key], axis=0)
        # if len(self.data['skeleton']) > 0: 
        #     self.random_resampling()

    
    def random_resampling(self):
        ros = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        num_samples, seq_len, acc_channels = self.data['accelerometer'].shape
        _, _ , skl_channels = self.data['skeleton'].shape
        acc_flatten = self.data['accelerometer'].reshape(num_samples, -1)
        skl_flatten = self.data['skeleton'].reshape(num_samples, -1)

        labels = self.data['labels']
        resampled_acc, resampled_labels = ros.fit_resample(acc_flatten,labels)
        resampled_skl, _ = ros.fit_resample(skl_flatten, labels)
        self.data['accelerometer'] = resampled_acc.reshape(-1, seq_len , acc_channels)
        self.data['skeleton'] = resampled_skl.reshape(-1, seq_len, skl_channels)
        self.data['labels'] = resampled_labels


    
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
    
