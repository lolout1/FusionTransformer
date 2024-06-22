import scipy 
from scipy.io import loadmat
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import re
import torch
import shutil
import os
from typing import Sequence
from bvh import Bvh
import torch.nn.functional as F
from bvh import Bvh
from sklearn.preprocessing import StandardScaler
from einops import rearrange
# TRAIN_SUBJECT = [2, 4, 5, 7,8, 10, 11,14, 15, 16, 17,19, 21, 23, 26]
# TEST_SUBJECT = [6]
TRAIN_SUBJECT = [ i for i in range(30, 37)]
TEST_SUBJECT = [37]

def avg_pool(sequence, window_size = 5, stride=1, max_length = 512 , shape = None):
    shape = sequence.shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis = 0).transpose(0,2, 1)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    stride =  ((sequence.shape[2]//max_length)+1 if max_length < sequence.shape[2] else 1)
    sequence = F.avg_pool1d(sequence,kernel_size=window_size, stride=stride)
    sequence = sequence.squeeze(0).numpy().transpose(1,0)
    sequence = sequence.reshape(-1, *shape[1:])
    return sequence


def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, input_shape: Sequence[int]) -> np.ndarray:
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled_sequence = avg_pool(sequence=sequence, max_length = max_sequence_length, 
                               shape = input_shape)
    new_sequence = np.zeros(shape, sequence.dtype)
    new_sequence[:len(pooled_sequence)] = pooled_sequence
    return new_sequence

def bvh2arr(file_path):
    with open(file = file_path, mode='r') as f: 
        bvh_data = Bvh(f.read())
        data = bvh_data.frames
        arr = np.array(data).astype(float)
    return arr      
  
def sliding_window(data, clearing_time_index, max_time, sub_window_size, stride_size):

    assert clearing_time_index >= sub_window_size - 1 , "Clearing value needs to be greater or equal to (window size - 1)"
    start = clearing_time_index - sub_window_size + 1 

    if max_time >= data.shape[0]-sub_window_size:
        max_time = max_time - sub_window_size + 1
        # 2510 // 100 - 1 25 #25999 1000 24000 = 24900

    sub_windows  = (
        start + 
        np.expand_dims(np.arange(sub_window_size), 0) + 
        np.expand_dims(np.arange(max_time, step = stride_size), 0).T
    )

    #labels = np.round(np.mean(labels[sub_windows], axis=1))
    return data[sub_windows] 

def process_data(raw_data, window_size, stride):
    # dataframe = pd.read_csv(file_path)
    # raw_data = dataframe[['w_accelerometer_x', 'w_accelerometer_y', 'w_accelerometer_z']].to_numpy()
    # labels = dataframe['outcome'].to_numpy()
    data = sliding_window(raw_data, window_size - 1,raw_data.shape[0],window_size,stride)
    return data

def bmhad_processing(data_dir = 'data/berkley_mhad', mode = 'train', acc_window_size = 32, skl_window_size = 32, num_windows = 10):
    file_paths = glob.glob(f'{data_dir}/{mode}_acc/*')
    skl_paths = f'{data_dir}/{mode}_skeleton/skl_'
    pattern = r's\d+_a\d+_r\d+'
    act_pattern = r'(a\d+)'
    label_pattern = r'(\d+)'
    skl_set =[]
    acc_set = []
    label_set = []

    for idx, path in enumerate(file_paths):
        data = np.genfromtxt(path)
        if np.size(data) == 0 : 
            continue

        acc_data = data[:, :3]
        desp = re.findall(pattern, file_paths[idx])[0]
        act_label = re.findall(act_pattern, path)[0]
        label = int(re.findall(label_pattern, act_label)[0])-1
        skl_file = skl_paths + desp + '.bvh'
        skl_data = bvh2arr(skl_file)
        skl_data = skl_data[::15, :]
        acc_stride = (acc_data.shape[0] - acc_window_size) // num_windows
        skl_stride =(skl_data.shape[0] - skl_window_size) // num_windows
        if acc_stride == 0 or skl_stride == 0:
            print(path)
            continue
        processed_acc = process_data(acc_data, acc_window_size, acc_stride)
        processed_skl = process_data(skl_data , skl_window_size, skl_stride)
        n,l, nc = processed_skl.shape
        processed_skl = processed_skl.reshape((n, l, -1, 3 ))
        sync_size = min(processed_skl.shape[0],processed_acc.shape[0])
        skl_set.append(processed_skl[:sync_size, :, : , :])
        acc_set.append(processed_acc[:sync_size, : , :])
        label_set.append(np.repeat(label, sync_size))
    concat_acc = np.concatenate(acc_set, axis = 0)
    concat_skl = np.concatenate(skl_set, axis = 0)
    concat_label = np.concatenate(label_set, axis = 0)
    dataset = { 'acc_data' : concat_acc,
                'skl_data' : concat_skl,  
                'labels': concat_label}

    #np.savez(file = f'/home/bgu9/Fall_Detection_KD_Multimodal/data/berkley_mhad/bhmad_uniformdis_skl50_{mode}', acc_data = concat_acc, skl_data = concat_skl, labels = concat_label )

    return dataset

def sf_processing(data_dir = 'data/smartfallmm', subjects = None,
                    skl_window_size = 32, 
                    num_windows = 10,
                    acc_window_size = 32,
                    num_joints = 32, num_channels = 3):
    skl_set = []
    acc_set = []
    label_set = []

    file_paths = glob.glob(f'{data_dir}/old_participant/skeleton/*.csv')
    print("file paths {}".format(len(file_paths)))
    #skl_path = f"{data_dir}/{mode}_skeleton_op/"
    #skl_path = f"{data_dir}/{mode}/skeleton/"
    acc_dir = f"{data_dir}/old_participant/phonewatch/w_accel"
    pattern = r'S\d+A\d+T\d+'
    act_pattern = r'(A\d+)'
    label_pattern = r'(\d+)'

    for idx,path in enumerate(file_paths):
        desp = re.findall(pattern, file_paths[idx])[0]
        if not int(desp[1:3]) in subjects:
            continue
        act_label = re.findall(act_pattern, path)[0]
        label = int(int(re.findall(label_pattern, act_label)[0])-1)

        acc_path = f'{acc_dir}/{desp}.csv'
        if os.path.exists(acc_path):
             acc_df = pd.read_csv(acc_path, header = 0).dropna()
        else: 
             continue

        acc_data = acc_df.bfill().iloc[2:, -3:].to_numpy(dtype=np.float32)
        
        skl_df  = pd.read_csv(path, index_col =False).dropna()
        skl_data = skl_df.bfill().iloc[:, -96:].to_numpy(dtype=np.float32)
        ######## avg poolin #########
        padded_acc = pad_sequence_numpy(sequence=acc_data, input_shape= acc_data.shape, max_sequence_length=acc_window_size)

        padded_skl = pad_sequence_numpy(sequence=skl_data, input_shape=skl_data.shape, max_sequence_length=skl_window_size)

        
        skl_data = rearrange(padded_skl, 't (j c) -> t j c' , j = 32, c = 3)
        acc_set.append(padded_acc)
        skl_set.append(skl_data)
        label_set.append(label)
        #skl_data = rearrange(skl_df.values[:, -96:], 't (j c) -> t j c' , j = 32, c = 3)

    #     acc_stride = 10
    #     processed_acc = process_data(acc_data, acc_window_size, acc_stride)
    #     processed_skl = process_data(skl_data, skl_window_size, acc_stride)
    #     sync_size = min(processed_acc.shape[0],processed_skl.shape[0])
    #     skl_set.append(processed_skl[:sync_size, :, : , :])
    #     acc_set.append(processed_acc[:sync_size, : , :])
    #     label_set.append(np.repeat(label, processed_acc.shape[0]))

    concat_acc = np.stack(acc_set, axis = 0)
    concat_skl = np.stack(skl_set, axis = 0)
    # #s,w,j,c = concat_skl.shape
    concat_label = np.stack(label_set, axis = 0)
     #print(concat_acc.shape[0], concat_label.shape[0])
    # _,count  = np.unique(concat_label, return_counts = True)
    # print(concat_acc.shape)
    # print(concat_skl.shape)
    # #np.savez('/home/bgu9/KD_Multimodal/train.npz' , data = concat_acc, labels = concat_label)
    dataset = { 'acc_data' : concat_acc,
                 'skl_data' : concat_skl, 
                 'labels': concat_label}
    return dataset


def czu_processing(data_dir = 'data/CZU-MHAD', mode = 'train',
                    acc_window_size = 32, skl_window_size = 32, num_windows = 10,
                    num_joints = 25, num_channels = 3):
    skl_set = []
    acc_set = []
    label_set = []

    file_paths = glob.glob(f'{data_dir}/{mode}/inertial/*.mat')
    print("file paths {}".format(len(file_paths)))
    #skl_path = f"{data_dir}/{mode}_skeleton_op/"
    skl_path = f"{data_dir}/{mode}/skeleton/"
    pattern = r'\w+_a\d+_t\d+'
    act_pattern = r'(a\d+)'
    label_pattern = r'(\d+)'
    for idx,path in enumerate(file_paths):
        desp = re.findall(pattern, file_paths[idx])[0]
        act_label = re.findall(act_pattern, path)[0]
        label = int(re.findall(label_pattern, act_label)[0])-1
        # if label in [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17,18,19,23,24]:
        #     acc_stride = 4
        #     skl_stride = 1
        # else: 
        #     acc_stride = 10
        #     skl_stride = 3
        acc_data = loadmat(path)['sensor'][3][0]

        acc_stride = (acc_data.shape[0] - acc_window_size) // num_windows
        acc_data = acc_data[::2, :-1]
        # print(acc_stride)
        processed_acc = process_data(acc_data, acc_window_size, acc_stride)
        # print(label)
        # print(processed_acc.shape)
        # skl_file = skl_path+desp+'_color_skeleton.npy'
        skl_data = loadmat( skl_path+desp+'.mat')['skeleton']
        skl_data = np.delete(skl_data, np.s_[3::4], axis = 1)

        skl_data = rearrange(skl_data, 't (j c) -> t j c' , j = num_joints, c = num_channels)
        
        skl_stride =(skl_data.shape[0] - skl_window_size) // num_windows
        if acc_stride <= 0 or skl_stride <= 0:
            print(path)
            continue
        #skl_data = np.squeeze(np.load(skl_file))
        t, j , c = skl_data.shape
        skl_data = rearrange(skl_data, 't j c -> t (j c)')
        processed_skl = process_data(skl_data, skl_window_size, skl_stride)
        skl_data = rearrange(processed_skl, 'n t (j c) -> n t j c', j =j, c =c)
        sync_size = min(skl_data.shape[0],processed_acc.shape[0])
        skl_set.append(skl_data[:sync_size, :, : , :])
        acc_set.append(processed_acc[:sync_size, : , :])
        label_set.append(np.repeat(label, sync_size))

    concat_acc = np.concatenate(acc_set, axis = 0)
    concat_skl = np.concatenate(skl_set, axis = 0)
    concat_label = np.concatenate(label_set, axis = 0)
    _,count  = np.unique(concat_label, return_counts = True)
    dataset = { 'acc_data' : concat_acc,
                'skl_data' : concat_skl, 
                'labels': concat_label}
    
    return dataset


def utd_processing(data_dir = 'data/UTD_MAAD', mode = 'val', acc_window_size = 32, skl_window_size = 32, num_windows = 10):
    skl_set = []
    acc_set = []
    label_set = []

    file_paths = glob.glob(f'{data_dir}/{mode}_inertial/*.mat')
    print("file paths {}".format(len(file_paths)))
    #skl_path = f"{data_dir}/{mode}_skeleton_op/"
    skl_path = f"{data_dir}/{mode}_skeleton/"
    pattern = r'a\d+_s\d+_t\d+'
    act_pattern = r'(a\d+)'
    label_pattern = r'(\d+)'
    for idx,path in enumerate(file_paths):
        desp = re.findall(pattern, file_paths[idx])[0]
        act_label = re.findall(act_pattern, path)[0]
        label = int(re.findall(label_pattern, act_label)[0])-1
        # if label in [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17,18,19,23,24]:
        #     acc_stride = 4
        #     skl_stride = 1
        # else: 
        #     acc_stride = 10
        #     skl_stride = 3
        acc_data = loadmat(path)['d_iner']
        acc_stride = (acc_data.shape[0] - acc_window_size) // num_windows
        acc_data = acc_data[::2, :]
        # print(acc_stride)
        processed_acc = process_data(acc_data, acc_window_size, acc_stride)
        # print(label)
        # print(processed_acc.shape)
        # skl_file = skl_path+desp+'_color_skeleton.npy'
        skl_data = loadmat( skl_path+desp+'_skeleton.mat')['d_skel']
        skl_data = rearrange(skl_data, 'j c t -> t j c')
        
        skl_stride =(skl_data.shape[0] - skl_window_size) // num_windows

        if acc_stride == 0 or skl_stride == 0:
            # print(path)
            continue
        #skl_data = np.squeeze(np.load(skl_file))
        t, j , c = skl_data.shape
        skl_data = rearrange(skl_data, 't j c -> t (j c)')
        processed_skl = process_data(skl_data, skl_window_size, skl_stride)
        skl_data = rearrange(processed_skl, 'n t (j c) -> n t j c', j =j, c =c)
        sync_size = min(skl_data.shape[0],processed_acc.shape[0])
        skl_set.append(skl_data[:sync_size, :, : , :])
        acc_set.append(processed_acc[:sync_size, : , :])
        label_set.append(np.repeat(label, sync_size))

    concat_acc = np.concatenate(acc_set, axis = 0)
    concat_skl = np.concatenate(skl_set, axis = 0)
    concat_label = np.concatenate(label_set, axis = 0)
    _,count  = np.unique(concat_label, return_counts = True)
    dataset = { 'acc_data' : concat_acc,
                'skl_data' : concat_skl, 
                'labels': concat_label}
    
    return dataset

def normalization(data_path = None,data = None,  new_path = None, acc_scaler = StandardScaler(), skl_scaler = StandardScaler(), mode = 'fit'):
    if data_path is not None and data is not None: 
        raise ValueError('Only one of data_path or data should be provided, not both')

    if data_path : 
        data = np.load(data_path)
    
    acc_data = data['acc_data']
    acc_ct, acc_ln, acc_cl = acc_data.shape
    reshape_acc = acc_data.reshape((acc_ct*acc_ln, -1))
    
    skl_data =data['skl_data']
    skl_ct, skl_ln, joints, skl_cl = skl_data.shape
    reshape_skl = skl_data.reshape(skl_ct*skl_ln, joints*skl_cl)
    
    if mode == 'fit' :
        acc_scaler.fit(reshape_acc)
        skl_scaler.fit(reshape_skl)

        
    norm_acc = acc_scaler.transform(reshape_acc). reshape(acc_ct, acc_ln, acc_cl)
    norm_skl = skl_scaler.transform(reshape_skl).reshape(skl_ct, skl_ln, joints, skl_cl)
    
    #np.savez(new_path, acc_data = norm_acc, skl_data = norm_skl, labels = data['labels'] )
    #dataset = {'acc_data': acc_data.astype(np.float64), 'skl_data': skl_data.astype(np.float64), 'labels': data['labels']}
    dataset = {'acc_data' : norm_acc, 'skl_data': norm_skl, 'labels': data['labels']}
    return dataset, acc_scaler, skl_scaler

def find_match_elements(pattern, elements): 
    #compile the regular expression
    try:
        regex_pattern = re.compile(pattern)
        #filtering the elements that match the regex pattern
        matching_elements = [element for element in elements if regex_pattern.search(element)]
        return matching_elements
    except:
        print(f'Error: {e}')
        
    return []

def move_files(source_folder, destination_folder, pattern):
    try:
        
        # Check if the source folder exists
        if not os.path.exists(source_folder):
            raise FileNotFoundError("Source folder does not exist.")
        
        # Check if the destination folder exists, if not, create it
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Get a list of files in the source folder
        files = os.listdir(source_folder)
        matched_files = find_match_elements( pattern, files)
        
        if not matched_files:
            raise Exception('Couldn\'t find files with the pattern')

        
        for file in matched_files:
            
            source_file_path = os.path.join(source_folder, file)
            destination_file_path = os.path.join(destination_folder, file)

            # Perform the move operation
            shutil.move(source_file_path, destination_file_path)
        print("Files moved successfully.")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # bmhad_processing(data_dir= '/home/bgu9/Fall_Detection_KD_Multimodal/data/berkley_mhad/',  acc_window_size = 50, skl_window_size = 50)
    # bmhad_processing(data_dir= '/home/bgu9/Fall_Detection_KD_Multimodal/data/berkley_mhad/',mode = 'val', acc_window_size = 50, skl_window_size = 50)
    #bmhad_processing(data_dir= '/home/bgu9/Fall_Detection_KD_Multimodal/data/berkley_mhad/',mode = 'test', acc_window_size = 50, skl_window_size = 50)
    dataset = sf_processing(subjects=TRAIN_SUBJECT)

