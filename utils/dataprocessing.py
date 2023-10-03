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
import torch.nn.functional as F
from bvh import Bvh
from sklearn.preprocessing import StandardScaler
from einops import rearrange

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

def utd_processing(data_dir = 'data/UTD_MAAD', mode = 'val'):
    skl_set = []
    acc_set = []
    label_set = []

    file_paths = glob.glob(f'{data_dir}/{mode}_inertial/*.mat')
    skl_path = f"{data_dir}/{mode}_skeleton_op/"
    pattern = r'a\d+_s\d+_t\d+'
    act_pattern = r'(a\d+)'
    label_pattern = r'(\d+)'
    acc_window_size = 128
    acc_stride = 10
    skl_window_size = 32
    skl_stride = 3
    for idx,path in enumerate(file_paths):
        desp = re.findall(pattern, file_paths[idx])[0]
        act_label = re.findall(act_pattern, path)[0]
        label = int(re.findall(label_pattern, act_label)[0])-1
        acc_data = loadmat(path)['d_iner']
        processed_acc = process_data(acc_data, acc_window_size, acc_stride)
        skl_file = skl_path+desp+'_color_skeleton.npy'
        skl_data = np.squeeze(np.load(skl_file))
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
    dataset = {'acc_data' : norm_acc, 'skl_data': norm_skl, 'labels': data['labels']}
    return dataset, acc_scaler, skl_scaler



    
    