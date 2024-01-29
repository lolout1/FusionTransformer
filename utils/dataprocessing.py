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
from bvh import Bvh
import torch.nn.functional as F
from bvh import Bvh
from sklearn.preprocessing import StandardScaler
from einops import rearrange

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

    np.savez(file = f'/home/bgu9/Fall_Detection_KD_Multimodal/data/berkley_mhad/bhmad_uniformdis_skl50_{mode}', acc_data = concat_acc, skl_data = concat_skl, labels = concat_label )

    return dataset

def sf_processing(data_dir = 'data/smartfallmm', mode = 'train',
                    skl_window_size = 32, num_windows = 10,
                    num_joints = 32, num_channels = 3):
    skl_set = []
    acc_set = []
    label_set = []

    file_paths = glob.glob(f'{data_dir}/{mode}/skeleton/*.csv')
    print("file paths {}".format(len(file_paths)))
    #skl_path = f"{data_dir}/{mode}_skeleton_op/"
    #skl_path = f"{data_dir}/{mode}/skeleton/"
    pattern = r'S\d+A\d+T\d+'
    act_pattern = r'(A\d+)'
    label_pattern = r'(\d+)'
    for idx,path in enumerate(file_paths):
        desp = re.findall(pattern, file_paths[idx])[0]
        act_label = re.findall(act_pattern, path)[0]
        label = int(re.findall(label_pattern, act_label)[0])-1

        # acc_data = loadmat(path)['sensor'][1][0]

        # acc_stride = (acc_data.shape[0] - acc_window_size) // num_windows
        # acc_data = acc_data[::2, :-1]
        # processed_acc = process_data(acc_data, acc_window_size, acc_stride)
        skl_df  = pd.read_csv(path)
        if skl_df.shape[1] > 97:
            continue
        skl_data = skl_df.iloc[: , 1:]
        #skl_data = np.delete(skl_data, np.s_[3::4], axis = 1)

        skl_data = rearrange(skl_data.values, 't (j c) -> t j c' , j = num_joints, c = num_channels)
        
        skl_stride =(skl_data.shape[0] - skl_window_size) // num_windows
        # if acc_stride <= 0 or skl_stride <= 0:
        #     print(path)
        #     continue
        #skl_data = np.squeeze(np.load(skl_file))
        t, j , c = skl_data.shape
        skl_data = rearrange(skl_data, 't j c -> t (j c)')
        processed_skl = process_data(skl_data, skl_window_size, skl_stride)
        skl_data = rearrange(processed_skl, 'n t (j c) -> n t j c', j =j, c =c)
        # sync_size = min(skl_data.shape[0],processed_acc.shape[0])
        skl_set.append(skl_data[:, :, : , :])
        # acc_set.append(processed_acc[:sync_size, : , :])
        label_set.append(np.repeat(label, skl_data.shape[0]))

    # concat_acc = np.concatenate(acc_set, axis = 0)
    concat_skl = np.concatenate(skl_set, axis = 0)
    s,w,j,c = concat_skl.shape
    concat_acc = np.random.randn(s,w,3)
    concat_label = np.concatenate(label_set, axis = 0)
    _,count  = np.unique(concat_label, return_counts = True)
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
        acc_data = loadmat(path)['sensor'][1][0]

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



if __name__ == "__main__":
    # bmhad_processing(data_dir= '/home/bgu9/Fall_Detection_KD_Multimodal/data/berkley_mhad/',  acc_window_size = 50, skl_window_size = 50)
    # bmhad_processing(data_dir= '/home/bgu9/Fall_Detection_KD_Multimodal/data/berkley_mhad/',mode = 'val', acc_window_size = 50, skl_window_size = 50)
    # bmhad_processing(data_dir= '/home/bgu9/Fall_Detection_KD_Multimodal/data/berkley_mhad/',mode = 'test', acc_window_size = 50, skl_window_size = 50)
    dataset = sf_processing()
