import pandas as pd
import numpy as np
import math
import torch
import random
import torch.nn.functional as F
import scipy.stats as s
from einops import rearrange
from tsaug import TimeWarp, Reverse, Drift, AddNoise


#################### MAIN #####################

#CREATE PYTORCH DATASET
'''
Input Args:
data = ncrc or ntu
num_frames = mocap and nturgb+d frame count!
acc_frames = frames from acc sensor per action
'''


class Utd_Dataset(torch.utils.data.Dataset):
    def __init__(self, npz_file):
        # Load data and labels from npz file
        dataset = np.load(npz_file)
        self.dataset = dataset['data']
        self.labels = dataset['labels']
        self.num_samples = self.dataset.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Get the batch containing the requested index
        data = self.dataset[index, :, : , :]
        data = torch.tensor(data)
        label = self.labels[index]
        label = label - 1
        label = torch.tensor(label)
        label = label.long()
        return data, label

class Berkley_mhad(torch.utils.data.Dataset):
    def __init__(self, npz_file):
        # Load data and labels from npz file
        dataset = np.load(npz_file)
        self.dataset = dataset['data']
        self.labels = dataset['labels']
        self.num_samples = self.dataset.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Get the batch containing the requested index
        data = self.dataset[index, :, :]
        data = torch.tensor(data)
        label = self.labels[index]
        label = label - 1
        label = torch.tensor(label)
        label = label.long()
        return data, label

class Bmhad_mm(torch.utils.data.Dataset):
    def __init__(self, npz_file, batch_size):
        # Load data and labels from npz file
        dataset = np.load(npz_file)
        self.acc_data = dataset['acc_data']
        self.skl_data = dataset['skl_data']
        self.labels = dataset['labels']
        self.num_samples = self.acc_data.shape[0]
        self.acc_seq = self.acc_data.shape[1]
        self.skl_joints = self.skl_data.shape[2]
        self.skl_seq = self.skl_data.shape[1]
        self.channels = self.skl_data.shape[3]
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Get the batch containing the requested index
        skl_data = torch.tensor(self.skl_data[index, :, :,:])
        acc_data = torch.tensor(self.acc_data[index, : , :])
        data = torch.zeros(self.skl_seq, self.skl_joints+self.acc_seq, self.channels)
        data[:, :self.skl_joints, :] = skl_data
        data[ 0, self.skl_joints:, :] = acc_data
        label = self.labels[index]
        label = label - 1
        label = torch.tensor(label)
        label = label.long()
        return data, label
    

class UTD_mm(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size, transform = None):
        # Load data and labels from npz file
        #dataset = np.load(npz_file)
        self.acc_data = dataset['acc_data']
        self.skl_data = dataset['skl_data']
        self.labels = dataset['labels']
        self.num_samples = self.acc_data.shape[0]
        self.acc_seq = self.acc_data.shape[1]
        self.skl_joints = self.skl_data.shape[2]
        self.skl_seq = self.skl_data.shape[1]
        self.skl_channels = self.skl_data.shape[3]
        self.channels = self.acc_data.shape[2]
        self.batch_size = batch_size
        self.transform = None
        # self.transform =  (
        #      AddNoise(scale=(0.01, 0.05)) @ 0.5  # random quantize to 10-, 20-, or 30- level sets
        #     + Drift(max_drift=(0.1, 0.5)) @ 0.4  # with 80% probability, random drift the signal up to 10% - 50%
        #     + Reverse() @ 0.5 ) # with 50% probability, reverse the sequence 

    def augment(self, data): 
        data = torch.transpose(data, -2, -1)
        transformed_data = self.transform.augment(data.numpy())
        data = torch.tensor(np.transpose(transformed_data))
        return data 
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Get the batch containing the requested index
        skl_data = torch.tensor(self.skl_data[index, :, :,:])
        acc_data = torch.tensor(self.acc_data[index, : , :])
        # data = torch.zeros(self.skl_seq, self.skl_joints+self.acc_seq, self.channels)
        # data[:, :self.skl_joints, :self.skl_channels] = skl_data
        # data[ 0, self.skl_joints:, :self.channels] = acc_data
        data = dict()


        if self.transform and np.random.random() > 0.6:
            T, N , C = skl_data.shape
            skl_data = rearrange(skl_data, 'T N C -> T (N C)')
            aug_skl = self.augment(skl_data)
            aug_skl = rearrange(aug_skl, 'T (N C) -> T N C', N = N , C = C)
            skl_data = aug_skl
            acc_data = self.augment(acc_data)
                
        data['acc_data'] = acc_data    
        data['skl_data'] = skl_data
        label = self.labels[index]
        #data, label = self.transform(data, label)
        label = torch.tensor(label)
        label = label.long()
        return data, label

