import pandas as pd
import numpy as np
import math
import torch
import random
import torch.nn.functional as F
pd.options.mode.chained_assignment = None  # default='warn'
import scipy.stats as s

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
    def __init__(self, npz_file, batch_size, transform = None):
        # Load data and labels from npz file
        dataset = np.load(npz_file)
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
        self.transform = transform

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
        data['skl_data'] = skl_data
        data['acc_data'] = acc_data 
        label = self.labels[index]
        #data, label = self.transform(data, label)
        label = torch.tensor(label)
        label = label.long()
        return data, label

