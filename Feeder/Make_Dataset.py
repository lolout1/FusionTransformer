import pandas as pd
import numpy as np
import math
import torch
import random
import torch.nn.functional as F
import scipy.stats as s
from einops import rearrange
#from tsaug import TimeWarp, Reverse, Drift, AddNoise


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
    def __init__(self, dataset, batch_size, transform = None):
        # Load data and labels from npz file
        #dataset = np.load(npz_file)
        self.acc_data = dataset['acc_data']
        self.skl_data = dataset['skl_data']
        self.labels = dataset['labels']
        self.num_samples = self.acc_data.shape[0]
        self.acc_seq = self.acc_data.shape[1]
        # self.skl_joints = self.skl_data.shape[2]
        # self.skl_seq = self.skl_data.shape[1]
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Get the batch containing the requested index
        data = dict()
        skl_data = torch.tensor(self.skl_data[index, :, :, :])
        #skl_data = skl_data.reshape((l, -1, 3))
        acc_data = torch.tensor(self.acc_data[index, : , :])
        data['skl_data'] = skl_data
        data['acc_data'] =  acc_data
        label = self.labels[index]
        label = torch.tensor(label)
        label = label.long()
        return data, label, index
    

class UTD_mm(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size):
        # Support both single modality (acc OR gyro) and multi-modal (acc AND gyro)
        self.has_accelerometer = 'accelerometer' in dataset
        self.has_gyroscope = 'gyroscope' in dataset

        # Determine inertial modalities available
        if self.has_accelerometer and self.has_gyroscope:
            self.inertial_modality = 'imu'  # Combined accelerometer + gyroscope
            self.acc_data = dataset['accelerometer']
            self.gyro_data = dataset['gyroscope']
        elif self.has_accelerometer:
            self.inertial_modality = 'accelerometer'
            self.acc_data = dataset['accelerometer']
            self.gyro_data = None
        elif self.has_gyroscope:
            self.inertial_modality = 'gyroscope'
            self.acc_data = dataset['gyroscope']
            self.gyro_data = None
        else:
            raise ValueError("Dataset must contain at least 'accelerometer' or 'gyroscope' data")

        self.labels = dataset['labels']
        self.has_skeleton = 'skeleton' in dataset
        self.skl_data = dataset['skeleton'] if self.has_skeleton else None
        #self.skl_data = np.random.randn(self.acc_data.shape[0], 32,3)
        self.num_samples = self.acc_data.shape[0]
        self.acc_seq = self.acc_data.shape[1]
        if self.has_skeleton:
            self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
            self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, -1, 3)
        else:
            self.skl_seq = self.skl_length = self.skl_features = 0
        self.channels = self.acc_data.shape[2]
        self.batch_size = batch_size
        self.transform = None
        self.crop_size = 64

    
    def random_crop(self,data : torch.Tensor) -> torch.Tensor:
        '''
        Function to add random cropping to the data
        Arg: 
            data : 
        Output: 
            crop_data: will return croped data
        '''
        length = data.shape[0]
        start_idx = np.random.randint(0, length-self.crop_size-1)
        return data[start_idx : start_idx+self.crop_size, :]

    def cal_smv(self, sample : torch.Tensor) -> torch.Tensor:
        '''
        Function to calculate SMV
        '''
        mean = torch.mean(sample, dim = -2, keepdim=True)
        zero_mean = sample - mean
        sum_squared =  torch.sum(torch.square(zero_mean), dim=-1, keepdim=True) 
        smv= torch.sqrt(sum_squared)
        return smv
    
    def calculate_weight(self, sample):
        """
        Calculate the magnitude (weight) of accelerometer data.

        Parameters:
        - data: A PyTorch tensor of shape (128, 3) where each row is [ax, ay, az].

        Returns:
        - A 1D PyTorch tensor of shape (128,) containing the magnitude for each row.
        """
        mean = torch.mean(sample, dim = -2, keepdim=True)
        zero_mean = sample - mean
        return torch.sqrt(torch.sum(zero_mean**2, dim=-1, keepdim=True))
    
    def calculate_pitch(self,data):
        """
        Calculate the pitch from accelerometer data.

        Parameters:
        - data: A PyTorch tensor of shape (128, 3) where each row is [ax, ay, az].

        Returns:
        - A 1D PyTorch tensor of shape (128,) containing the pitch angle for each row in radians.
        """
        ax = data[:, 0]
        ay = data[:, 1]
        az = data[:, 2]
        return torch.atan2(-ax, torch.sqrt(ay**2 + az**2)).unsqueeze(1)
    

    def calculate_roll(self,data):
        """
        Calculate the roll from accelerometer data.

        Parameters:
        - data: A PyTorch tensor of shape (128, 3) where each row is [ax, ay, az].

        Returns:
        - A 1D PyTorch tensor of shape (128,) containing the roll angle for each row in radians.
        """
        ax = data[:, 0]
        ay = data[:, 1]
        az = data[:, 2]
        return torch.atan2(ay, az).unsqueeze(1)




    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        acc_data = torch.tensor(self.acc_data[index, : , :])
        data = dict()
        if self.has_skeleton:
            skl_data = torch.tensor(self.skl_data[index, :, :, :])
            data['skeleton'] = skl_data

        # Handle combined accelerometer + gyroscope data
        if self.inertial_modality == 'imu' and self.gyro_data is not None:
            gyro_data = torch.tensor(self.gyro_data[index, :, :])

            # Calculate features for accelerometer
            acc_smv = self.cal_smv(acc_data)
            # acc_weight = self.calculate_weight(acc_data)

            # Calculate features for gyroscope (magnitude)
            gyro_magnitude = torch.sqrt(torch.sum(gyro_data**2, dim=-1, keepdim=True))

            # Concatenate: [acc_smv, ax, ay, az, gx, gy, gz, gyro_mag] = 8 channels
            # For simpler model to avoid overfitting, use: [ax, ay, az, gx, gy, gz] = 6 channels
            imu_data = torch.cat((acc_data, gyro_data), dim=-1)
            data['accelerometer'] = imu_data  # Store as 'accelerometer' for backward compatibility
        else:
            # Single modality (accelerometer or gyroscope only)
            watch_smv = self.cal_smv(acc_data)
            watch_weight = self.calculate_weight(acc_data)
            # watch_roll = self.calculate_roll(acc_data)
            # watch_pitch = self.calculate_pitch(acc_data)
            acc_data = torch.cat((watch_smv, acc_data), dim=-1)
            data['accelerometer'] = acc_data

        label = self.labels[index]
        label = torch.tensor(label)
        label = label.long()
        return data, label, index




if __name__ == "__main__":
    data = torch.randn((8, 128, 3))
    smv = cal_smv(data)
