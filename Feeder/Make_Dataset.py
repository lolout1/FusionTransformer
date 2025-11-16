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

            # CRITICAL: Validate that acc and gyro have same length
            # This catches data preprocessing bugs early (before training)
            if self.acc_data.shape[0] != self.gyro_data.shape[0]:
                raise ValueError(
                    f"FATAL: Accelerometer and gyroscope data have mismatched lengths!\n"
                    f"  Accelerometer: {self.acc_data.shape[0]} samples\n"
                    f"  Gyroscope:     {self.gyro_data.shape[0]} samples\n"
                    f"  Difference:    {abs(self.acc_data.shape[0] - self.gyro_data.shape[0])} samples\n"
                    f"\n"
                    f"This indicates a data preprocessing error. Possible causes:\n"
                    f"  1. DTW alignment failed for some trials (check loader.py logs)\n"
                    f"  2. Old .npz file created with buggy code (regenerate dataset)\n"
                    f"  3. Trials concatenated before synchronization (check _synchronize_modalities)\n"
                    f"\n"
                    f"To fix:\n"
                    f"  - Enable debug mode: dataset_args.debug = True\n"
                    f"  - Check skip_stats in loader output\n"
                    f"  - Regenerate .npz files with fixed loader.py"
                )
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

        # Additional validation: Ensure labels match data length
        if len(self.labels) != self.num_samples:
            raise ValueError(
                f"Labels length ({len(self.labels)}) doesn't match data length ({self.num_samples})"
            )
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

        # Handle combined accelerometer + gyroscope/orientation data
        if self.inertial_modality == 'imu' and self.gyro_data is not None:
            gyro_or_orient = torch.tensor(self.gyro_data[index, :, :])

            # Check if this is raw gyro (3 ch) or orientation (3 ch from fusion)
            # Both are 3 channels, but we distinguish by adding SMV for orientation
            # Option 1: Raw gyro -> [ax, ay, az, gx, gy, gz] = 6 channels
            # Option 2: Orientation -> [SMV, ax, ay, az, roll, pitch, yaw] = 7 channels

            # For now, assume if gyro exists, try both paths:
            # Check if we should add engineered features (SMV)
            # If gyro_data represents orientation (from sensor fusion), add SMV
            # If gyro_data is raw gyro, concatenate directly

            # Simple heuristic: if gyro values are very large (>50), likely orientation angles
            # Otherwise, raw gyro (typically < 10 rad/s for wrist motion)
            max_val = gyro_or_orient.abs().max().item()

            if max_val > 50:  # Likely orientation in degrees
                # Sensor fusion output: add SMV to acc, then concat orientation
                watch_smv = self.cal_smv(acc_data)
                acc_with_smv = torch.cat((watch_smv, acc_data), dim=-1)  # 4 channels
                imu_data = torch.cat((acc_with_smv, gyro_or_orient), dim=-1)  # 7 channels
            else:
                # Raw gyroscope: concatenate acc + gyro
                imu_data = torch.cat((acc_data, gyro_or_orient), dim=-1)  # 6 channels

            data['accelerometer'] = imu_data  # Store as 'accelerometer' for backward compatibility

        else:
            # Single modality (accelerometer or gyroscope only)
            watch_smv = self.cal_smv(acc_data)
            watch_weight = self.calculate_weight(acc_data)
            # watch_roll = self.calculate_roll(acc_data)
            # watch_pitch = self.calculate_pitch(acc_data)
            acc_data = torch.cat((watch_smv, acc_data), dim=-1)  # 4 channels
            data['accelerometer'] = acc_data

        label = self.labels[index]
        label = torch.tensor(label)
        label = label.long()
        return data, label, index




if __name__ == "__main__":
    data = torch.randn((8, 128, 3))
    smv = cal_smv(data)
