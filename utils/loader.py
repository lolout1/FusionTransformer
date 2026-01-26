'''
Dataset Builder

IMPORTANT - Subjects with poor gyroscope data quality (use for TRAINING ONLY):
  [29, 32, 35, 39] - These subjects have corrupt gyroscope timestamps that
  cause all their fall trials to be discarded after timestamp alignment.
  They should NOT be used for validation or testing with IMU (acc+gyro) models.
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
from utils.alignment import (
    AlignmentConfig, AlignmentResult,
    create_alignment_config_from_kwargs, align_imu_modalities
)
from utils.kalman import kalman_fusion_for_loader, assemble_kalman_features
from utils.kalman_smoothing import kalman_smoothing_for_loader



def csvloader(file_path: str, **kwargs):
    '''
    Loads csv data. Raises ValueError for malformed files to enable skip logic.
    '''
    try:
        file_data = pd.read_csv(file_path, index_col=False, header=0).dropna().bfill()
    except Exception as e:
        raise ValueError(f"CSV parse error: {e}")

    if 'skeleton' in file_path:
        cols = 96
    else:
        cols = 3

    try:
        # Select last `cols` columns starting from row 2
        activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
    except ValueError as e:
        # Handle non-numeric data (e.g., UUIDs in data columns)
        raise ValueError(f"Non-numeric data in CSV: {e}")

    if activity_data.size == 0:
        raise ValueError("Empty data after parsing")

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
                   reference_key: str = 'skeleton',
                   class_aware_stride: bool = False,
                   fall_stride: int = 32,
                   adl_stride: int = 10) -> Dict[str, np.ndarray]:
    '''
    Sliding Window with optional class-aware stride.

    Args:
        data: Dictionary containing modality data
        clearing_time_index: Start index for windowing
        max_time: Maximum time to consider
        sub_window_size: Window size (e.g., 128)
        stride_size: Default stride size (used if class_aware_stride=False)
        label: Class label (0=ADL, 1=Fall)
        reference_key: Modality to use as reference for length
        class_aware_stride: If True, use different strides for falls vs ADLs
        fall_stride: Stride for fall class (default 32, ~75% overlap)
        adl_stride: Stride for ADL class (default 10, ~92% overlap, generates 3x more samples)

    Returns:
        Dictionary with windowed data
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

    # Class-aware stride: Use smaller stride for ADLs to generate more samples
    if class_aware_stride:
        if label == 0:  # ADL class
            effective_stride = adl_stride
        else:  # Fall class (label == 1)
            effective_stride = fall_stride
    else:
        effective_stride = stride_size

    sub_windows  = (
        start +
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(np.arange(max_time, step = effective_stride), 0).T
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


def align_gyro_to_acc(data: Dict[str, np.ndarray], use_fast_dtw: bool = True, max_length_diff: int = 10) -> Dict[str, np.ndarray]:
    """
    Align gyroscope data to accelerometer data using Dynamic Time Warping (DTW).
    Treats accelerometer as the ground truth reference signal.

    This function handles potential temporal misalignment between accelerometer and
    gyroscope sensors, which can occur due to different sampling rates, timing jitter,
    or sensor fusion issues.

    Performance:
        - Uses dtaidistance (C-optimized) by default for maximum speed
        - Falls back to fastdtw for compatibility
        - Optimized for multi-core CPU environments (A100 servers)
        - Typical per-trial processing: ~10-50ms depending on sequence length

    Args:
        data: Dictionary containing 'accelerometer' and 'gyroscope' modalities
              Expected shapes: (num_samples, 3) for both modalities
        use_fast_dtw: If True, uses fastdtw (approximation, faster for long sequences)
                      If False, uses dtaidistance (exact, C-optimized, better for <500 samples)
        max_length_diff: Maximum allowed length difference between acc and gyro (default: 10)
                        If difference > max_length_diff, raises ValueError to skip trial

    Returns:
        data: Dictionary with gyroscope aligned to accelerometer
              Both modalities will have the same length after alignment

    Raises:
        ValueError: If length difference exceeds max_length_diff (indicates data quality issue)

    Note:
        - Only applied when both 'accelerometer' and 'gyroscope' are present
        - Uses magnitude (L2 norm) of 3D signals as reference for DTW
        - Removes duplicate indices after alignment to ensure clean temporal mapping
        - Accelerometer data remains unchanged (ground truth)
        - DTW works best for small temporal misalignments, not large data gaps
    """
    # Only apply DTW if both modalities are present
    if 'accelerometer' not in data or 'gyroscope' not in data:
        return data

    acc_data = data['accelerometer']
    gyro_data = data['gyroscope']

    # Validate length difference - DTW is for temporal alignment, not fixing missing data
    length_diff = abs(len(acc_data) - len(gyro_data))
    if length_diff > max_length_diff:
        raise ValueError(
            f"Length mismatch too large for DTW alignment: "
            f"acc={len(acc_data)}, gyro={len(gyro_data)}, diff={length_diff} > {max_length_diff}. "
            f"This likely indicates a data quality issue rather than temporal misalignment."
        )

    # Calculate magnitude (L2 norm) of 3D signals for DTW reference
    # Using magnitude makes DTW robust to individual axis variations
    acc_magnitude = norm(acc_data, axis=1)
    gyro_magnitude = norm(gyro_data, axis=1)

    # Perform DTW alignment: align gyroscope to accelerometer
    # Choose algorithm based on sequence length and accuracy requirements
    if use_fast_dtw or len(acc_magnitude) > 500:
        # FastDTW: O(N) approximation, good for long sequences
        distance, path = fastdtw(
            gyro_magnitude[:, np.newaxis],   # Query: gyroscope (to be aligned)
            acc_magnitude[:, np.newaxis],    # Reference: accelerometer (truth)
            dist=euclidean
        )
    else:
        # dtaidistance: Exact DTW with C optimization, better for shorter sequences
        # Note: Returns path in different format than fastdtw
        path_indices = dtw.warping_path(
            gyro_magnitude.flatten(),
            acc_magnitude.flatten()
        )
        path = path_indices

    # Extract aligned indices and remove duplicates
    # gyro_ids: indices of gyroscope samples that align to accelerometer
    # acc_ids: indices of accelerometer samples (reference timeline)
    gyro_ids, acc_ids = filter_repeated_ids(path)

    # Apply alignment: keep only the aligned samples
    # Accelerometer is trimmed to matched indices (but maintains its temporal ordering)
    # Gyroscope is resampled according to the DTW path
    data['accelerometer'] = filter_data_by_ids(acc_data, list(acc_ids))
    data['gyroscope'] = filter_data_by_ids(gyro_data, list(gyro_ids))

    return data


def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    """
    Apply Butterworth filter with padlen safety check.

    Handles short sequences gracefully by returning original data
    if sequence is too short for the filter.
    """
    # Minimum length for filtfilt: 3 * max(len(a), len(b)) = 3 * (order + 1)
    min_length = 3 * (order + 1)
    if len(data) < min_length:
        return data  # Too short to filter, return unchanged

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    # Ensure cutoff is valid (0 < normal_cutoff < 1)
    if normal_cutoff <= 0 or normal_cutoff >= 1:
        return data

    try:
        b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
        return filtfilt(b, a, data, axis=0)
    except ValueError:
        # Fallback if filter fails (e.g., padlen issues)
        return data

def convert_gyro_to_radians(data: np.ndarray) -> np.ndarray:
    """
    Convert gyroscope data from deg/s to rad/s.

    Args:
        data: Gyroscope data in deg/s (N, 3) or (N, 4) with magnitude

    Returns:
        Gyroscope data in rad/s
    """
    return data * (np.pi / 180.0) 

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
        # Add configurable preprocessing flags
        # Filtering (research-based for Android watch IMU data)
        # Default: DISABLED (enable_filtering=False)
        # Accelerometer: LOW-pass 5.5 Hz (removes high-freq noise)
        # Gyroscope: HIGH-pass 0.5 Hz (removes low-freq drift)
        self.enable_filtering = kwargs.get('enable_filtering', False)  # Default: OFF
        self.enable_normalization = kwargs.get('enable_normalization', True)
        # Selective normalization: normalize only specific modalities
        # Options: 'all' (default), 'gyro_only', 'acc_only', 'none'
        # When enable_normalization=True and normalize_modalities='gyro_only',
        # only gyroscope data is normalized (robust gyro normalization)
        self.normalize_modalities = kwargs.get('normalize_modalities', 'all')

        # Separate filter parameters for accelerometer vs gyroscope
        # Accelerometer: Low-pass filter (human motion < 5 Hz)
        self.acc_filter_cutoff = kwargs.get('acc_filter_cutoff', 5.5)    # Hz (low-pass)
        self.acc_filter_type = kwargs.get('acc_filter_type', 'low')      # Low-pass

        # Gyroscope: High-pass filter (removes drift, preserves rotation)
        self.gyro_filter_cutoff = kwargs.get('gyro_filter_cutoff', 0.5)  # Hz (high-pass)
        self.gyro_filter_type = kwargs.get('gyro_filter_type', 'high')   # High-pass

        # Gravity removal for accelerometer (converts raw acc to linear acceleration)
        # Uses high-pass filter to remove DC component (gravity ~0 Hz)
        # Cutoff 0.3-0.5 Hz recommended (gravity is static, motion > 0.5 Hz)
        self.remove_gravity = kwargs.get('remove_gravity', False)  # Default: OFF
        self.gravity_filter_cutoff = kwargs.get('gravity_filter_cutoff', 0.3)  # Hz

        # Common sampling rate
        self.filter_fs = kwargs.get('filter_fs', 25)                     # Sampling rate (Hz)
        self.filter_order = kwargs.get('filter_order', 4)                # 4th-order Butterworth

        # Backward compatibility: if 'filter_cutoff' is provided, use for acc only
        if 'filter_cutoff' in kwargs:
            self.acc_filter_cutoff = kwargs['filter_cutoff']
        self.use_skeleton = kwargs.get('use_skeleton', True)
        self.align_with_skeleton = kwargs.get('enable_skeleton_alignment', self.use_skeleton)
        # DTW alignment for gyroscope to accelerometer (per-trial, pre-windowing)
        # DEPRECATED: Use enable_timestamp_alignment instead
        self.enable_gyro_alignment = kwargs.get('enable_gyro_alignment', False)

        # Timestamp-based alignment (replaces DTW)
        # Parses ISO timestamps from CSVs, finds overlap region, interpolates to common grid
        self.enable_timestamp_alignment = kwargs.get('enable_timestamp_alignment', False)
        self.alignment_target_rate = kwargs.get('alignment_target_rate', 30.0)
        self.alignment_method = kwargs.get('alignment_method', 'linear')
        self.min_overlap_ratio = kwargs.get('min_overlap_ratio', 0.8)
        self.max_time_gap_ms = kwargs.get('max_time_gap_ms', 1000.0)
        self.min_output_samples = kwargs.get('min_output_samples', 64)
        self.length_threshold = kwargs.get('length_threshold', 10)

        # Simple truncation mode (RECOMMENDED for acc+gyro)
        # Instead of complex alignment, just truncate both modalities to min length
        # Much simpler and more robust than DTW or timestamp interpolation
        self.enable_simple_truncation = kwargs.get('enable_simple_truncation', False)
        self.max_truncation_diff = kwargs.get('max_truncation_diff', 50)  # Max samples to truncate
        # Conservative interpolation controls (prevent interpolation when timestamps drifted)
        self.max_duration_ratio = kwargs.get('max_duration_ratio', 1.2)  # Default: 20% duration diff max
        self.max_rate_divergence = kwargs.get('max_rate_divergence', 0.3)  # Default: 30% rate diff max

        # Class-aware stride configuration (helps balance ADL vs Fall samples)
        self.enable_class_aware_stride = kwargs.get('enable_class_aware_stride', False)
        self.fall_stride = kwargs.get('fall_stride', 32)  # Default: 75% overlap for falls
        self.adl_stride = kwargs.get('adl_stride', 10)    # Default: 92% overlap for ADLs (3x more samples)

        # Track modality validation per subject
        self.subject_modality_stats = defaultdict(lambda: {
            'accelerometer': 0, 'gyroscope': 0, 'skeleton': 0,
            'total_trials': 0, 'valid_trials': 0, 'fall_trials': 0, 'adl_trials': 0,
            'skipped_missing_modality': 0,
            'skipped_length_mismatch': 0,
            'skipped_too_short': 0,
            'skipped_preprocessing_error': 0,
            'skipped_file_load_error': 0,
            'skipped_dtw_length_mismatch': 0,
            'skipped_poor_gyro_hard': 0,
            'skipped_poor_gyro_adaptive': 0,
            'skipped_kalman_fusion': 0,
            # Timestamp alignment tracking
            'timestamp_aligned': 0,
            'timestamp_use_as_is': 0,
            'skipped_timestamp_unsync': 0,
            'skipped_insufficient_overlap': 0,
            'skipped_duration_drift': 0,  # NEW: Discarded due to duration ratio exceeded
            'skipped_rate_drift': 0,      # NEW: Discarded due to sampling rate divergence
            # Simple truncation tracking
            'simple_truncation_applied': 0,
            'skipped_truncation_too_large': 0,
            # Class-level tracking
            'file_count': 0,
            'window_count': 0,
            'fall_windows': 0,
            'adl_windows': 0,
            # Motion filtering tracking
            'motion_total_windows': 0,
            'motion_passed_windows': 0,
            'motion_rejected_windows': 0
        })
        self.required_modalities = kwargs.get('required_modalities', [])
        self.discard_mismatched_modalities = kwargs.get('discard_mismatched_modalities', False)
        self.length_sensitive_modalities = kwargs.get(
            'length_sensitive_modalities',
            ['accelerometer', 'gyroscope']
        )
        if not self.length_sensitive_modalities:
            self.length_sensitive_modalities = ['accelerometer', 'gyroscope']

        # Quality filtering configuration
        self.enable_gyro_quality_check = kwargs.get('enable_gyro_quality_check', False)
        self.quality_mode = kwargs.get('quality_mode', 'none')  # 'none', 'hard', 'adaptive'
        self.quality_threshold_snr = kwargs.get('quality_threshold_snr', 1.0)
        self.quality_method = kwargs.get('quality_method', 'simple')

        # Window filtering configuration
        # min_windows_per_trial: 1 means >= 1 (include single-window trials)
        #                        2 means > 1 (exclude single-window trials)
        self.min_windows_per_trial = kwargs.get('min_windows_per_trial', 1)

        # Motion filtering configuration (matches Android app)
        self.enable_motion_filtering = kwargs.get('enable_motion_filtering', False)
        self.motion_threshold = kwargs.get('motion_threshold', 10.0)
        self.motion_min_axes = kwargs.get('motion_min_axes', 2)

        # Sensor fusion configuration
        self.enable_sensor_fusion = kwargs.get('enable_sensor_fusion', False)
        self.fusion_method = kwargs.get('fusion_method', 'madgwick')
        self.fusion_frequency = kwargs.get('fusion_frequency', 30.0)
        self.fusion_params = kwargs.get('fusion_params', {})

        # Kalman filter fusion configuration
        self.enable_kalman_fusion = kwargs.get('enable_kalman_fusion', False)
        self.kalman_config = {
            'kalman_filter_type': kwargs.get('kalman_filter_type', 'linear'),
            'kalman_output_format': kwargs.get('kalman_output_format', 'euler'),
            'kalman_include_smv': kwargs.get('kalman_include_smv', True),
            'kalman_include_uncertainty': kwargs.get('kalman_include_uncertainty', False),
            'kalman_include_innovation': kwargs.get('kalman_include_innovation', False),
            'filter_fs': kwargs.get('filter_fs', 30.0),
            # Linear KF parameters
            'kalman_Q_orientation': kwargs.get('kalman_Q_orientation', 0.01),
            'kalman_Q_rate': kwargs.get('kalman_Q_rate', 0.1),
            'kalman_R_acc': kwargs.get('kalman_R_acc', 0.1),
            'kalman_R_gyro': kwargs.get('kalman_R_gyro', 0.5),
            # EKF parameters
            'kalman_Q_quat': kwargs.get('kalman_Q_quat', 0.001),
            'kalman_Q_bias': kwargs.get('kalman_Q_bias', 0.0001),
        }

        # Kalman SMOOTHING configuration (per-channel denoising, NOT fusion)
        # This is different from kalman_fusion which combines acc+gyro → orientation
        # Kalman smoothing applies 1D Kalman filter to each channel for noise reduction
        self.enable_kalman_smoothing = kwargs.get('enable_kalman_smoothing', False)
        self.kalman_smooth_config = {
            'kalman_smooth_fs': kwargs.get('kalman_smooth_fs', 30.0),
            'kalman_smooth_Q_acc': kwargs.get('kalman_smooth_Q_acc', 0.01),
            'kalman_smooth_R_acc': kwargs.get('kalman_smooth_R_acc', 0.05),
            'kalman_smooth_Q_gyro': kwargs.get('kalman_smooth_Q_gyro', 0.05),
            'kalman_smooth_R_gyro': kwargs.get('kalman_smooth_R_gyro', 0.1),
            'kalman_smooth_bidirectional': kwargs.get('kalman_smooth_bidirectional', False),
        }

        # Global skip tracking
        self.skip_stats = {
            # Config parameters for logging
            'min_windows_per_trial': self.min_windows_per_trial,
            # Trial counts
            'total_trials': 0,
            'valid_trials': 0,
            'fall_trials': 0,
            'adl_trials': 0,
            'skipped_missing_modality': 0,
            'skipped_length_mismatch': 0,
            'skipped_too_short': 0,
            'skipped_preprocessing_error': 0,
            'skipped_file_load_error': 0,
            'skipped_dtw_length_mismatch': 0,
            'skipped_poor_gyro_hard': 0,
            'skipped_poor_gyro_adaptive': 0,
            'skipped_kalman_fusion': 0,
            # Timestamp alignment statistics
            'timestamp_aligned': 0,
            'timestamp_use_as_is': 0,
            'skipped_timestamp_unsync': 0,
            'skipped_insufficient_overlap': 0,
            'skipped_duration_drift': 0,  # NEW: Discarded due to duration ratio exceeded
            'skipped_rate_drift': 0,      # NEW: Discarded due to sampling rate divergence
            # Simple truncation statistics
            'simple_truncation_applied': 0,
            'skipped_truncation_too_large': 0,
            # Motion filtering statistics
            'motion_total_windows': 0,
            'motion_passed_windows': 0,
            'motion_rejected_windows': 0,
            'motion_rejection_rate': 0.0
        }

        # Track preprocessing error details (for clean summary output)
        self.preprocessing_error_details = Counter()

        # Debug mode control
        self.debug = kwargs.get('debug', False)

        # Gyroscope unit conversion (deg/s -> rad/s)
        self.convert_gyro_to_rad = kwargs.get('convert_gyro_to_rad', False)

        # Skip file logging
        self.log_skipped_files = kwargs.get('log_skipped_files', False)
        self.skipped_files = []

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

    def _validate_required_modalities(self, trial_data: Dict[str, np.ndarray], subject_id: str, trial_id: str) -> bool:
        """
        Validate that all required modalities are present in trial_data.
        Returns True if valid, False if missing required modalities.
        """
        if not self.required_modalities:
            return True  # No validation required

        missing_modalities = []
        for modality in self.required_modalities:
            if modality not in trial_data or trial_data[modality] is None:
                missing_modalities.append(modality)

        if missing_modalities:
            # Track the skip
            self.skip_stats['skipped_missing_modality'] += 1
            self.subject_modality_stats[subject_id]['skipped_missing_modality'] += 1
            return False

        return True

    def _check_length_mismatch(self, trial_data: Dict[str, np.ndarray]) -> Tuple[bool, Dict[str, int]]:
        """
        Determine whether the configured modalities share the same sample count.

        Returns:
            (has_mismatch, modality_lengths)
        """
        if not self.discard_mismatched_modalities:
            return False, {}

        tracked_modalities = {}
        for modality in self.length_sensitive_modalities:
            if modality in trial_data and isinstance(trial_data[modality], np.ndarray):
                tracked_modalities[modality] = trial_data[modality].shape[0]

        # Require at least two modalities to compare lengths
        if len(tracked_modalities) < 2:
            return False, tracked_modalities

        unique_lengths = set(tracked_modalities.values())
        return len(unique_lengths) > 1, tracked_modalities

    def _synchronize_modalities(
        self,
        trial_data: Dict[str, np.ndarray],
        subject_id: str = None,
        action_id: str = None,
        trial_id: str = None
    ) -> int:
        """Trim all modalities to the minimum available length while optionally enforcing equality."""
        has_mismatch, tracked_lengths = self._check_length_mismatch(trial_data)
        if has_mismatch:
            # Track the skip
            if subject_id:
                self.subject_modality_stats[subject_id]['skipped_length_mismatch'] += 1
            self.skip_stats['skipped_length_mismatch'] += 1

            context_tokens = []
            if subject_id is not None:
                context_tokens.append(f"subject {subject_id}")
            if action_id is not None:
                context_tokens.append(f"action {action_id}")
            if trial_id is not None:
                context_tokens.append(f"trial {trial_id}")
            context = ', '.join(context_tokens) if context_tokens else 'Trial'
            length_text = ', '.join(f'{k}={v}' for k, v in tracked_lengths.items())
            raise ValueError(f'{context}: mismatched sample counts across modalities ({length_text})')

        lengths = [value.shape[0] for key, value in trial_data.items() if key != 'labels']
        if not lengths:
            raise ValueError('Trial data contains no modalities to process')
        min_length = min(lengths)
        if min_length < self.max_length:
            # Track the skip
            if subject_id:
                self.subject_modality_stats[subject_id]['skipped_too_short'] += 1
            self.skip_stats['skipped_too_short'] += 1
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
    
    def _maybe_filter(self, modality: str, data: np.ndarray) -> np.ndarray:
        """
        Optionally apply Butterworth filtering to inertial signals.

        Research-based filtering for Android watch IMU data:
        - Accelerometer: LOW-pass 5.5 Hz (preserves human motion < 5 Hz, removes high-freq noise)
        - Gyroscope: HIGH-pass 0.5 Hz (removes low-freq drift, preserves rotation)
        - 4th-order Butterworth (standard in fall detection literature)

        Filter rationale:
        - Accelerometer measures linear acceleration → Low-pass removes sensor noise
        - Gyroscope measures angular velocity → High-pass removes drift (integration error)

        References:
        - Smartphone fall detection: 5 Hz low-pass for accelerometer
        - IMU preprocessing: High-pass for gyroscope drift removal
        """
        # Gravity removal (independent of enable_filtering)
        if modality == 'accelerometer' and self.remove_gravity:
            # High-pass filter to remove gravity (DC component)
            # Converts raw accelerometer to linear acceleration
            data = butterworth_filter(
                data,
                cutoff=self.gravity_filter_cutoff,
                fs=self.filter_fs,
                order=self.filter_order,
                filter_type='high'  # High-pass removes DC (gravity)
            )

        if not self.enable_filtering:
            return data

        if modality == 'accelerometer':
            # Accelerometer: LOW-pass filter (removes high-frequency noise)
            # Preserves human motion frequencies (0-5 Hz)
            # Note: If remove_gravity is enabled, this applies AFTER gravity removal
            return butterworth_filter(
                data,
                cutoff=self.acc_filter_cutoff,
                fs=self.filter_fs,
                order=self.filter_order,
                filter_type=self.acc_filter_type  # 'low'
            )
        elif modality == 'gyroscope':
            # Gyroscope: HIGH-pass filter (removes low-frequency drift)
            # Preserves rotational motion (> 0.5 Hz)
            return butterworth_filter(
                data,
                cutoff=self.gyro_filter_cutoff,
                fs=self.filter_fs,
                order=self.filter_order,
                filter_type=self.gyro_filter_type  # 'high'
            )
        else:
            # Other modalities: no filtering
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
                32,  # Default stride (used if class_aware_stride=False)
                label,
                reference_key=reference_key,
                class_aware_stride=self.enable_class_aware_stride,
                fall_stride=self.fall_stride,
                adl_stride=self.adl_stride
            )

        # Motion filtering (match Android app behavior)
        if self.enable_motion_filtering:
            from utils.preprocessing import filter_windows_by_motion, compute_motion_statistics

            # Track motion statistics BEFORE filtering
            windows_before = len(data.get('labels', [])) if 'labels' in data else len(data[reference_key])

            # Compute detailed motion statistics
            motion_stats = compute_motion_statistics(
                data,
                reference_key=reference_key if self.mode != 'avg_pool' else 'accelerometer',
                threshold=self.motion_threshold,
                min_axes=self.motion_min_axes
            )

            # Apply motion filtering
            data = filter_windows_by_motion(
                data,
                reference_key=reference_key if self.mode != 'avg_pool' else 'accelerometer',
                threshold=self.motion_threshold,
                min_axes=self.motion_min_axes
            )

            if data is None:
                raise ValueError("No windows passed motion threshold")

            # Track motion statistics in global stats
            self.skip_stats['motion_total_windows'] += motion_stats['total_windows']
            self.skip_stats['motion_passed_windows'] += motion_stats['active_windows']
            self.skip_stats['motion_rejected_windows'] += motion_stats['quiet_windows']

            # Return motion statistics for per-trial tracking
            return data, motion_stats

        return data, None

    def _add_trial_data(self, trial_data):

        for modality, modality_data in trial_data.items():
            self.data[modality].append(modality_data)
    
    def _len_check(self, d):
        """Check if all modalities have minimum required windows."""
        return all(len(v) >= self.min_windows_per_trial for v in d.values())

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
                # Track total trials
                subject_id = str(trial.subject_id)
                action_id = str(trial.action_id)
                self.skip_stats['total_trials'] += 1
                self.subject_modality_stats[subject_id]['total_trials'] += 1

                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                    # Track fall vs ADL trials
                    if label == 1:
                        self.skip_stats['fall_trials'] += 1
                        self.subject_modality_stats[subject_id]['fall_trials'] += 1
                    else:
                        self.skip_stats['adl_trials'] += 1
                        self.subject_modality_stats[subject_id]['adl_trials'] += 1
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
                        unimodal_data = self._maybe_filter(modality, unimodal_data)

                        # Convert gyroscope from deg/s to rad/s if enabled
                        if modality == 'gyroscope' and self.convert_gyro_to_rad:
                            unimodal_data = convert_gyro_to_radians(unimodal_data)

                        trial_data[modality] = unimodal_data
                        if modality in ['accelerometer', 'gyroscope'] and unimodal_data.shape[0]>250:
                            trial_data[modality] = self.select_subwindow_pandas(unimodal_data)                            
                        # if modality == 'skeleton':
                        #     print(unimodal_data.shape)

                    except Exception as e:
                        executed = False
                        # Track file loading errors silently (like other preprocessing errors)
                        error_msg = f"File load error: {str(e)[:100]}"
                        self.preprocessing_error_details[error_msg] += 1
                        self.skip_stats['skipped_file_load_error'] = self.skip_stats.get('skipped_file_load_error', 0) + 1
                        self.subject_modality_stats[subject_id]['skipped_file_load_error'] = \
                            self.subject_modality_stats[subject_id].get('skipped_file_load_error', 0) + 1
                        if self.debug:
                            print(f"Skipping S{subject_id}A{action_id} modality={modality}: {e}")

                if executed :
                    # Track which modalities were loaded for this subject
                    trial_sequence = getattr(trial, 'sequence_number', getattr(trial, 'trial_id', 'unknown'))
                    trial_id = str(trial_sequence)
                    for modality in trial_data.keys():
                        if modality in self.subject_modality_stats[subject_id]:
                            self.subject_modality_stats[subject_id][modality] += 1

                    # Validate required modalities are present
                    if not self._validate_required_modalities(trial_data, subject_id, trial_id):
                        # Log skipped file
                        if self.log_skipped_files:
                            file_paths = [f for f in trial.files.values()]
                            self.skipped_files.append({
                                'subject': subject_id,
                                'action': action_id,
                                'trial': trial_id,
                                'reason': 'missing_required_modality',
                                'files': file_paths
                            })
                        continue  # Skip this trial if validation fails

                    if self.align_with_skeleton and 'skeleton' in trial_data:
                        trial_data = align_sequence(trial_data)
                        # os.remove(file_path)

                    # Per-trial alignment and length validation
                    # Four modes (in order of precedence):
                    # 1. enable_simple_truncation = True: Just truncate to min length (RECOMMENDED)
                    # 2. enable_timestamp_alignment = True: Use timestamp-based interpolation
                    # 3. enable_gyro_alignment = True: Use DTW to align gyro to acc if diff < 10 (DEPRECATED)
                    # 4. None enabled: Skip trial if acc/gyro lengths don't match exactly
                    if 'accelerometer' in trial_data and 'gyroscope' in trial_data:
                        acc_len = trial_data['accelerometer'].shape[0]
                        gyro_len = trial_data['gyroscope'].shape[0]
                        length_diff = abs(acc_len - gyro_len)

                        if self.enable_simple_truncation:
                            # Mode 1: Simple truncation (RECOMMENDED)
                            # Just truncate both modalities to the shorter length
                            # Much simpler and more robust than DTW or timestamp interpolation
                            if length_diff <= self.max_truncation_diff:
                                if length_diff > 0:
                                    min_len = min(acc_len, gyro_len)
                                    trial_data['accelerometer'] = trial_data['accelerometer'][:min_len]
                                    trial_data['gyroscope'] = trial_data['gyroscope'][:min_len]

                                    self.skip_stats['simple_truncation_applied'] += 1
                                    self.subject_modality_stats[subject_id]['simple_truncation_applied'] += 1

                                    if self.debug:
                                        print(f"S{subject_id}A{action_id}T{trial_id}: Simple truncation "
                                              f"(acc={acc_len}, gyro={gyro_len}) → {min_len}")
                                # else: lengths already match, no truncation needed
                            else:
                                # Difference too large - indicates data quality issue
                                if self.debug:
                                    print(f"Skipping S{subject_id}A{action_id}T{trial_id}: "
                                          f"Length diff too large for truncation (acc={acc_len}, gyro={gyro_len}, "
                                          f"diff={length_diff} > {self.max_truncation_diff})")
                                self.skip_stats['skipped_truncation_too_large'] += 1
                                self.subject_modality_stats[subject_id]['skipped_truncation_too_large'] += 1

                                if self.log_skipped_files:
                                    file_paths = [f for f in trial.files.values()]
                                    self.skipped_files.append({
                                        'subject': subject_id,
                                        'action': action_id,
                                        'trial': trial_id,
                                        'reason': 'truncation_diff_too_large',
                                        'acc_len': acc_len,
                                        'gyro_len': gyro_len,
                                        'diff': length_diff,
                                        'max_allowed': self.max_truncation_diff,
                                        'files': file_paths
                                    })
                                continue

                        elif self.enable_timestamp_alignment:
                            # Mode 1: Timestamp-based alignment (RECOMMENDED)
                            # Uses ISO timestamps from CSVs to create uniform time grid
                            acc_path = trial.files.get('accelerometer')
                            gyro_path = trial.files.get('gyroscope')

                            if not acc_path or not gyro_path:
                                if self.debug:
                                    print(f"Skipping S{subject_id}A{action_id}T{trial_id}: "
                                          f"Missing file paths for timestamp alignment")
                                self.skip_stats['skipped_missing_modality'] += 1
                                self.subject_modality_stats[subject_id]['skipped_missing_modality'] += 1
                                continue

                            # Create alignment config with conservative interpolation controls
                            align_config = AlignmentConfig(
                                target_rate=self.alignment_target_rate,
                                alignment_method=self.alignment_method,
                                min_overlap_ratio=self.min_overlap_ratio,
                                max_time_gap_ms=self.max_time_gap_ms,
                                min_output_samples=self.min_output_samples,
                                length_threshold=self.length_threshold,
                                # Conservative: only interpolate if timestamps are truly synchronized
                                max_duration_ratio=self.max_duration_ratio,
                                max_rate_divergence=self.max_rate_divergence
                            )

                            # Perform timestamp-based alignment
                            align_result = align_imu_modalities(acc_path, gyro_path, align_config)

                            if not align_result.success:
                                # Alignment failed - track reason and skip
                                reason = align_result.reason.lower()
                                if 'duration ratio' in reason or 'duration_ratio' in reason:
                                    # NEW: Discarded due to timestamp drift (duration mismatch)
                                    self.skip_stats['skipped_duration_drift'] += 1
                                    self.subject_modality_stats[subject_id]['skipped_duration_drift'] += 1
                                elif 'rate divergence' in reason or 'sampling rate' in reason:
                                    # NEW: Discarded due to sampling rate divergence
                                    self.skip_stats['skipped_rate_drift'] += 1
                                    self.subject_modality_stats[subject_id]['skipped_rate_drift'] += 1
                                elif 'offset' in reason or 'unsync' in reason:
                                    self.skip_stats['skipped_timestamp_unsync'] += 1
                                    self.subject_modality_stats[subject_id]['skipped_timestamp_unsync'] += 1
                                elif 'overlap' in reason:
                                    self.skip_stats['skipped_insufficient_overlap'] += 1
                                    self.subject_modality_stats[subject_id]['skipped_insufficient_overlap'] += 1
                                else:
                                    self.skip_stats['skipped_length_mismatch'] += 1
                                    self.subject_modality_stats[subject_id]['skipped_length_mismatch'] += 1

                                if self.debug:
                                    print(f"Skipping S{subject_id}A{action_id}T{trial_id}: "
                                          f"Timestamp alignment failed - {align_result.reason}")

                                # Log skipped file
                                if self.log_skipped_files:
                                    self.skipped_files.append({
                                        'subject': subject_id,
                                        'action': action_id,
                                        'trial': trial_id,
                                        'reason': f'timestamp_alignment_{align_result.action}',
                                        'details': align_result.reason,
                                        'acc_len': acc_len,
                                        'gyro_len': gyro_len,
                                        'files': [acc_path, gyro_path]
                                    })
                                continue

                            # Update trial_data with aligned arrays
                            trial_data['accelerometer'] = align_result.aligned_acc
                            trial_data['gyroscope'] = align_result.aligned_gyro

                            # Track alignment action
                            if align_result.action == 'use_as_is':
                                self.skip_stats['timestamp_use_as_is'] += 1
                                self.subject_modality_stats[subject_id]['timestamp_use_as_is'] += 1
                            elif align_result.action == 'aligned':
                                self.skip_stats['timestamp_aligned'] += 1
                                self.subject_modality_stats[subject_id]['timestamp_aligned'] += 1

                            if self.debug:
                                print(f"S{subject_id}A{action_id}T{trial_id}: {align_result.action} "
                                      f"(acc={acc_len}→{align_result.output_samples}, "
                                      f"gyro={gyro_len}→{align_result.output_samples}, "
                                      f"rate={align_result.output_rate_hz:.0f}Hz)")

                        elif not self.enable_gyro_alignment:
                            # Mode 2: No alignment - require exact length match (STRICT)
                            if length_diff > 0:
                                if self.debug:
                                    print(f"Skipping S{subject_id}A{action_id}T{trial_id}: "
                                          f"Length mismatch without alignment (acc={acc_len}, gyro={gyro_len}, diff={length_diff})")
                                # Log skipped file
                                if self.log_skipped_files:
                                    file_paths = [f for f in trial.files.values()]
                                    self.skipped_files.append({
                                        'subject': subject_id,
                                        'action': action_id,
                                        'trial': trial_id,
                                        'reason': 'gyroscope_length_mismatch',
                                        'acc_len': acc_len,
                                        'gyro_len': gyro_len,
                                        'diff': length_diff,
                                        'files': file_paths
                                    })
                                self.skip_stats['skipped_length_mismatch'] += 1
                                self.subject_modality_stats[subject_id]['skipped_length_mismatch'] += 1
                                continue
                        else:
                            # Mode 3: DTW alignment (DEPRECATED - use timestamp alignment instead)
                            if length_diff >= 10:
                                if self.debug:
                                    print(f"Skipping S{subject_id}A{action_id}T{trial_id}: "
                                          f"DTW length diff too large (acc={acc_len}, gyro={gyro_len}, diff={length_diff} >= 10)")
                                self.skip_stats['skipped_dtw_length_mismatch'] += 1
                                self.subject_modality_stats[subject_id]['skipped_dtw_length_mismatch'] += 1
                                continue
                            elif length_diff > 0:
                                # Apply DTW alignment (gyro aligned to acc as ground truth)
                                try:
                                    trial_data = align_gyro_to_acc(trial_data, use_fast_dtw=True, max_length_diff=10)
                                    if self.debug:
                                        acc_len_after = trial_data['accelerometer'].shape[0]
                                        gyro_len_after = trial_data['gyroscope'].shape[0]
                                        print(f"S{subject_id}A{action_id}T{trial_id}: DTW aligned "
                                              f"(before: acc={acc_len}, gyro={gyro_len}) → "
                                              f"(after: acc={acc_len_after}, gyro={gyro_len_after})")
                                except ValueError as err:
                                    print(f"Skipping S{subject_id}A{action_id}T{trial_id}: DTW alignment failed: {err}")
                                    self.skip_stats['skipped_dtw_length_mismatch'] += 1
                                    self.subject_modality_stats[subject_id]['skipped_dtw_length_mismatch'] += 1
                                    continue

                    # Gyroscope quality assessment
                    if self.enable_gyro_quality_check and 'gyroscope' in trial_data:
                        from utils.quality import assess_gyro_quality

                        is_acceptable, metrics = assess_gyro_quality(
                            trial_data['gyroscope'],
                            method=self.quality_method,
                            threshold=self.quality_threshold_snr
                        )

                        if self.quality_mode == 'hard':
                            if not is_acceptable:
                                # Hard filter: skip trial entirely
                                print(f"Skipping S{subject_id}A{action_id}T{trial_id}: "
                                      f"Poor gyro quality (SNR={metrics['snr']:.2f})")
                                self.skip_stats['skipped_poor_gyro_hard'] += 1
                                self.subject_modality_stats[subject_id]['skipped_poor_gyro_hard'] += 1
                                continue

                        elif self.quality_mode == 'adaptive':
                            if not is_acceptable:
                                # Adaptive: fall back to accelerometer-only
                                del trial_data['gyroscope']
                                self.skip_stats['skipped_poor_gyro_adaptive'] += 1
                                self.subject_modality_stats[subject_id]['skipped_poor_gyro_adaptive'] += 1

                    # Kalman SMOOTHING (per-channel denoising, NOT fusion)
                    # Applied BEFORE normalization and BEFORE fusion operations
                    # This is different from Kalman fusion which combines acc+gyro → orientation
                    if self.enable_kalman_smoothing and 'accelerometer' in trial_data and 'gyroscope' in trial_data:
                        try:
                            trial_data = kalman_smoothing_for_loader(trial_data, self.kalman_smooth_config)
                        except Exception as err:
                            if self.debug:
                                print(f"Warning: Kalman smoothing failed for S{subject_id}A{action_id}T{trial_id}: {err}")
                            # Continue with raw signals if smoothing fails

                    # Sensor fusion (only if both acc+gyro present and quality passed)
                    if self.enable_sensor_fusion and 'accelerometer' in trial_data and 'gyroscope' in trial_data:
                        from utils.sensor_fusion import apply_sensor_fusion

                        try:
                            trial_data = apply_sensor_fusion(
                                trial_data,
                                method=self.fusion_method,
                                frequency=self.fusion_frequency,
                                **self.fusion_params
                            )
                        except Exception as err:
                            print(f"Warning: Sensor fusion failed for S{subject_id}A{action_id}T{trial_id}: {err}")
                            # Continue with raw gyro data if fusion fails

                    # Kalman filter fusion (replaces gyroscope with orientation features)
                    if self.enable_kalman_fusion and 'accelerometer' in trial_data and 'gyroscope' in trial_data:
                        try:
                            trial_data = kalman_fusion_for_loader(trial_data, self.kalman_config)
                            # Assemble final features: [smv, ax, ay, az, roll, pitch, yaw, ...]
                            kalman_features = assemble_kalman_features(
                                trial_data,
                                include_smv=self.kalman_config.get('kalman_include_smv', True)
                            )
                            # Replace accelerometer with full Kalman features
                            trial_data['accelerometer'] = kalman_features
                            # Remove gyroscope (consumed by Kalman fusion into orientation)
                            if 'gyroscope' in trial_data:
                                del trial_data['gyroscope']
                            # Remove intermediate Kalman outputs (already incorporated)
                            if 'orientation' in trial_data:
                                del trial_data['orientation']
                            if 'uncertainty' in trial_data:
                                del trial_data['uncertainty']
                            if 'innovation' in trial_data:
                                del trial_data['innovation']
                        except Exception as err:
                            print(f"Skipping S{subject_id}A{action_id}T{trial_id}: Kalman fusion failed: {err}")
                            self.skip_stats['skipped_kalman_fusion'] += 1
                            self.subject_modality_stats[subject_id]['skipped_kalman_fusion'] += 1
                            continue

                    try:
                        self._synchronize_modalities(
                            trial_data,
                            subject_id=subject_id,
                            action_id=getattr(trial, 'action_id', None),
                            trial_id=getattr(trial, 'sequence_number', None)
                        )
                    except ValueError as err:
                        if self.debug:
                            print(f"Skipping trial due to modality length issue: {err}")
                        continue
                    try:
                        trial_data, motion_stats = self.process(trial_data, label)
                    except ValueError as err:
                        # Silently count preprocessing errors by type (summary printed at end)
                        error_msg = str(err)
                        self.preprocessing_error_details[error_msg] += 1
                        self.skip_stats['skipped_preprocessing_error'] += 1
                        self.subject_modality_stats[subject_id]['skipped_preprocessing_error'] += 1
                        continue

                    #print(trial_data['skeleton'][0].shape)
                    if self._len_check(trial_data):
                        # Track per-subject statistics
                        self.subject_modality_stats[subject_id]['file_count'] += 1

                        # Count windows generated from this trial
                        num_windows = len(trial_data['labels'])
                        self.subject_modality_stats[subject_id]['window_count'] += num_windows

                        # Track class distribution (fall vs ADL)
                        fall_count = int(np.sum(trial_data['labels'] == 1))
                        adl_count = int(np.sum(trial_data['labels'] == 0))
                        self.subject_modality_stats[subject_id]['fall_windows'] += fall_count
                        self.subject_modality_stats[subject_id]['adl_windows'] += adl_count

                        # Track motion filtering statistics if available
                        if motion_stats is not None:
                            self.subject_modality_stats[subject_id]['motion_total_windows'] += motion_stats['total_windows']
                            self.subject_modality_stats[subject_id]['motion_passed_windows'] += motion_stats['active_windows']
                            self.subject_modality_stats[subject_id]['motion_rejected_windows'] += motion_stats['quiet_windows']

                        self._add_trial_data(trial_data)
                        self.skip_stats['valid_trials'] += 1
                        self.subject_modality_stats[subject_id]['valid_trials'] += 1
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
        Function to normalize the data with selective modality support.

        Normalization modes (via normalize_modalities):
        - 'all': Normalize all modalities (default, backward compatible)
        - 'gyro_only': Only normalize gyroscope data (robust gyro normalization)
        - 'acc_only': Only normalize accelerometer data
        - 'none': Skip normalization entirely (same as enable_normalization=False)

        Robust gyroscope normalization rationale:
        - Gyroscope data often has different units/scales across devices (deg/s vs rad/s)
        - Normalizing gyro brings it to a standard scale for the model
        - Accelerometer already has a natural reference (gravity ~9.8 m/s^2)
        - Keeping acc unnormalized preserves physical interpretation
        '''
        if not self.enable_normalization or self.normalize_modalities == 'none':
            return self.data

        for key, value in self.data.items():
            if key == 'labels':
                continue

            # Selective normalization based on modality
            should_normalize = False
            if self.normalize_modalities == 'all':
                should_normalize = True
            elif self.normalize_modalities == 'gyro_only':
                # Only normalize gyroscope-related modalities
                should_normalize = key in ['gyroscope', 'gyro_magnitude', 'fused']
            elif self.normalize_modalities == 'acc_only':
                # Only normalize accelerometer
                should_normalize = key == 'accelerometer'

            if should_normalize:
                num_samples, length = value.shape[:2]
                norm_data = StandardScaler().fit_transform(value.reshape(num_samples*length, -1))
                self.data[key] = norm_data.reshape(num_samples, length, -1)

        return self.data

    def get_validation_report(self) -> Dict[str, Dict]:
        """
        Get validation report showing modality completeness per subject.

        Returns:
            Dictionary with subject validation statistics
        """
        return dict(self.subject_modality_stats)

    def print_validation_summary(self) -> None:
        """
        Print a summary of modality validation results.
        """
        if not self.subject_modality_stats:
            print("[Validation] No validation statistics available")
            return

        print("\n" + "=" * 70)
        print("DATA VALIDATION SUMMARY")
        print("=" * 70)

        total_subjects = len(self.subject_modality_stats)
        subjects_with_all_modalities = 0
        excluded_subjects = []

        for subject_id, stats in sorted(self.subject_modality_stats.items()):
            has_all_required = True
            if self.required_modalities:
                for modality in self.required_modalities:
                    if stats[modality] == 0:
                        has_all_required = False
                        break

            if has_all_required and stats['valid_trials'] > 0:
                subjects_with_all_modalities += 1
            else:
                excluded_subjects.append(subject_id)

        print(f"Total subjects processed: {total_subjects}")
        print(f"Subjects with all required modalities: {subjects_with_all_modalities}")
        print(f"Subjects excluded (missing modalities): {len(excluded_subjects)}")

        if excluded_subjects:
            print(f"\nExcluded subjects: {', '.join(excluded_subjects)}")

        if self.required_modalities:
            print(f"\nRequired modalities: {', '.join(self.required_modalities)}")

        print("=" * 70 + "\n")

    def print_skip_summary(self) -> None:
        """
        Print detailed summary of trial skip statistics.
        """
        print("\n" + "=" * 70)
        print("TRIAL PROCESSING SUMMARY")
        print("=" * 70)

        # Log key config parameters
        min_win = self.skip_stats.get('min_windows_per_trial', 1)
        win_check = ">= 1" if min_win == 1 else f"> {min_win - 1}"
        print(f"Window filter: {win_check} windows per trial (min_windows_per_trial={min_win})")
        print()

        total = self.skip_stats['total_trials']
        valid = self.skip_stats['valid_trials']
        skipped_total = total - valid

        print(f"Total trials attempted: {total}")
        print(f"Successfully processed: {valid} ({100*valid/total:.1f}%)" if total > 0 else "Successfully processed: 0")
        print(f"Skipped trials: {skipped_total} ({100*skipped_total/total:.1f}%)\n" if total > 0 else "Skipped trials: 0\n")

        if skipped_total > 0:
            print("Skip reasons breakdown:")
            print(f"  - Missing required modalities: {self.skip_stats['skipped_missing_modality']}")
            print(f"  - Length mismatch between modalities: {self.skip_stats['skipped_length_mismatch']}")
            print(f"  - Sequence too short (< {self.max_length} samples): {self.skip_stats['skipped_too_short']}")
            print(f"  - Preprocessing errors: {self.skip_stats['skipped_preprocessing_error']}")
            if self.skip_stats.get('skipped_file_load_error', 0) > 0:
                print(f"  - File load errors (malformed CSV/data): {self.skip_stats['skipped_file_load_error']}")
            print(f"  - DTW length mismatch (acc-gyro diff > 10): {self.skip_stats['skipped_dtw_length_mismatch']}")
            if self.enable_kalman_fusion:
                print(f"  - Kalman fusion failed: {self.skip_stats['skipped_kalman_fusion']}")
            if self.enable_simple_truncation:
                print(f"  - Truncation diff too large (> {self.max_truncation_diff}): {self.skip_stats['skipped_truncation_too_large']}")
            if self.enable_timestamp_alignment:
                print(f"  - Timestamp unsynchronized: {self.skip_stats['skipped_timestamp_unsync']}")
                print(f"  - Insufficient overlap: {self.skip_stats['skipped_insufficient_overlap']}")
                print(f"  - Duration drift (timestamps drifted): {self.skip_stats['skipped_duration_drift']}")
                print(f"  - Rate divergence (sampling rates differ): {self.skip_stats['skipped_rate_drift']}")

            # Print detailed preprocessing error breakdown
            if self.skip_stats['skipped_preprocessing_error'] > 0 and self.preprocessing_error_details:
                print("\nPreprocessing error details:")
                for error_msg, count in self.preprocessing_error_details.most_common():
                    print(f"  - '{error_msg}': Skipped {count} trials")

        # Print simple truncation summary if enabled
        if self.enable_simple_truncation:
            truncated_count = self.skip_stats['simple_truncation_applied']
            print(f"\nSimple truncation summary:")
            print(f"  - Trials truncated to align acc/gyro: {truncated_count}")
            print(f"  - Max allowed difference: {self.max_truncation_diff} samples")

        # Print timestamp alignment summary if enabled
        if self.enable_timestamp_alignment:
            aligned_count = self.skip_stats['timestamp_aligned']
            use_as_is_count = self.skip_stats['timestamp_use_as_is']
            total_successful = aligned_count + use_as_is_count
            print(f"\nTimestamp alignment summary:")
            print(f"  - Total aligned trials: {total_successful}")
            print(f"  - Used as-is (diff <= {self.length_threshold}): {use_as_is_count}")
            print(f"  - Interpolated to {self.alignment_target_rate}Hz: {aligned_count}")

        # Print per-subject statistics for subjects with issues
        print("\nPer-subject breakdown (subjects with skipped trials):")
        problematic_subjects = []
        for subject_id, stats in sorted(self.subject_modality_stats.items()):
            total_sub = stats['total_trials']
            valid_sub = stats['valid_trials']
            skipped_sub = total_sub - valid_sub
            if skipped_sub > 0:
                problematic_subjects.append((subject_id, stats))

        if problematic_subjects:
            # Show first 5 subjects if not in debug mode, all if debug mode
            display_count = len(problematic_subjects) if self.debug else 5
            for subject_id, stats in problematic_subjects[:display_count]:
                total_sub = stats['total_trials']
                valid_sub = stats['valid_trials']
                skipped_sub = total_sub - valid_sub
                print(f"  Subject {subject_id}: {valid_sub}/{total_sub} valid "
                      f"(skipped: missing={stats['skipped_missing_modality']}, "
                      f"mismatch={stats['skipped_length_mismatch']}, "
                      f"short={stats['skipped_too_short']}, "
                      f"error={stats['skipped_preprocessing_error']})")
            if len(problematic_subjects) > display_count:
                print(f"  ... and {len(problematic_subjects) - display_count} more subjects with issues")
        else:
            print("  None - all subjects processed successfully!")

        print("=" * 70 + "\n")

    def compute_motion_rejection_rate(self) -> None:
        """
        Compute motion rejection rate from tracked statistics.
        """
        if self.skip_stats['motion_total_windows'] > 0:
            self.skip_stats['motion_rejection_rate'] = (
                self.skip_stats['motion_rejected_windows'] /
                self.skip_stats['motion_total_windows']
            )
        else:
            self.skip_stats['motion_rejection_rate'] = 0.0

    def get_subject_comprehensive_stats(self, subjects: List[int]) -> pd.DataFrame:
        """
        Get comprehensive per-subject statistics as a DataFrame.

        Args:
            subjects: List of subject IDs to include

        Returns:
            DataFrame with per-subject statistics including:
            - subject_id, file_count, window_count
            - fall_windows, adl_windows
            - motion statistics (if motion filtering was enabled)
            - skip reasons
        """
        rows = []
        for subject_id in subjects:
            subject_id_str = str(subject_id)
            if subject_id_str not in self.subject_modality_stats:
                continue

            stats = self.subject_modality_stats[subject_id_str]
            row = {
                'subject_id': subject_id,
                'file_count': stats['file_count'],
                'total_trials_attempted': stats['total_trials'],
                'valid_trials': stats['valid_trials'],
                'window_count': stats['window_count'],
                'fall_windows': stats['fall_windows'],
                'adl_windows': stats['adl_windows'],
                'skipped_missing': stats['skipped_missing_modality'],
                'skipped_mismatch': stats['skipped_length_mismatch'],
                'skipped_short': stats['skipped_too_short'],
                'skipped_dtw': stats['skipped_dtw_length_mismatch'],
                'skipped_preprocessing': stats['skipped_preprocessing_error']
            }

            # Add motion filtering statistics if available
            if self.enable_motion_filtering:
                row['motion_total'] = stats['motion_total_windows']
                row['motion_passed'] = stats['motion_passed_windows']
                row['motion_rejected'] = stats['motion_rejected_windows']
                if stats['motion_total_windows'] > 0:
                    row['motion_rejection_rate'] = (
                        stats['motion_rejected_windows'] / stats['motion_total_windows']
                    )
                else:
                    row['motion_rejection_rate'] = 0.0

            rows.append(row)

        return pd.DataFrame(rows)

