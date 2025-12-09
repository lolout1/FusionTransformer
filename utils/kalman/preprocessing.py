"""
Trial-level preprocessing with Kalman filtering.

Integrates with existing loader.py DatasetBuilder class.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .filters import create_filter
from .features import compute_smv, build_kalman_features


def process_trial_kalman(acc_data: np.ndarray,
                         gyro_data: np.ndarray,
                         filter_type: str = 'linear',
                         output_format: str = 'euler',
                         include_uncertainty: bool = False,
                         include_innovation: bool = False,
                         Q_params: Optional[Dict] = None,
                         R_params: Optional[Dict] = None,
                         dt: float = 1/30.0,
                         return_filter_state: bool = False) -> Dict:
    """
    Process a single trial through Kalman filter.

    Args:
        acc_data: (T, 3) accelerometer in m/s²
        gyro_data: (T, 3) gyroscope in rad/s (MUST be converted from deg/s)
        filter_type: 'linear' (Euler state) or 'ekf' (quaternion + bias)
        output_format: 'euler' (roll, pitch, yaw) or 'quaternion' (q0, q1, q2, q3)
        include_uncertainty: Include P diagonal as features
        include_innovation: Include innovation magnitude as feature
        Q_params: Override default process noise
        R_params: Override default measurement noise
        dt: Time step in seconds
        return_filter_state: Return final filter state for analysis

    Returns:
        dict with keys:
            'orientation': (T, 3) or (T, 4) depending on output_format
            'uncertainty': (T, 3) if include_uncertainty
            'innovation': (T, 1) if include_innovation
            'filter_state': final KF state if return_filter_state
            'gyro_bias': (3,) estimated bias if filter_type='ekf'

    Raises:
        ValueError: If acc_data and gyro_data have different lengths
        ValueError: If gyro appears to be in deg/s (values > 10 rad/s)
    """
    # Validation
    if len(acc_data) != len(gyro_data):
        raise ValueError(f"acc_data ({len(acc_data)}) and gyro_data ({len(gyro_data)}) "
                        "have different lengths")

    gyro_max = np.abs(gyro_data).max()
    if gyro_max > 10:
        raise ValueError(f"Gyroscope max value is {gyro_max:.1f} rad/s, which suggests "
                        "input may be in deg/s. Convert to rad/s first.")

    # Build filter parameters
    params = {}
    if Q_params:
        params.update(Q_params)
    if R_params:
        params.update(R_params)

    # Create filter
    kf = create_filter(filter_type, **params)

    T = len(acc_data)

    # Determine orientation dimension
    if output_format == 'quaternion' and filter_type == 'ekf':
        ori_dim = 4
    else:
        ori_dim = 3

    # Storage
    orientations = np.zeros((T, ori_dim))
    uncertainties = np.zeros((T, 3))
    innovations = np.zeros(T)

    # Process each timestep
    for t in range(T):
        if filter_type == 'linear':
            kf.predict(dt)
            kf.update(acc_data[t], gyro_data[t])
            orientations[t] = kf.get_orientation()
        else:
            kf.predict(gyro_data[t], dt)
            kf.update(acc_data[t])
            if output_format == 'quaternion':
                orientations[t] = kf.get_orientation_quaternion()
            else:
                orientations[t] = kf.get_orientation_euler()

        uncertainties[t] = kf.get_uncertainty()
        innovations[t] = kf.get_innovation_magnitude()

    # Build result
    result = {'orientation': orientations}

    if include_uncertainty:
        result['uncertainty'] = uncertainties

    if include_innovation:
        result['innovation'] = innovations.reshape(-1, 1)

    if return_filter_state:
        if filter_type == 'ekf':
            result['filter_state'] = {
                'quaternion': kf.get_orientation_quaternion(),
                'gyro_bias': kf.get_gyro_bias(),
                'P': kf.P.copy()
            }
        else:
            result['filter_state'] = {
                'state': kf.x.copy(),
                'P': kf.P.copy()
            }

    if filter_type == 'ekf':
        result['gyro_bias'] = kf.get_gyro_bias()

    return result


def kalman_fusion_for_loader(trial_data: Dict[str, np.ndarray],
                              config: Dict) -> Dict[str, np.ndarray]:
    """
    Apply Kalman fusion to trial data (integration with loader.py).

    Replaces 'gyroscope' with 'orientation' (and optionally 'uncertainty', 'innovation').

    Args:
        trial_data: Dict with 'accelerometer' and 'gyroscope' keys
        config: Configuration dict with kalman parameters

    Returns:
        Updated trial_data with orientation instead of gyroscope
    """
    if 'accelerometer' not in trial_data or 'gyroscope' not in trial_data:
        raise ValueError("Kalman fusion requires both 'accelerometer' and 'gyroscope'")

    acc = trial_data['accelerometer']
    gyro = trial_data['gyroscope']

    # Extract config
    filter_type = config.get('kalman_filter_type', 'linear')
    output_format = config.get('kalman_output_format', 'euler')
    include_uncertainty = config.get('kalman_include_uncertainty', False)
    include_innovation = config.get('kalman_include_innovation', False)
    fs = config.get('filter_fs', 30.0)

    # Q and R parameters
    Q_params = {}
    R_params = {}

    if 'kalman_Q_orientation' in config:
        Q_params['Q_orientation'] = config['kalman_Q_orientation']
    if 'kalman_Q_rate' in config:
        Q_params['Q_rate'] = config['kalman_Q_rate']
    if 'kalman_Q_quat' in config:
        Q_params['Q_quat'] = config['kalman_Q_quat']
    if 'kalman_Q_bias' in config:
        Q_params['Q_bias'] = config['kalman_Q_bias']
    if 'kalman_R_acc' in config:
        R_params['R_acc'] = config['kalman_R_acc']
    if 'kalman_R_gyro' in config:
        R_params['R_gyro'] = config['kalman_R_gyro']

    # Process
    result = process_trial_kalman(
        acc_data=acc,
        gyro_data=gyro,
        filter_type=filter_type,
        output_format=output_format,
        include_uncertainty=include_uncertainty,
        include_innovation=include_innovation,
        Q_params=Q_params if Q_params else None,
        R_params=R_params if R_params else None,
        dt=1.0 / fs
    )

    # Update trial data
    updated = trial_data.copy()
    del updated['gyroscope']
    updated['orientation'] = result['orientation']

    if include_uncertainty:
        updated['uncertainty'] = result['uncertainty']
    if include_innovation:
        updated['innovation'] = result['innovation']

    return updated


def assemble_kalman_features(trial_data: Dict[str, np.ndarray],
                             include_smv: bool = True) -> np.ndarray:
    """
    Assemble final feature array from Kalman-processed trial data.

    Args:
        trial_data: Dict with 'accelerometer', 'orientation', optional 'uncertainty', 'innovation'
        include_smv: Prepend signal magnitude vector

    Returns:
        (T, C) feature array
    """
    acc = trial_data['accelerometer']
    ori = trial_data['orientation']

    features = [acc, ori]

    if include_smv:
        smv = compute_smv(acc).reshape(-1, 1)
        features.insert(0, smv)

    if 'uncertainty' in trial_data:
        features.append(trial_data['uncertainty'])

    if 'innovation' in trial_data:
        features.append(trial_data['innovation'])

    return np.hstack(features)


def validate_gyro_units(gyro_data: np.ndarray, threshold: float = 10.0) -> bool:
    """
    Check if gyroscope data appears to be in rad/s.

    Args:
        gyro_data: (T, 3) gyroscope data
        threshold: Maximum expected rad/s for valid data

    Returns:
        True if data appears to be in rad/s
    """
    max_val = np.abs(gyro_data).max()
    return max_val < threshold


def convert_gyro_to_rads(gyro_data: np.ndarray) -> np.ndarray:
    """Convert gyroscope from deg/s to rad/s."""
    return np.deg2rad(gyro_data)
