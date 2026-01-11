"""
Trial-level preprocessing with Kalman filtering.

Integrates with existing loader.py DatasetBuilder class.

Supports multiple filter types:
- 'linear': Linear Kalman Filter (Euler state, no bias estimation)
- 'ekf': Extended Kalman Filter (quaternion state + gyro bias)
- 'ukf': Standard Unscented Kalman Filter (quaternion + gyro bias, simpler/stable)
- 'sr_ukf': Square Root UKF (quaternion + gyro bias, numerically robust)
- 'madgwick': Madgwick AHRS Filter (gradient descent optimization)
- 'adaptive': Adaptive Kalman Filter (dynamic measurement noise)
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
                         return_filter_state: bool = False,
                         madgwick_beta: float = 0.1,
                         adaptive_threshold_g: float = 2.0,
                         adaptive_R_scale_max: float = 10.0,
                         vqf_tau_acc: float = 3.0,
                         **kwargs) -> Dict:
    """
    Process a single trial through orientation filter.

    Args:
        acc_data: (T, 3) accelerometer in m/s²
        gyro_data: (T, 3) gyroscope in rad/s (MUST be converted from deg/s)
        filter_type: 'linear', 'ekf', 'madgwick', 'adaptive', or 'vqf'
        output_format: 'euler' (roll, pitch, yaw) or 'quaternion' (q0, q1, q2, q3)
        include_uncertainty: Include P diagonal as features (KF only)
        include_innovation: Include innovation magnitude as feature (KF only)
        Q_params: Override default process noise
        R_params: Override default measurement noise
        dt: Time step in seconds
        return_filter_state: Return final filter state for analysis
        madgwick_beta: Madgwick filter beta parameter (default 0.1)
        adaptive_threshold_g: Adaptive KF acceleration threshold in g
        adaptive_R_scale_max: Adaptive KF max R scaling factor

    Returns:
        dict with keys:
            'orientation': (T, 3) or (T, 4) depending on output_format
            'uncertainty': (T, 3) if include_uncertainty (KF only)
            'innovation': (T, 1) if include_innovation (KF only)
            'filter_state': final filter state if return_filter_state
            'gyro_bias': (3,) estimated bias if filter supports it

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

    T = len(acc_data)

    # Handle different filter types
    if filter_type == 'none':
        # No Kalman filtering - return raw gyro as "orientation"
        result = {'orientation': gyro_data.copy()}
        if include_uncertainty:
            result['uncertainty'] = np.zeros((T, 3))
        if include_innovation:
            result['innovation'] = np.zeros((T, 1))
        return result

    if filter_type == 'madgwick':
        from .madgwick import process_trial_madgwick
        orientations = process_trial_madgwick(
            acc_data, gyro_data,
            beta=madgwick_beta,
            sample_freq=1.0/dt,
            output_format=output_format
        )
        result = {'orientation': orientations}
        # Madgwick doesn't have uncertainty/innovation
        if include_uncertainty:
            result['uncertainty'] = np.zeros((T, 3))
        if include_innovation:
            result['innovation'] = np.zeros((T, 1))
        return result

    elif filter_type == 'adaptive':
        from .adaptive_kalman import process_trial_adaptive
        orientations, diagnostics = process_trial_adaptive(
            acc_data, gyro_data,
            filter_type='linear',
            threshold_g=adaptive_threshold_g,
            R_scale_max=adaptive_R_scale_max,
            output_format=output_format,
            dt=dt,
            return_diagnostics=True,
            **(Q_params or {}),
            **(R_params or {})
        )
        result = {'orientation': orientations, 'diagnostics': diagnostics}
        if include_uncertainty:
            result['uncertainty'] = np.zeros((T, 3))
        if include_innovation:
            result['innovation'] = np.zeros((T, 1))
        return result

    elif filter_type == 'vqf':
        from .vqf_wrapper import process_trial_vqf, VQF_AVAILABLE
        if not VQF_AVAILABLE:
            raise ImportError("VQF library not installed. Install with: pip install vqf")
        orientations = process_trial_vqf(
            acc_data, gyro_data,
            sample_freq=1.0/dt,
            tau_acc=vqf_tau_acc,
            output_format=output_format
        )
        result = {'orientation': orientations}
        # VQF doesn't have uncertainty/innovation
        if include_uncertainty:
            result['uncertainty'] = np.zeros((T, 3))
        if include_innovation:
            result['innovation'] = np.zeros((T, 1))
        return result

    elif filter_type == 'sr_ukf':
        # Square Root Unscented Kalman Filter - most robust for falls
        from .sr_ukf import process_trial_sr_ukf

        # Build SR-UKF parameters
        ukf_params = {}
        if Q_params:
            if 'Q_quat' in Q_params:
                ukf_params['Q_quat'] = Q_params['Q_quat']
            if 'Q_bias' in Q_params:
                ukf_params['Q_bias'] = Q_params['Q_bias']
        if R_params:
            if 'R_acc' in R_params:
                ukf_params['R_acc'] = R_params['R_acc']

        # Adaptive R parameters
        ukf_params['enable_adaptive_R'] = kwargs.get('enable_adaptive_R', True)
        ukf_params['adaptive_threshold_g'] = adaptive_threshold_g
        ukf_params['adaptive_R_scale_max'] = adaptive_R_scale_max

        result = process_trial_sr_ukf(
            acc_data, gyro_data,
            output_format=output_format,
            dt=dt,
            **ukf_params
        )

        # SR-UKF returns dict with orientation, uncertainty, innovation, gyro_bias
        return result

    elif filter_type == 'ukf':
        # Standard Unscented Kalman Filter - simpler and more numerically stable
        from .ukf import process_trial_ukf

        # Build UKF parameters
        ukf_params = {}
        if Q_params:
            if 'Q_quat' in Q_params:
                ukf_params['Q_quat'] = Q_params['Q_quat']
            if 'Q_bias' in Q_params:
                ukf_params['Q_bias'] = Q_params['Q_bias']
        if R_params:
            if 'R_acc' in R_params:
                ukf_params['R_acc'] = R_params['R_acc']

        # Adaptive R parameters
        ukf_params['enable_adaptive_R'] = kwargs.get('enable_adaptive_R', True)
        ukf_params['adaptive_threshold_g'] = adaptive_threshold_g
        ukf_params['adaptive_R_scale_max'] = adaptive_R_scale_max

        result = process_trial_ukf(
            acc_data, gyro_data,
            output_format=output_format,
            dt=dt,
            **ukf_params
        )

        # UKF returns dict with orientation, uncertainty, innovation, gyro_bias
        return result

    # Standard Kalman filters (linear, ekf)
    # Build filter parameters
    params = {}
    if Q_params:
        params.update(Q_params)
    if R_params:
        params.update(R_params)

    # Create filter
    kf = create_filter(filter_type, **params)

    # Determine orientation dimension
    if output_format == 'quaternion' and filter_type == 'ekf':
        ori_dim = 4
    elif output_format == 'gravity_vector':
        ori_dim = 3  # [gx, gy, gz]
    else:
        ori_dim = 3  # euler [roll, pitch, yaw]

    # Storage
    orientations = np.zeros((T, ori_dim))
    uncertainties = np.zeros((T, 3))
    innovations = np.zeros(T)

    # Process each timestep
    for t in range(T):
        if filter_type == 'linear':
            kf.predict(dt)
            kf.update(acc_data[t], gyro_data[t])
            if output_format == 'gravity_vector':
                orientations[t] = kf.get_gravity_vector()
            else:
                orientations[t] = kf.get_orientation()
        else:
            kf.predict(gyro_data[t], dt)
            kf.update(acc_data[t])
            if output_format == 'quaternion':
                orientations[t] = kf.get_orientation_quaternion()
            elif output_format == 'gravity_vector':
                orientations[t] = kf.get_gravity_vector()
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

    # Filter-specific parameters
    madgwick_beta = config.get('madgwick_beta', 0.1)
    adaptive_threshold_g = config.get('adaptive_threshold_g', 2.0)
    adaptive_R_scale_max = config.get('adaptive_R_scale_max', 10.0)
    vqf_tau_acc = config.get('vqf_tau_acc', 3.0)

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
        dt=1.0 / fs,
        madgwick_beta=madgwick_beta,
        adaptive_threshold_g=adaptive_threshold_g,
        adaptive_R_scale_max=adaptive_R_scale_max,
        vqf_tau_acc=vqf_tau_acc
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


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi] range."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def assemble_kalman_features(trial_data: Dict[str, np.ndarray],
                             include_smv: bool = True,
                             yaw_replacement: str = 'none',
                             gyro_data: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Assemble final feature array from Kalman-processed trial data.

    Args:
        trial_data: Dict with 'accelerometer', 'orientation', optional 'uncertainty', 'innovation'
        include_smv: Prepend signal magnitude vector
        yaw_replacement: How to handle yaw (7th channel):
            - 'none': Keep Kalman-filtered yaw (default, original behavior)
            - 'gyro_mag': Replace yaw with gyroscope magnitude sqrt(gx²+gy²+gz²)
            - 'gyro_z': Replace yaw with integrated gyro Z-axis angle
            - 'relative': Use relative yaw wrap(yaw(t) - yaw(t0)) - drift-invariant
        gyro_data: Original (T, 3) gyroscope data in rad/s (required for gyro_mag/gyro_z)

    Returns:
        (T, C) feature array

    Notes:
        - yaw_replacement='relative' is recommended for robustness since it removes
          absolute yaw drift while preserving yaw dynamics during falls
        - gyro_mag captures rotation intensity without direction
        - gyro_z is similar to yaw rate integrated, useful for rotation detection
    """
    acc = trial_data['accelerometer']
    ori = trial_data['orientation'].copy()  # Copy to avoid modifying original

    # Apply yaw replacement if specified
    if yaw_replacement != 'none':
        if ori.shape[1] >= 3:  # Euler angles: [roll, pitch, yaw]
            yaw_idx = 2  # yaw is the 3rd column (index 2)

            if yaw_replacement == 'relative':
                # Relative yaw: wrap(yaw(t) - yaw(t0))
                # This removes absolute drift while preserving relative rotation
                yaw_initial = ori[0, yaw_idx]
                ori[:, yaw_idx] = _wrap_angle(ori[:, yaw_idx] - yaw_initial)

            elif yaw_replacement == 'gyro_mag':
                # Replace yaw with gyroscope magnitude
                if gyro_data is None:
                    raise ValueError("gyro_data required for yaw_replacement='gyro_mag'")
                gyro_mag = np.sqrt(np.sum(gyro_data**2, axis=1))
                ori[:, yaw_idx] = gyro_mag

            elif yaw_replacement == 'gyro_z':
                # Replace yaw with integrated gyro Z-axis angle
                if gyro_data is None:
                    raise ValueError("gyro_data required for yaw_replacement='gyro_z'")
                # Integrate gyro_z over time (cumulative sum * dt)
                # Assumes dt=1/30 (30Hz), but the relative magnitude matters more
                dt = 1.0 / 30.0
                gyro_z_integrated = np.cumsum(gyro_data[:, 2]) * dt
                # Wrap to [-pi, pi] to prevent unbounded growth
                ori[:, yaw_idx] = _wrap_angle(gyro_z_integrated)

            else:
                raise ValueError(f"Unknown yaw_replacement: {yaw_replacement}")

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
