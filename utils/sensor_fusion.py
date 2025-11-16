"""
Sensor fusion algorithms for IMU orientation estimation.

Implements Madgwick and complementary filters to fuse accelerometer and
gyroscope data into orientation estimates. These filters correct gyroscope
drift using accelerometer gravity reference.

References:
    Madgwick, S. (2010). "An efficient orientation filter for inertial and
    inertial/magnetic sensor arrays." University of Bristol Tech Report.
    https://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/

    Zhang et al. (2024). "Human Activity Recognition Based on Deep Learning
    Regardless of Sensor Orientation." Applied Sciences, 14(9), 3637.
    DOI: 10.3390/app14093637 (Achieved 97.13% accuracy using Madgwick+ResNet)

    ahrs library documentation:
    https://ahrs.readthedocs.io/en/latest/filters/madgwick.html
"""

import numpy as np
from typing import Dict, Tuple
from scipy.spatial.transform import Rotation


def madgwick_fusion(acc_data: np.ndarray,
                   gyro_data: np.ndarray,
                   frequency: float = 30.0,
                   beta: float = 0.1) -> np.ndarray:
    """
    Madgwick AHRS algorithm for orientation estimation from IMU.

    Fuses accelerometer (gravity reference) and gyroscope (rotation rate)
    to estimate orientation angles. Corrects gyroscope drift over time.

    Algorithm:
        1. Predict orientation from gyroscope integration
        2. Correct using accelerometer gradient descent
        3. Output quaternion (converted to Euler angles)

    Args:
        acc_data: Accelerometer measurements, shape (N, 3) in m/s²
            Expected: ~9.81 m/s² magnitude when static (Earth's gravity)
        gyro_data: Gyroscope measurements, shape (N, 3) in rad/s
            Angular velocity around each axis
        frequency: Sampling frequency in Hz (default 30.0 for Android IMU)
        beta: Algorithm gain (convergence rate)
            Higher = faster convergence but more noise
            Lower = smoother but slower response
            Typical range: 0.03-0.3, default 0.1

    Returns:
        orientation: Euler angles, shape (N, 3) in degrees
            [roll, pitch, yaw] representing device orientation
            - roll: Rotation around x-axis (left/right tilt)
            - pitch: Rotation around y-axis (forward/back tilt)
            - yaw: Rotation around z-axis (heading)

    Example:
        >>> acc = np.random.randn(1000, 3) * 2 + np.array([0, 0, 9.81])
        >>> gyro = np.random.randn(1000, 3) * 0.1
        >>> angles = madgwick_fusion(acc, gyro, frequency=30.0, beta=0.1)
        >>> print(f"Roll: {angles[:, 0].mean():.1f}°")

    Notes:
        - Requires both accelerometer and gyroscope
        - Works best when device experiences some rotation
        - Beta=0.1 is good default for 30Hz sampling
        - Beta=0.033 recommended for 100Hz+ sampling
    """
    try:
        from ahrs.filters import Madgwick
    except ImportError:
        raise ImportError("ahrs library required. Install with: pip install ahrs")

    # Initialize Madgwick filter
    madgwick = Madgwick(frequency=frequency, beta=beta)

    # Initial quaternion (no rotation)
    q = np.array([1.0, 0.0, 0.0, 0.0])

    quaternions = []
    for i in range(len(acc_data)):
        # Update filter with current measurements
        # Note: ahrs uses gyr then acc order
        q = madgwick.updateIMU(q, gyr=gyro_data[i], acc=acc_data[i])
        quaternions.append(q.copy())

    # Convert quaternions to Euler angles
    quaternions = np.array(quaternions)
    rotation = Rotation.from_quat(quaternions)
    euler_angles = rotation.as_euler('xyz', degrees=True)

    return euler_angles


def complementary_filter(acc_data: np.ndarray,
                         gyro_data: np.ndarray,
                         frequency: float = 30.0,
                         alpha: float = 0.98) -> np.ndarray:
    """
    Complementary filter for orientation estimation.

    Simpler alternative to Madgwick. Combines gyroscope (high-frequency)
    and accelerometer (low-frequency) using weighted average.

    Filter equation:
        angle = alpha * (angle + gyro*dt) + (1-alpha) * acc_angle

    Args:
        acc_data: Accelerometer, shape (N, 3) in m/s²
        gyro_data: Gyroscope, shape (N, 3) in rad/s
        frequency: Sampling frequency in Hz
        alpha: Gyroscope trust factor (0-1)
            0.98 = trust gyro 98% for high-freq motion
            0.02 = trust acc 2% for low-freq drift correction
            Typical range: 0.95-0.99

    Returns:
        orientation: Euler angles, shape (N, 3) in degrees

    Example:
        >>> angles = complementary_filter(acc, gyro, frequency=30.0, alpha=0.98)

    Notes:
        - Faster than Madgwick (less computation)
        - Good for systems with limited resources
        - Alpha near 1.0 trusts gyroscope more (responsive but drifts)
        - Alpha near 0.0 trusts accelerometer more (stable but noisy)
    """
    try:
        from ahrs.filters import Complementary
    except ImportError:
        raise ImportError("ahrs library required. Install with: pip install ahrs")

    # Initialize complementary filter
    # Note: ahrs uses gain = 1 - alpha
    comp_filter = Complementary(frequency=frequency, gain=1-alpha)

    q = np.array([1.0, 0.0, 0.0, 0.0])
    quaternions = []

    for i in range(len(acc_data)):
        q = comp_filter.updateIMU(q, gyr=gyro_data[i], acc=acc_data[i])
        quaternions.append(q.copy())

    quaternions = np.array(quaternions)
    rotation = Rotation.from_quat(quaternions)
    euler_angles = rotation.as_euler('xyz', degrees=True)

    return euler_angles


def apply_sensor_fusion(trial_data: Dict[str, np.ndarray],
                        method: str = 'madgwick',
                        frequency: float = 30.0,
                        **params) -> Dict[str, np.ndarray]:
    """
    Apply sensor fusion to trial data and update dictionary.

    Replaces 'gyroscope' entry with 'orientation' containing Euler angles.
    This enables models to use orientation features instead of raw gyroscope.

    Args:
        trial_data: Dictionary with 'accelerometer' and 'gyroscope' keys
            Each should have shape (N, 3)
        method: Fusion algorithm
            'madgwick' - Gradient descent orientation filter (default)
            'complementary' - Weighted gyro+acc combination
        frequency: Sampling rate in Hz
        **params: Algorithm-specific parameters
            For Madgwick: madgwick_beta (default 0.1)
            For Complementary: comp_alpha (default 0.98)

    Returns:
        updated_trial_data: Dictionary with 'orientation' instead of 'gyroscope'

    Example:
        >>> trial = {'accelerometer': acc, 'gyroscope': gyro}
        >>> fused = apply_sensor_fusion(trial, method='madgwick', frequency=30.0)
        >>> print(f"Keys: {list(fused.keys())}")  # ['accelerometer', 'orientation']

    Raises:
        ValueError: If unknown fusion method or missing required modalities
    """
    # Validate inputs
    if 'accelerometer' not in trial_data or 'gyroscope' not in trial_data:
        raise ValueError("Sensor fusion requires both 'accelerometer' and 'gyroscope'. "
                        f"Available keys: {list(trial_data.keys())}")

    acc = trial_data['accelerometer']
    gyro = trial_data['gyroscope']

    # Apply fusion algorithm
    if method == 'madgwick':
        beta = params.get('madgwick_beta', 0.1)
        orientation = madgwick_fusion(acc, gyro, frequency=frequency, beta=beta)

    elif method == 'complementary':
        alpha = params.get('comp_alpha', 0.98)
        orientation = complementary_filter(acc, gyro, frequency=frequency, alpha=alpha)

    else:
        raise ValueError(f"Unknown fusion method: {method}. "
                        f"Supported: 'madgwick', 'complementary'")

    # Update trial data: remove gyro, add orientation
    updated_data = trial_data.copy()
    del updated_data['gyroscope']
    updated_data['orientation'] = orientation

    return updated_data


def compute_orientation_features(orientation: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract additional features from orientation angles.

    Computes derived features that may be useful for activity recognition:
    - Orientation magnitude (total rotation)
    - Rate of change (angular velocity from angles)
    - Tilt relative to gravity

    Args:
        orientation: Euler angles, shape (N, 3) in degrees

    Returns:
        features: Dictionary of derived features
            - 'magnitude': Overall orientation magnitude
            - 'rate': Rate of orientation change (degrees/sample)
            - 'tilt': Tilt away from upright (degrees)

    Example:
        >>> angles = madgwick_fusion(acc, gyro)
        >>> features = compute_orientation_features(angles)
        >>> print(f"Max tilt: {features['tilt'].max():.1f}°")
    """
    # Magnitude of rotation (Euclidean norm of Euler angles)
    magnitude = np.linalg.norm(orientation, axis=1)

    # Rate of change (finite difference)
    rate = np.zeros_like(orientation)
    rate[1:] = np.diff(orientation, axis=0)

    # Tilt from vertical (assuming z-axis is up when upright)
    # Combine roll and pitch to get total tilt
    roll = orientation[:, 0]
    pitch = orientation[:, 1]
    tilt = np.sqrt(roll**2 + pitch**2)

    features = {
        'magnitude': magnitude,
        'rate': rate,
        'tilt': tilt
    }

    return features
