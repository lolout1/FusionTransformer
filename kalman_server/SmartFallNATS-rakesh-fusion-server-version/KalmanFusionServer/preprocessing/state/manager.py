import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

from ..base import KalmanState, KalmanFilterProtocol
from ..filters import LinearKalmanFilter


@dataclass
class UserState:
    kalman_state: Optional[KalmanState] = None
    last_timestamp_ms: float = 0.0
    last_update: float = field(default_factory=time.time)
    cached_orientations: Optional[np.ndarray] = None
    cached_timestamps: Optional[np.ndarray] = None


class UserStateManager:
    """Thread-safe per-user Kalman state manager with orientation caching.

    Features:
    - Persistent Kalman state across inference windows (per user)
    - Orientation caching for 127-sample overlap optimization
    - Timestamp-based overlap detection
    - Automatic gap-based reset (default 10s)
    - TTL-based cleanup for inactive sessions
    """

    def __init__(
        self,
        timeout_ms: float = 10000.0,
        cache_ttl_ms: float = 60000.0,
        window_size: int = 128,
        default_fs_hz: float = 30.0,
    ):
        self.timeout_ms = timeout_ms
        self.cache_ttl_ms = cache_ttl_ms
        self.window_size = window_size
        self.default_fs_hz = default_fs_hz
        self._states: Dict[str, UserState] = {}
        self._lock = threading.Lock()

    def get_or_create(self, user_id: str) -> UserState:
        with self._lock:
            if user_id not in self._states:
                self._states[user_id] = UserState()
            state = self._states[user_id]
            state.last_update = time.time()
            return state

    def update(self, user_id: str, state: UserState) -> None:
        with self._lock:
            state.last_update = time.time()
            self._states[user_id] = state

    def detect_overlap(
        self,
        user_state: UserState,
        current_timestamps: np.ndarray,
    ) -> Tuple[bool, int]:
        """Detect overlap between current window and cached orientations.

        Returns:
            (is_overlap, n_reusable): Whether overlap detected and how many cached orientations to reuse.
        """
        if user_state.cached_timestamps is None or len(user_state.cached_timestamps) == 0:
            return False, 0

        if current_timestamps is None or len(current_timestamps) == 0:
            return False, 0

        # Expected sample duration in ms
        sample_duration_ms = 1000.0 / self.default_fs_hz

        # Previous window's last timestamp
        prev_end_ts = user_state.cached_timestamps[-1]
        curr_start_ts = current_timestamps[0]

        # For 127-sample overlap, current window should start at
        # prev_end_ts - (window_size - 1) * sample_duration
        expected_overlap_start = prev_end_ts - (self.window_size - 2) * sample_duration_ms

        # Allow 1.5 sample durations of tolerance
        tolerance = 1.5 * sample_duration_ms

        if abs(curr_start_ts - expected_overlap_start) < tolerance:
            # Full overlap detected (127 samples)
            n_reusable = self.window_size - 1
            return True, n_reusable

        # Check for partial overlap by matching timestamps
        n_reusable = 0
        for i, cached_ts in enumerate(user_state.cached_timestamps):
            if i >= len(current_timestamps):
                break
            if abs(cached_ts - current_timestamps[i]) < tolerance:
                n_reusable += 1
            else:
                break

        return n_reusable > 0, n_reusable

    def should_reset(
        self,
        user_state: UserState,
        current_start_ts: float,
    ) -> bool:
        """Check if Kalman state should be reset due to time gap."""
        if user_state.last_timestamp_ms == 0:
            return True  # First window

        gap_ms = current_start_ts - user_state.last_timestamp_ms
        return gap_ms > self.timeout_ms

    def process_window(
        self,
        user_id: str,
        acc: np.ndarray,
        gyro: np.ndarray,
        timestamps: np.ndarray,
        kalman_filter: KalmanFilterProtocol,
    ) -> np.ndarray:
        """Process a window with state persistence and orientation caching.

        Args:
            user_id: Unique user/device identifier
            acc: Accelerometer data (N, 3)
            gyro: Gyroscope data (N, 3) in rad/s
            timestamps: Per-sample timestamps in ms (N,)
            kalman_filter: Kalman filter instance (state will be set from cache)

        Returns:
            orientations: Kalman-filtered orientations (N, 3) as [roll, pitch, yaw]
        """
        user_state = self.get_or_create(user_id)
        n_samples = acc.shape[0]
        orientations = np.zeros((n_samples, 3))

        # Check for time gap -> reset
        if self.should_reset(user_state, timestamps[0]):
            kalman_filter.reset(acc[0])
            user_state.kalman_state = None
            user_state.cached_orientations = None
            user_state.cached_timestamps = None
        elif user_state.kalman_state is not None:
            # Restore previous filter state
            kalman_filter.set_state(user_state.kalman_state)

        # Check for overlap
        is_overlap, n_reusable = self.detect_overlap(user_state, timestamps)

        if is_overlap and n_reusable > 0 and user_state.cached_orientations is not None:
            # Reuse cached orientations
            n_cached = min(n_reusable, len(user_state.cached_orientations), n_samples - 1)
            orientations[:n_cached] = user_state.cached_orientations[-n_cached:]
            start_idx = n_cached
        else:
            start_idx = 0

        # Compute dt from timestamps
        if len(timestamps) > 1:
            dt = (timestamps[1] - timestamps[0]) / 1000.0
        else:
            dt = 1.0 / self.default_fs_hz

        # Process remaining samples
        for i in range(start_idx, n_samples):
            if i > 0:
                dt_i = (timestamps[i] - timestamps[i - 1]) / 1000.0
                if dt_i <= 0:
                    dt_i = dt
            else:
                dt_i = dt

            kalman_filter.predict(dt_i)
            kalman_filter.update(acc[i], gyro[i])
            orientations[i] = kalman_filter.get_orientation()

        # Update cache for next window
        user_state.kalman_state = kalman_filter.get_state()
        user_state.last_timestamp_ms = timestamps[-1]
        user_state.cached_orientations = orientations.copy()
        user_state.cached_timestamps = timestamps.copy()

        self.update(user_id, user_state)
        return orientations

    def cleanup_stale(self) -> int:
        """Remove stale sessions older than cache_ttl_ms. Returns count removed."""
        now = time.time()
        ttl_seconds = self.cache_ttl_ms / 1000.0
        removed = 0

        with self._lock:
            stale_ids = [
                uid for uid, state in self._states.items()
                if now - state.last_update > ttl_seconds
            ]
            for uid in stale_ids:
                del self._states[uid]
                removed += 1

        return removed

    def clear(self) -> None:
        """Clear all user states."""
        with self._lock:
            self._states.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._states)
