import time

import numpy as np


class SlidingWindowEstimator:
    def __init__(self, initial_value, *, window_size: int = 16):
        self._window = np.zeros(16, dtype=np.float64)
        self._window_pos = 0
        self._window_size = 0

        self._value = initial_value
        self._last_time = time.perf_counter()

    @property
    def value(self) -> float:
        return self._value

    def accuracy(self) -> float:
        return self._window_size / len(self._window)

    def reset(self):
        self._window_pos = 0
        self._window_size = 0

    def update(self, skip_estimate: bool = False):
        current_time = time.perf_counter()
        delta_time = current_time - self._last_time
        self._last_time = current_time

        if skip_estimate:
            return

        # Reject outliers that are too short (less than 50% of current estimate)
        if delta_time < self._value * 0.5:
            return

        # Insert the new value
        self._window[self._window_pos] = delta_time
        self._window_pos = (self._window_pos + 1) % len(self._window)

        # Increase window size until we reach full capacity
        if self._window_size < len(self._window):
            self._window_size += 1

        # Calculate a new estimate
        self._value = np.sum(self._window[: self._window_size]) / self._window_size
