import time


class AccurateTimer:
    def __init__(self, interval: float = 0.01, *, busy_wait_duration: float = 0.001):
        self._interval = interval
        self._busy_wait_duration = busy_wait_duration

        self._current_time = time.perf_counter()
        self._next_time = self._current_time + self._interval

    @property
    def interval(self) -> float:
        return self._interval

    @property
    def time(self) -> float:
        return self._current_time

    def sleep(self):
        self._current_time = time.perf_counter()
        self._next_time += self._interval

        # Coarse sleep with 1 ms margin
        sleep_time = self._next_time - self._current_time - self._busy_wait_duration
        if sleep_time > 0:
            time.sleep(sleep_time)

        # Busy-wait for precise timing
        while time.perf_counter() < self._next_time:
            pass
