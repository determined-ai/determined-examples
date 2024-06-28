import torch


class CUDAEventTimer:
    """
    Helper class for timing CUDA operations.

    Example usage:

    ```python
    # Time with `start` and `stop` methods:

    timer = CUDAEventTimer()
    for iteration in range(repeats):
        timer.start()
        # Do some computation here
        timer.stop()
    time_list_s = timer.time_list_s # List of each iteration's duration in seconds
    time_s_mean= timer.mean_time_s


    # Or use as a context manager:
    timer = CUDAEventTimer()
    with timer:
        # Do some computation here
    elapsed_time_s = timer.total_time_s
    ```
    """

    def __init__(self) -> None:
        self._start_events: list[torch.cuda.Event] = []
        self._end_events: list[torch.cuda.Event] = []

    @property
    def time_s_list(self) -> list[float]:
        # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/11
        torch.cuda.synchronize()
        time_list_s = [
            s.elapsed_time(e) / 1e3 for s, e in zip(self._start_events, self._end_events)
        ]
        return time_list_s

    @property
    def time_s_total(self) -> float:
        total_time_s = sum(self.time_s_list)
        return total_time_s

    @property
    def time_s_mean(self) -> float:
        return self.time_s_total / len(self._start_events)

    @property
    def time_s_std(self) -> float:
        return torch.tensor(self.time_s_list).std().item()

    def start(self) -> None:
        self._start_events.append(torch.cuda.Event(enable_timing=True))
        self._end_events.append(torch.cuda.Event(enable_timing=True))
        self._start_events[-1].record()

    def stop(self) -> None:
        self._end_events[-1].record()

    def __enter__(self) -> None:
        self.start()

    def __exit__(self, *args, **kwargs) -> None:
        self.stop()
