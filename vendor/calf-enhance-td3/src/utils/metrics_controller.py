from mlflow.entities import Metric
import time
import mlflow
from collections import defaultdict
from collections import deque
import numpy as np


def get_current_time_millis():
    """
    Returns the time in milliseconds since the epoch as an integer number.
    """
    return int(time.time() * 1000)


class MetricsCollector:

    def __init__(self, rolling_window_size: int = 20):
        self.pending_metrics: list[Metric] = []
        self.rolling_window_size = rolling_window_size
        self.rolling_window = defaultdict(lambda: deque(maxlen=rolling_window_size))

    def append_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
        timestamp: int | None = None,
    ):
        self.pending_metrics.append(
            Metric(
                key=key,
                value=value,
                timestamp=timestamp or get_current_time_millis(),
                step=step,
            )
        )

    def collect_metrics_from_final_episode_info(self, info: dict, step: int):
        episode_return = info["episode"]["r"]
        episode_length = info["episode"]["l"]
        self.rolling_window["episodic_return"].append(episode_return)
        self.rolling_window["episodic_length"].append(episode_length)
        self.append_metric("charts/episodic_return", episode_return, step=step)
        self.append_metric("charts/episodic_length", episode_length, step=step)
        self.append_metric(
            f"charts/episodic_return_rolling_{self.rolling_window_size}",
            np.mean(self.rolling_window["episodic_return"]),
            step=step,
        )
        self.append_metric(
            f"charts/episodic_length_rolling_{self.rolling_window_size}",
            np.mean(self.rolling_window["episodic_length"]),
            step=step,
        )

    def log_pending_metrics(self, synchronous: bool = True):
        active_run = mlflow.active_run()
        if active_run is not None and len(self.pending_metrics) > 0:
            run_id = active_run.info.run_id
            mlflow.MlflowClient().log_batch(
                run_id=run_id, metrics=self.pending_metrics, synchronous=synchronous
            )
            self.pending_metrics.clear()
