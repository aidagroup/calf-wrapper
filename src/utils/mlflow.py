import dataclasses
import functools
import importlib.metadata
import mlflow
import numpy as np
import os
import platform
import socket
import subprocess
import sys

from typing import Dict, Any, Tuple, Union, Optional, List
from stable_baselines3.common.logger import (
    HumanOutputFormat,
    KVWriter,
    Logger,
    configure,
    INFO,
)
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class MlflowConfig:
    tracking_uri: str
    """MLflow tracking URI"""

    experiment_name: str
    """MLflow experiment name"""

    run_name: Optional[str] = None
    """MLflow run name"""


class MLflowOutputFormat(KVWriter):
    """Dumps key/value pairs into MLflow's numeric format."""

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


class SilentLogger(Logger):
    def __init__(
        self,
        folder: Optional[str] = None,
        output_formats: Optional[List[KVWriter]] = None,
    ):
        self.name_to_value = defaultdict(
            float
        )  # Preserve the original Logger attributes
        self.name_to_count = defaultdict(int)
        self.name_to_excluded = defaultdict(
            lambda: None
        )  # Use a dictionary for exclusions
        self.level = INFO
        self.folder = folder
        self.output_formats = output_formats or []

        if folder is not None:
            os.makedirs(folder, exist_ok=True)


def create_mlflow_logger():
    logger = SilentLogger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
    )
    return logger


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        flattened = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, child_prefix))
        return flattened
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {prefix: value}
    if isinstance(value, (list, tuple)):
        return {prefix: list(value)}
    if hasattr(value, "__dict__"):
        public = {k: v for k, v in vars(value).items() if not k.startswith("_")}
        return _flatten(public, prefix)
    return {prefix: str(value)}


def reproducibility_tags() -> Dict[str, str]:
    status = _git_value("status", "--porcelain")
    packages = ("gymnasium", "mlflow", "numpy", "stable-baselines3", "torch")
    tags = {
        "repro.git_commit": _git_value("rev-parse", "HEAD"),
        "repro.git_branch": _git_value("branch", "--show-current"),
        "repro.git_dirty": str(bool(status)),
        "repro.hostname": socket.gethostname(),
        "repro.platform": platform.platform(),
        "repro.python": platform.python_version(),
    }
    for package in packages:
        try:
            tags[f"repro.package.{package}"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            tags[f"repro.package.{package}"] = "not-installed"
    return tags


def mlflow_monitoring():
    def inner1(func):
        @functools.wraps(func)
        def inner2(*args, **kwargs):
            mlflow_config: MlflowConfig = args[0].mlflow
            mlflow.set_tracking_uri(mlflow_config.tracking_uri)
            if len(args) == 1 and hasattr(args[0], "notrain") and args[0].notrain:
                return func(*args, **kwargs)
            else:
                mlflow.set_experiment(mlflow_config.experiment_name)

                # print("run_name:", run_name)
                with mlflow.start_run(run_name=mlflow_config.run_name):
                    mlflow.set_tags(reproducibility_tags())
                    mlflow.set_tag("repro.run_status", "RUNNING")
                    if len(args):
                        params = _flatten(args[0])
                        params.pop("mlflow.tracking_uri", None)
                        params.pop("mlflow.experiment_name", None)
                        params.pop("mlflow.run_name", None)
                        mlflow.log_params(params)
                    try:
                        result = func(*args, **kwargs)
                    except BaseException as error:
                        mlflow.set_tag("repro.run_status", "FAILED")
                        mlflow.set_tag("repro.failure_type", type(error).__name__)
                        raise
                    mlflow.set_tag("repro.run_status", "COMPLETED")
                    return result

        return inner2

    return inner1
