# optcdf/tuner.py

import subprocess
import optuna
from typing import Callable, Dict, Optional, Any, Literal
import mlflow
from pathlib import Path
import uuid


class Tuner:
    def __init__(
        self,
        script: str,
        metric: str,
        mlflow_exp_name: str,
        subcommands: Optional[list[str]] = None,
        mlflow_tracking_uri: Optional[str] = None,
        direction: Literal["minimize", "maximize"] = "minimize",
        pruner: Optional[optuna.pruners.BasePruner] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        params_fn: Optional[Callable[[optuna.Trial], Dict[str, Any]]] = None,
    ):
        """Initialize the tuner.

        Args:
            script: Path to the training script
            metric: Name of the metric to optimize
            mlflow_exp_name: Name of the MLflow experiment for the tuner
            mlflow_tracking_uri: MLflow tracking URI. If None, use the default tracking URI from the script's parent directory.
            direction: Direction of optimization ("minimize" or "maximize")
            pruner: Optuna pruner instance
            sampler: Optuna sampler instance
            study_name: Name of the study
            storage: Storage URL for the study
            params_fn: Function that takes a trial and returns parameters
        """
        self.script = Path(script)
        self.subcommands = subcommands
        self.metric = metric
        self.mlflow_exp_name = mlflow_exp_name
        if mlflow_tracking_uri is None:
            self.mlflow_tracking_uri = "file://" + str(
                self.script.parent.absolute() / "mlruns"
            )
            print(f"Using default MLflow tracking URI: {self.mlflow_tracking_uri}")
        else:
            self.mlflow_tracking_uri = mlflow_tracking_uri
        self.direction = direction
        self.pruner = pruner or optuna.pruners.MedianPruner()
        self.sampler = sampler or optuna.samplers.TPESampler()
        self.study_name = study_name or "optimization"
        self.storage = storage
        self.params_fn = params_fn

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        # Get parameters for this trial

        params = self.params_fn(trial)
        # Convert parameters to command line arguments
        cmd = ["python", str(self.script)]
        if self.subcommands is not None:
            cmd.extend(self.subcommands)
        for k, v in params.items():
            cmd.extend([f"--{k}", str(v)])
        cmd.extend(["--mlflow.tracking_uri", self.mlflow_tracking_uri])
        cmd.extend(["--mlflow.experiment_name", self.mlflow_exp_name])
        run_name = (
            self.study_name
            + "-trial-"
            + str(trial.number)
            + "-"
            + str(uuid.uuid4())[:8]
        )
        cmd.extend(["--mlflow.run_name", run_name])
        # Run the training script
        try:
            subprocess.run(cmd, check=True)

            # Get the metric value from MLflow
            client = mlflow.tracking.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
            runs = client.search_runs(
                experiment_ids=[
                    mlflow.get_experiment_by_name(self.mlflow_exp_name).experiment_id
                ],
                filter_string=f"tags.mlflow.runName = '{run_name}'",
                max_results=1,
            )

            if not runs:
                raise RuntimeError("No MLflow runs found")

            metric_value = runs[0].data.metrics.get(self.metric)
            if metric_value is None:
                raise RuntimeError(f"Metric {self.metric} not found in MLflow run")

            return metric_value

        except subprocess.CalledProcessError as e:
            raise optuna.TrialPruned(
                f"Training script failed with exit code {e.returncode}"
            )

    def tune(
        self,
        num_trials: int = 100,
        timeout: Optional[int] = None,
    ) -> optuna.Study:
        """Run the hyperparameter optimization.

        Args:
            num_trials: Number of trials to run
            num_seeds: Number of random seeds to try for each parameter set
            timeout: Time limit in seconds for the optimization

        Returns:
            The completed Optuna study
        """

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            pruner=self.pruner,
            sampler=self.sampler,
            load_if_exists=True,
        )

        study.optimize(
            self._objective,
            n_trials=num_trials,
            timeout=timeout,
        )

        return study
