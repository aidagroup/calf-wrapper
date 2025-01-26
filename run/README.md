The `run` directory contains the code for running the experiments and reproducing the results.

The folder is organized as follows:

- `scripts`: the scripts for running the experiments
    - `eval_calf_wrapper.py`: the script for evaluating the experiments with the CALF wrapper
    - `eval_checkpoints.py`: the script for evaluating the checkpoints without the CALF wrapper
- `eval.py`: the script for evaluating the experiments
- `train_ppo.py`: the script for training the PPO agent

## Running the experiments

### Pendulum

```sh
python train_ppo.py --env-id=Pendulum-v1
python eval.py pendulum --n-envs=100 --eval-mode=fallback --mlflow.experiment-name=eval/pendulum/fallback
python scripts/eval_calf_wrapper.py pendulum
python scripts/eval_checkpoints.py pendulum
```

We note that `python train_ppo.py --env-id=Pendulum-v1` will save the checkpoints in the `artifacts` directory.

The `python scripts/eval_calf_wrapper.py pendulum` will evaluate the checkpoints in the `artifacts` directory with the CALF wrapper.

The `python scripts/eval_checkpoints.py pendulum` will evaluate the checkpoints in the `artifacts` directory without the CALF wrapper with only the PPO agent.

### Cartpole

```sh
python train_ppo.py --env-id=CartpoleSwingupEnvLong-v0
python eval.py cartpole --n-envs=100 --eval-mode=fallback --mlflow.experiment-name=eval/cartpole/fallback
python scripts/eval_calf_wrapper.py cartpole
python scripts/eval_checkpoints.py cartpole
```

## MlFlow

We use MlFlow for logging the experiments. Everything is stored in the `mlruns` directory. Run the following command to view the experiments:
```sh
mlflow ui
```


## Notes

We use tyro for parsing the arguments. 