# Source code for the paper "CALF: evalution enhacement for model-free RL with stability guarantees"

## Overview

This code is the source code for the paper "CALF: evalution enhacement for model-free RL with stability guarantees".

The code is organized into the following directories:

- `src`: the source code for the CALF wrapper and the experiments
    - `callback.py`: the callback for rewards collecting
    - `calf_wrapper.py`: the source code for the CALF wrapper
    - `controllers`: the source code for the controllers
        - `cartpole.py`: source code for the cartpole controller which is used for the cartpole swingup task
        - `pendulum.py`: source code for the pendulum controller which is used for the pendulum swingup task
    - `envs`: the source code for the environments
        - `cartpole.py`: source code for the cartpole environment for the swingup task
    - `utils`: the source code for the utils
- `run`: the code for running the experiments and reproducing the results
    - `eval.py`: the script for evaluating the experiments
    - `train_ppo.py`: the script for training the PPO agent
    - `scripts`: the scripts for running the experiments
        - `eval_calf_wrapper.py`: the script for evaluating the experiments with the CALF wrapper
        - `eval_checkpoints.py`: the script for evaluating the checkpoints without the CALF wrapper
- `tests`: the tests for the code for the CALF wrapper and controllers

## Setup

clone the repo and install the dependencies via

```sh
pip install -e .
```

## Run experiments and reproduce the results

All the run files are in the `run` directory. Visit the directory for more information. There you will find
the readme with the instructions for running the experiments.

## Tests

The tests are in the `tests` directory. Run them with

```sh
pytest
```

## License

This code is licensed under the MIT license. See the LICENSE file for more details.
