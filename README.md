# CALF-Wrapper

This is the companion repository for the article *A Universal Policy Wrapper
With Guarantees*. It contains the CALF-Wrapper implementation, the published
model checkpoints and evaluation data, and the code used to reproduce the
experimental figures and tables.

CALF-Wrapper combines a pretrained reinforcement-learning base policy with a
predefined fallback policy during deployment.

## Setup

Install [uv](https://docs.astral.sh/uv/) and Git LFS, then run:

```sh
git lfs install
git lfs pull
uv sync --frozen
```

## Evaluation

Run the complete evaluation and regenerate the article figures and tables:

```sh
uv run calfwrapper eval main
```

Individual environments can also be evaluated separately:

```sh
uv run calfwrapper eval pendulum
uv run calfwrapper eval cartpole
uv run calfwrapper eval auv
uv run calfwrapper eval robot
```

Evaluation outputs are written to `outputs/evaluation/`. Published evaluation
data are stored in `reference/`, generated figures in `figures/`, and generated
tables in `tables/`.

## Training

Run any of the available training configurations:

```sh
uv run calfwrapper train pendulum-ppo
uv run calfwrapper train cartpole-ppo
uv run calfwrapper train auv-td3
uv run calfwrapper train robot-td3

uv run calfwrapper train pendulum-ppo-lagrangian
uv run calfwrapper train cartpole-ppo-lagrangian
uv run calfwrapper train auv-td3-lagrangian
uv run calfwrapper train robot-td3-lagrangian
```

Run all configurations with:

```sh
uv run calfwrapper train all
```

Training runs and their artifacts are written to `outputs/training/`.

## Published checkpoints

The checkpoints used for the article evaluation are stored in
`artifacts/checkpoints/` and tracked with Git LFS.

## Repository structure

```text
calfwrapper/
  environments/       CartPole, AUV, and robot environments
  fallback/            Fallback policies for all four environments
  models/              Policy and critic model loading
  training/            PPO, TD3, and Lagrangian training implementations
  cli.py                Public train and evaluation commands
  experiments.py        Article environment configurations
artifacts/checkpoints/
  base/                 Base-policy checkpoints by environment and stage
  lagrangian/           Lagrangian checkpoints by environment and stage
reference/              Published evaluation data
figures/                Reproduced article figures
tables/                 Reproduced article tables
tests/                  Automated validation
outputs/                Training and evaluation outputs
```
