# CALF-Wrapper

CALF-Wrapper is a deployment-time policy wrapper that combines a pretrained
reinforcement-learning base policy with a predefined fallback policy. This
repository contains the implementation, fixed training configurations,
published checkpoints, reference trial records, and generators for every
experimental figure and table reported in the article.

## Setup

Install [uv](https://docs.astral.sh/uv/) and Git LFS, then run:

```sh
git lfs install
git lfs pull
uv sync --frozen
```

Checkpoint files are verified automatically before evaluation.

## Exact evaluation reproduction

Run one environment or the complete article evaluation:

```sh
uv run calfwrapper eval pendulum
uv run calfwrapper eval cartpole
uv run calfwrapper eval auv
uv run calfwrapper eval robot
uv run calfwrapper eval main
```

The published evaluation used CUDA inference. The CLI therefore defaults to
`--device cuda:0`. A different CUDA device can be selected explicitly:

```sh
uv run calfwrapper eval main --device cuda:1
```

Each command uses the complete published evaluation protocol. The protocol is
fixed by the selected environment and requires no additional arguments.
The complete command evaluates all four environments, all three checkpoint
stages, all six CALF-Wrapper operating modes, the base and fallback policies,
and the corresponding Lagrangian baselines. It then runs the three sensitivity
and robustness studies and regenerates Figures 2--5 and Tables 6--11.

The 10,000 central and Lagrangian evaluation trials are compared field by field
with the published trial records. A mismatch makes the command fail. Generated
trial records, summaries, verification report, figures, and tables are written
below `outputs/evaluation/`.

The article figures in `figures/` and the generated LaTeX tables in `tables/`
are tracked as reference outputs. Figure PDFs are regular Git objects. Only the
model checkpoints use Git LFS.

## Training

Training uses fixed named configurations. Each command uses the complete
article training budget and the same training seed as the checkpoints used by
the evaluation: Pendulum PPO seed 6, CartPole PPO seed 45, AUV TD3 seed 0,
Robot TD3 seed 2, Pendulum PPO-Lagrangian seed 10, CartPole PPO-Lagrangian
seed 42, AUV TD3-Lagrangian seed 4, and Robot TD3-Lagrangian seed 1.

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

Run all eight configurations with `uv run calfwrapper train all`. Use
`--dry-run` to print the resolved command or `--smoke` to execute a minimal
training update and checkpoint save to validate the local installation.

## Repository layout

```text
artifacts/checkpoints/  Published checkpoints tracked with Git LFS
calfwrapper/            CLI, evaluation, training, figures, tables, and checks
calfwrapper/training/   Fixed implementations used by the training commands
figures/                Article Figures 2--5, tracked directly by Git
reference/trials/       Published trial records used for exact verification
reference/studies/      Published sensitivity and robustness results
reference/runtime/      Raw native-runtime measurements used for Table 9
src/                    Environments, fallback controllers, and policy models
tables/                 Generated article Tables 6--11
vendor/                 Self-contained TD3 implementation used by the project
tests/                  Automated tests
outputs/                Generated training and evaluation outputs (ignored)
```

Table 9 is regenerated from the committed raw measurements because its latency
values are specific to the reported NVIDIA GeForce RTX 3090 hardware run. The
other tables and all four experimental figures are generated from evaluation
records produced by the public command above.

The manuscript itself is intentionally not stored in this repository.
