# CALF-Wrapper

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="gfx/logo.png" alt="CALF-Wrapper Logo" width="400">
</p>

Implementation of the paper "Unviversal policy wrapper with guarantees".



## Overview

CALF-Wrapper provides a universal policy wrapper with formal guarantees. The repository contains:

- Implementation of the CALF wrapper algorithm
- Controllers for pendulum and cartpole tasks
- Training and evaluation scripts
- Reproduction code for paper experiments

## Project Structure

```
.
├── src/                    # Core implementation
│   ├── calf_wrapper.py    # Main CALF wrapper implementation
│   ├── callback.py        # Reward collection callbacks
│   ├── controllers/       # Task-specific controllers
│   ├── envs/             # Environment implementations
│   └── utils/            # Utility functions
├── run/                   # Training and evaluation scripts
│   ├── train_ppo.py      # PPO training script
│   ├── eval.py           # Evaluation script
│   └── scripts/          # Additional experiment scripts
└── reproduce/            # 
    ├── cartpole
    └── pendulum

```

## Installation

1. Install uv package manager:
```sh
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or follow the guide at https://docs.astral.sh/uv/getting-started/installation/
```

2. Create virtual environment:
```sh
uv venv --python=3.13
```

3. Install dependencies:
```sh
uv sync

# If the above fails:
rm -rf uv.lock
uv pip install -r pyproject.toml
```

## Usage

### Training

Train PPO agents on supported environments:
```sh
uv run run/train_ppo.py pendulum
uv run run/train_ppo.py cartpole
```

### Evaluation

Run experiments and reproduce paper results:
```sh
# See run/README.md for detailed instructions
uv run run/eval.py
```

Actually we have prepared for you a list of reproducable bash scripts that fully reproduce the plots from the original paper

### Tests

Run the test suite:
```sh
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
