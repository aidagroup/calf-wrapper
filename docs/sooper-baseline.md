# SOOPER baseline protocol

This repository adapts **SOOPER: Safe Online Optimism for Pessimistic Expansion
in RL** (Wendl et al., ICLR 2026) to the CALF-Wrapper Gymnasium environments.
The primary references are the [paper](https://arxiv.org/abs/2601.19612), the
[project page](https://yardenas.github.io/sooper/), and the
[official implementation](https://github.com/lasgroup/safe-learning) at commit
`76fa2f8f576e60a4417227793dd162f031ba89be`.

## Algorithm correspondence

- `SOOPERSafetyFilter` implements Algorithm 1.  It tracks realized discounted
  costs and accepts the learned action only when accumulated cost plus the
  ensemble-pessimistic prior cost-to-go is below the fixed budget.
- `ProbabilisticEnsemble` models state deltas, rewards, and costs.  Bootstrap
  disagreement estimates epistemic uncertainty.
- `PriorValueEnsemble` performs fitted policy evaluation of the conservative
  controller and provides the pessimistic backup reward and cost heads
  (`Qr`/`Qc`) used by the official implementation.
- Model-generated rollouts terminate whenever the same cost criterion would
  invoke the prior in the real environment.  The terminal reward is the
  pessimistic predicted reward-to-go of the prior.
- Exploration and safe-set expansion are represented by separate uncertainty
  coefficients, `lambda_explore` and `lambda_expand`, in the planning reward.
- A TD3 actor--critic is optimized on the resulting planning MDP and real/model
  replay, following the MBPO-style practical implementation in the paper.
- The existing CALF fallback controller is the conservative policy prior.  The
  learned actor is initialized by distilling the selected bare-backbone
  checkpoint on prior-rollout states for PPO.  For the CleanRL TD3 tasks, the
  matching actor and twin-critic architectures permit an exact state copy, so
  online learning starts bit-for-bit from the selected bare backbone.

## Deliberate implementation-level differences

The official release uses JAX/Brax and its own SAC/MBPO stack.  The adaptation
uses PyTorch, Gymnasium, and a TD3 planner so all three methods share the exact
CALF environments, initial-state generators, horizons, rewards, and controller
interfaces.  Ensemble members use jointly learned reward/cost heads instead of
the Brax transition structure.  Cost budgets are calibrated on explicitly
stored prior rollouts because the CALF fallback controllers provide a
goal-reaching guarantee, not the calibrated probabilistic CMDP model assumed
by SOOPER's theorem.  Consequently, experiments report an empirical
constraint-satisfaction rate and do **not** transfer SOOPER's theorem to these
tasks.

These are engineering substitutions, not removal of SOOPER's defining
mechanisms.  A budget-only shield without the ensemble, model planning,
uncertainty bonuses, terminating planning MDP, and prior terminal value must
not be labeled SOOPER in this repository.

## Reproducible execution

All runs start from committed source and fixed model checkpoints.  A screening
matrix is distributed deterministically after a seeded shuffle:

```bash
uv run python scripts/run_sooper_matrix.py \
  --matrix configs/sooper/smoke.json \
  --result-root run/artifacts/sooper/smoke \
  --model-root /mnt/raid0/calf-eval-wrapper \
  --worker-index 0 --worker-count 1 --shuffle-seed 20260720 --device cuda:0
```

Matrix checkpoint paths may be relative to `--model-root`; local and gor
workers can therefore consume the same committed matrix while resolving
checkpoints from their respective artifact mirrors.
For persistent multi-GPU execution, `scripts/launch_sooper_workers.py` records
the exact commit, matrix hash, deterministic global shard indices, devices,
commands, tmux sessions, and log paths in a launch manifest.

Interrupted tasks resume from their newest local checkpoint.  Each checkpoint
contains world-model, actor/critics, optimizers, real and model replay buffers,
all random-number-generator states, progress rows, and the resolved budget.
Checkpoint and final result batches are uploaded to MLflow and downloaded
again for size and SHA-256 verification.

Held-out evaluation uses explicit seeds and never modifies the checkpoint:

```bash
uv run python run/eval_sooper.py \
  --checkpoint RUN/checkpoints/sooper_checkpoint_000050.pt \
  --seeds 81001,81002,81003,81004,81005 \
  --device cuda:0 --output-dir RUN/held-out \
  --tracking-uri http://192.168.1.5:5001 \
  --experiment-name calf-wrapper/sooper/held-out \
  --run-name ENV-CHECKPOINT-held-out
```

Bare-backbone, fallback, and CALF controls are evaluated with the same explicit
seed list and resolved cost budget through `run/eval_comparison_controls.py`.

For a frozen evaluation-time reward stress test, pass the same non-default
underwater penalty to every method without retraining or changing the cost
budget:

```bash
uv run python scripts/run_frozen_heldout.py \
  --protocol RUN/frozen_held_out_protocol.json \
  --screening-root RUN/screening-v1 \
  --model-root /mnt/raid0/calf-eval-wrapper \
  --output-root RUN/held-out-penalty50-v1 \
  --device cuda:0 --underwater-intrusion-penalty 50 \
  --tracking-uri http://192.168.1.5:5001 \
  --experiment-name calf-wrapper/sooper/held-out-penalty50-v1
```

The raw trial tables record the undiscounted intrusion-step count, so the
changed return can be audited independently of the discounted SOOPER cost.

The complete screening table is retained.  Environment/checkpoint/mode
selection is performed only on screening results; held-out seeds are used once
for confirmation.

Two repeated runs are checked with:

```bash
uv run python scripts/compare_sooper_reproducibility.py RUN_A RUN_B
```

The check requires identical raw CSV files, stable summary fields, network and
optimizer tensors, replay buffers, RNG states, and accumulated progress.  The
same comparison is used between an uninterrupted run and a resumed run.
