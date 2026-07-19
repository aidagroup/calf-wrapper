# Vendored CALF-Enhance TD3 runtime

This directory is a tracked source copy of
`aidagroup/calf-enhance@afb5edc49427054c99d6fbfe87b603d126724eb8`.
Its `uv.lock` SHA-256 is
`26812bc65b4f091bf16da07e10b7d67c9ae21ccc9d4432704795da6850055f40`.

CALF-Wrapper intentionally keeps this runtime separate from its PPO virtual
environment. `uv run --project vendor/calf-enhance-td3 --frozen` reconstructs
the exact TD3 dependency environment from the tracked lock file.

The local integration patch makes two scoped changes:

- the MLflow helper permits execution from a copied directory when
  `MLFLOW_DISABLE_GIT=1`;
- the TD3 trainer writes periodic and final model checkpoints and uploads them
  to the active MLflow run under `checkpoints/`.

The environment dynamics and TD3 update logic remain those of the pinned
source revision.
