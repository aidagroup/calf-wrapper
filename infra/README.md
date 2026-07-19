# CALF-Wrapper tracking stack

This stack is dedicated to CALF-Wrapper and does not share containers, ports,
storage, databases, buckets, or networks with CALF-Enhance.

On `gor`, create the untracked configuration and start the services:

```bash
cp infra/.env.example infra/.env
# Replace both example passwords in infra/.env.
docker compose --env-file infra/.env -f infra/docker-compose.yml config --quiet
docker compose --env-file infra/.env -f infra/docker-compose.yml up -d --build
docker compose --env-file infra/.env -f infra/docker-compose.yml ps
```

The server itself uses `http://127.0.0.1:5001` as the tracking URI. Other
machines on the private network use `http://192.168.1.5:5001`. MinIO is
available at ports `9030` (S3 API) and `9031` (console); experiment clients use
MLflow's artifact proxy and therefore do not need MinIO credentials.

The real `infra/.env` and persistent `.dockerdata/` directory are ignored by
Git. The example credentials are deliberately unusable placeholders.

After the tracking stack is healthy, preview the new two-environment TD3
matrix from the checked-out experiment commit:

```bash
uv run python scripts/run_td3_matrix.py \
  --tracking-uri http://127.0.0.1:5001 \
  --gpus 0,1 \
  --dry-run
```

The default matrix contains 20 runs: seeds 0--9 for `UnderwaterDrone-v0` and
seeds 1--10 for `RobotNavigationConstSpeedCatch-v0`. Without `--dry-run`, the
launcher creates one detached `tmux` queue per GPU. Each GPU runs one job at a
time and processes its assigned seeds sequentially, while both GPU queues run
in parallel and survive SSH disconnects. Before committing GPU time, run the
two-job smoke matrix:

```bash
uv run python scripts/run_td3_matrix.py \
  --tracking-uri http://127.0.0.1:5001 \
  --gpus 0,1 \
  --seeds 0 \
  --smoke
```

The launcher delegates every job to the CALF-Enhance source tree copied under
`vendor/calf-enhance-td3` and uses its frozen dependency lock. Periodic and
final checkpoints are stored locally under `run/artifacts/` and uploaded to
each MLflow run under `checkpoints/`. The wrapper PPO environment is not
reused for TD3, so dependency changes cannot perturb the published PPO
reproduction. The smoke path keeps the historical `learning_starts=25000`
and therefore reproduces an exact 1,000-step prefix rather than introducing a
different short-run training schedule.
