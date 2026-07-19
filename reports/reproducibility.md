# CALF-Wrapper reproducibility report

Date: 2026-07-19

## Executive verdict

The published CALF-Wrapper evaluation is exactly reproducible from the archived
PPO checkpoints on both a local workstation and the independent `gor` server.
With an absolute tolerance of zero, all 42 reproduced values per environment
(three checkpoint indices plus reward mean, reward standard deviation, and goal
rate for fallback/base/conservative/balanced/brave) match the published values.
The two plotting JSON files contain 51 numeric leaves each; all 51 match exactly
when the nine archived Residual RL values are included. Both PDFs generated from
the remote replay export are byte-for-byte identical to the archived PDFs.

Fresh single-seed PPO training is not numerically identical to the archived
training. This is reported separately and is not concealed by a post-hoc
tolerance.

| Environment | Archived-checkpoint replay | Fresh PPO retraining | Interpretation |
|---|---|---|---|
| Pendulum-v1 | **PASS**, 42/42 exact | **FAIL** for full result reproduction | Conservative CALF still reaches the goal in 100%, 100%, and 93.33% of early/mid/late trials while the fresh base reaches 0% at all stages, but balanced/brave late-stage behavior and reward distributions do not reproduce. |
| CartPoleSwingUpEnv | **PASS**, 42/42 exact | **WARNING** | Conservative CALF retains 100% goal reaching at all stages and the late policy behavior is qualitatively recovered, but the reward distributions and several balanced/brave goal rates differ. |

The strict zero-tolerance comparison labels both fresh-training environments
`FAIL`; the `WARNING` above is a scientific interpretation of CartPole's
qualitative recovery, not a claim of numerical equality. A multi-training-seed
sensitivity analysis is required before making a stronger statement about PPO
training-distribution reproducibility.

## Source and code provenance

The checkout was reset to the canonical `origin/main` before any fix.

| Purpose | Commit |
|---|---|
| Original immutable baseline | `967b474e673fda5f85a5da385df8ef9a3db4eac9` |
| Reproduction workflow and isolated infrastructure | `19427728b5e29c5735dd8902a95229b5d46c6868` |
| Active-interpreter launcher fix | `80c619d45df66115eafc26ab651617b065debb7d` |
| Documented CartPole 300,000-step horizon | `270d24ed3abb5847a755a432ef33f00ebdba62db` |
| Isolated reference-replay namespace; experiment-producing replay SHA | `48cb203552e99909134c6af64a420140369df1d6` |
| Complete raw export, comparison, and deterministic plot evidence | `5aa8ec7` |

Branch: `feat/reproducible-remote-experiments`.
Every experiment-producing commit was pushed before execution, and every full
or replay MLflow run reports `repro.git_dirty=False`.

## Clean-baseline audit and diagnosed failures

On exact `origin/main`:

- `uv sync --frozen` installed the locked environment.
- The original test suite passed: 6 tests.
- Base, fallback, and CALF-Wrapper evaluation entry points executed.
- PPO training could not start because `run/train_ppo.py` referenced undefined
  `current_dir`; the preserved failure was `NameError: name 'current_dir' is not
  defined`.
- Evaluation with the archived checkpoints did not reproduce the published
  metrics because the baseline had changed the historical rollout from
  `range(n_steps)` to `range(n_steps - 1)` and converted actions to float64.
- Restoring the published rollout length and action dtype made the checkpoint
  evaluations exact. The fixes also made PPO's effective rollout horizon
  explicit (`n_steps=2048`), matching the historical Stable-Baselines3 default.

The final suite contains 8 tests and passes on both local and remote checkouts.

## Environments

| Item | Local (`gvidon`) | Remote (`gor`) |
|---|---|---|
| OS/kernel | Ubuntu 24.04, Linux 7.0.0-28 | Ubuntu 24.04, Linux 6.17.0-23 |
| CPU | AMD Threadripper 3990X, 64 cores/128 threads | AMD Ryzen 9 5950X, 16 cores/32 threads |
| RAM | 125 GiB | 125 GiB |
| GPU | 2 x RTX 3090, 24 GiB | GTX 1080 Ti, 11 GiB |
| NVIDIA driver | 580.159.03 | 580.105.08 |
| Python | 3.13.2 | 3.13.2 |
| uv | 0.6.6 | 0.11.29 |
| PyTorch / CUDA runtime | 2.5.1+cu124 / 12.4 | 2.5.1+cu124 / 12.4 |
| Stable-Baselines3 | 2.3.2 | 2.3.2 |
| Gymnasium | 0.29.0 | 0.29.0 |
| MLflow | 2.20.0 | 2.20.0 |
| NumPy | 2.1.3 | 2.1.3 |

PyTorch detects CUDA on both machines. The documented presets intentionally
train and evaluate on CPU, so the device parameter recorded by the production
runs is `cpu`.

## Isolated remote infrastructure

The wrapper stack runs at `/mnt/raid0/calf-eval-wrapper` with dedicated
PostgreSQL, MinIO, and MLflow containers and persistent bind storage.

| Service | Port(s) | Final health |
|---|---:|---|
| CALF-Wrapper MLflow | 5001 | healthy; `/health` returns `OK` locally and remotely |
| CALF-Wrapper MinIO | 9030, 9031 | healthy |
| CALF-Wrapper PostgreSQL | internal 5432 | healthy |
| Existing CALF-Enhance MLflow | 5000 | healthy; `/health` returns `OK` |
| Existing CALF-Enhance MinIO | 9010, 9011 | remained running |

The stacks do not share ports, containers, networks, database storage, or
buckets. A local-to-remote smoke run (`290adb0e790c4bf6a380cb6025b96078`)
logged and downloaded an artifact successfully; the downloaded artifact SHA-256
was `95e0ecf3e76731215ad6eb55caed46975f231abcecc8518beb2ec4d2b2ec8ab9`.

## Commands

Validation and deployment:

```bash
uv sync --frozen
uv run python -m pytest
uv run python -m compileall -q run scripts src tests publication/plots.py
uv run python -m black --check run scripts src tests publication/plots.py
uv run python scripts/run_reproduction.py \
  --tracking-uri http://127.0.0.1:5001 \
  --artifact-root artifacts/reproduction \
  --dry-run
docker compose --env-file infra/.env -f infra/docker-compose.yml config --quiet
docker compose --env-file infra/.env -f infra/docker-compose.yml up -d --build
```

Fresh full workload on `gor`:

```bash
uv run python scripts/run_reproduction.py \
  --tracking-uri http://127.0.0.1:5001 \
  --artifact-root artifacts/reproduction \
  --environment all
```

Archived-checkpoint replay on `gor`:

```bash
uv run python scripts/run_reproduction.py \
  --tracking-uri http://127.0.0.1:5001 \
  --artifact-root artifacts/reference-checkpoints \
  --experiment-prefix calf-wrapper/reference-replay \
  --environment all \
  --skip-training
```

Export and strict comparison from the local machine:

```bash
uv run python scripts/export_results.py \
  --tracking-uri http://192.168.1.5:5001 \
  --experiment-prefix calf-wrapper/reference-replay \
  --output-dir reports/results/reference-replay \
  --reference-dir reference-results
uv run python scripts/compare_results.py \
  --reference-dir reference-results \
  --actual-dir reports/results/reference-replay \
  --output reports/results/reference-replay/comparison-zero.json \
  --absolute-tolerance 0
cd publication
uv run python plots.py \
  --data-dir ../reports/results/reference-replay \
  --output-dir /tmp/calf-wrapper-reference-plots
```

## MLflow experiments and principal runs

| Experiment | Relevant run(s) |
|---|---|
| `calf-wrapper/reproduction/train/pendulum` | full train `8617bdc7e5ae44858474f1d2d298e853` |
| `calf-wrapper/reproduction/train/cartpole` | full train `313c2afec03c40e78fc1cbd9f112c873` |
| `calf-wrapper/reproduction/eval/pendulum` | 13 selected completed runs in `fresh-training/runs.csv` |
| `calf-wrapper/reproduction/eval/cartpole` | 13 selected completed runs in `fresh-training/runs.csv` |
| `calf-wrapper/reference-replay/eval/pendulum` | 13 completed runs in `reference-replay/runs.csv` |
| `calf-wrapper/reference-replay/eval/cartpole` | 13 completed runs in `reference-replay/runs.csv` |

Pendulum requested 102,000 timesteps and completed the final PPO rollout at
102,400, producing 34 checkpoints through 102,000. CartPole requested 300,000
and completed at 301,056, producing 100 checkpoints through 300,000.

One earlier CartPole run, `d6994b6999f6465eabd755f3d8d1c4e3`, was
intentionally interrupted after discovering that the launcher incorrectly used
270,000 as the training horizon. It is preserved as `FAILED` / `FAILED` with
`repro.failure_type=KeyboardInterrupt`; it was not deleted or selected by the
exporter. The corrected full run is the one listed above.

All 26 replay evaluations are `FINISHED` / `COMPLETED`, report hostname `gor`,
commit `48cb203552e99909134c6af64a420140369df1d6`, branch
`feat/reproducible-remote-experiments`, `repro.git_dirty=False`, and 30 trials.
Every run contains `raw/trial_metrics.csv` and
`episode_data/episode_data.json` in MLflow.

## Machine-readable evidence

Each result directory contains:

- `runs.csv`: 26 selected aggregate evaluation rows with run IDs, status,
  artifact URI, Git SHA, hostname, mean, standard deviation, 95% confidence
  interval half-width, goal rate, trial count, and wrapper utilization;
- `trials.csv`: 780 individual trials (26 runs x 30 trials) with reward and goal
  outcome;
- `pendulum.json` and `cartpole.json`: plotting aggregates without numerical
  rounding;
- `comparison-zero.json`: leaf-level comparison against the published values;
- `provenance.json`: selection policy and external-baseline provenance.

Evidence directories:

- [`results/reference-replay`](results/reference-replay)
- [`results/fresh-training`](results/fresh-training)

The archived Residual RL values originate from CALF-Enhance. This repository
contains neither that method's implementation nor its checkpoints. The exporter
carries those nine values through only to reconstruct the published comparison
plots, records that fact in `provenance.json`, and excludes them from the
CALF-Wrapper reproducibility verdict. This prevents an imported baseline from
being misrepresented as a reproduced run.

## Fresh-training results

All values below are direct MLflow exports; no display rounding is applied to
reward statistics. Wrapper utilization is shown as base/fallback action percent.
Every row uses 30 evaluation trials.

### Pendulum-v1

| Run | Mean reward | Std | 95% CI half-width | Goal % | Base / fallback action % |
|---|---:|---:|---:|---:|---:|
| `fallback` | -329.90240478515625 | 285.9991455078125 | 102.34348429297397 | 100.0 | — |
| `base_early` | -902.5858154296875 | 241.8598175048828 | 86.54842694951233 | 0.0 | — |
| `calf_wrapper_conservative_early` | -443.04412841796875 | 159.61752319335938 | 57.11839644178325 | 100.0 | 14.85 / 85.15 |
| `calf_wrapper_balanced_early` | -704.3568725585938 | 146.69476318359375 | 52.49404549362806 | 10.0 | 55.32 / 44.68 |
| `calf_wrapper_brave_early` | -897.44921875 | 252.9197540283203 | 90.50617150827873 | 0.0 | 94.85 / 5.15 |
| `base_mid` | -431.9053955078125 | 215.0899200439453 | 76.96894306270343 | 0.0 | — |
| `calf_wrapper_conservative_mid` | -390.7264099121094 | 161.22052001953125 | 57.69202193477264 | 100.0 | 14.48 / 85.52 |
| `calf_wrapper_balanced_mid` | -530.2777099609375 | 146.59059143066406 | 52.456770673582994 | 60.0 | 55.78 / 44.22 |
| `calf_wrapper_brave_mid` | -495.2796630859375 | 168.25184631347656 | 60.208150291921896 | 0.0 | 94.97 / 5.03 |
| `base_late` | -305.6617736816406 | 401.23345947265625 | 143.57955350992552 | 0.0 | — |
| `calf_wrapper_conservative_late` | -455.4114074707031 | 184.25714111328125 | 65.93557404247781 | 93.33333333333333 | 14.22 / 85.78 |
| `calf_wrapper_balanced_late` | -681.5042724609375 | 59.66505813598633 | 21.350867878247307 | 0.0 | 56.33 / 43.67 |
| `calf_wrapper_brave_late` | -524.4652099609375 | 204.6350555419922 | 73.22771562007321 | 0.0 | 95.10 / 4.90 |

### CartPoleSwingUpEnv

| Run | Mean reward | Std | 95% CI half-width | Goal % | Base / fallback action % |
|---|---:|---:|---:|---:|---:|
| `fallback` | -1068.7037353515625 | 787.009521484375 | 281.62774659404334 | 100.0 | — |
| `base_early` | -2817.0419921875 | 287.72125244140625 | 102.95972788738707 | 0.0 | — |
| `calf_wrapper_conservative_early` | -1163.4124755859375 | 917.3973388671875 | 328.2864385106442 | 100.0 | 6.99 / 93.01 |
| `calf_wrapper_balanced_early` | -2295.6416015625 | 1305.447509765625 | 467.148415020248 | 46.666666666666664 | 49.57 / 50.43 |
| `calf_wrapper_brave_early` | -2750.105224609375 | 516.9437255859375 | 184.9859391243565 | 0.0 | 90.81 / 9.19 |
| `base_mid` | -1943.359130859375 | 191.4663848876953 | 68.5153648949686 | 0.0 | — |
| `calf_wrapper_conservative_mid` | -1091.8323974609375 | 840.358642578125 | 300.7184942037401 | 100.0 | 10.73 / 89.27 |
| `calf_wrapper_balanced_mid` | -2082.80029296875 | 1353.7532958984375 | 484.4343927331263 | 66.66666666666666 | 49.90 / 50.10 |
| `calf_wrapper_brave_mid` | -2587.516357421875 | 331.4922790527344 | 118.62298614741584 | 0.0 | 90.84 / 9.16 |
| `base_late` | -697.6286010742188 | 552.4442138671875 | 197.68962124658333 | 100.0 | — |
| `calf_wrapper_conservative_late` | -958.3067626953125 | 769.2614135742188 | 275.2766744305631 | 100.0 | 13.62 / 86.38 |
| `calf_wrapper_balanced_late` | -1032.1826171875 | 909.6451416015625 | 325.5123449975508 | 100.0 | 51.37 / 48.63 |
| `calf_wrapper_brave_late` | -683.9425048828125 | 497.32501220703125 | 177.96546913877557 | 76.66666666666667 | 90.99 / 9.01 |

## Exact checkpoint and publication-artifact checks

Archived checkpoint SHA-256 values were identical after transfer to `gor`:

| Checkpoint | SHA-256 |
|---|---|
| Pendulum 30k | `bb44014761c44956a354a05bbf8e1add7ca2dc980174706187a1aeb627386645` |
| Pendulum 36k | `7936039c6c0b8b376d206dc7123f8e43ea5953d31d12aeb5a27874820dc7c25f` |
| Pendulum 102k | `39be8cdb2b783b952238722b34262598e08d4c6d75875b9192589d7f7d156a45` |
| CartPole 99k | `07b621a181cf63ce524e4b4527131b253f1011cc6b57b08b4c7ad48608852df6` |
| CartPole 108k | `0dc0fa301ed99a1435ab811fc2dfaad3ea2428721b2434beb40252abd3fb9528` |
| CartPole 270k | `6ad1504cc0ec7fe9160e7fbb11aeaef62aa010b34adbb0e228fd664876a3b6c6` |

| Artifact | Result | SHA-256 |
|---|---|---|
| Pendulum plotting JSON | 51/51 numeric leaves exact | `bae3a5bbb7bc0063d757f76d5781859dff47cc60342e5c10a93d43a19d940be7` (archived serialization) |
| CartPole plotting JSON | 51/51 numeric leaves exact | `fe1e96701a0ee6cc5391716708932f718ff4af9e7f150bf1dc95813e7034a047` (archived serialization) |
| Pendulum PDF from remote export | byte-identical | `07943e60eab5a9d045b81c27a185bb7c48ce40191e3d189ee0ae14397a77bf90` |
| CartPole PDF from remote export | byte-identical | `e54412068f62b8a32f18e2c6e32cc78e140cb81b569a8debc6ef4dd895e30a82` |

The generated aggregate JSON uses normalized indentation, so its file hash is
not expected to equal the hand-formatted archived file. The relevant claim is
stronger than a rounded comparison but narrower than byte identity: every
numeric JSON leaf has the same parsed value. PDF identity is genuinely byte for
byte (`cmp` exit status 0 for both files).

## Final validation

- 8/8 tests passed locally and remotely.
- Python compilation passed for `run`, `scripts`, `src`, tests, and plotting.
- Black reported all 22 checked Python files correctly formatted.
- The launcher dry-run emitted all 28 expected commands.
- Docker Compose validation passed.
- The remote tracking stack and the pre-existing CALF-Enhance stack were healthy
  after all experiments.
- The replay export contains 26 aggregate rows and 780 raw trial rows; no run is
  incomplete.

The exact checkpoint replay establishes that the published CALF-Wrapper numbers
and figures are reproducible from the preserved artifacts. The fresh-training
result shows that a single PPO seed is insufficient to claim training-level
numerical reproducibility, especially for Pendulum; this should be addressed by
the planned multi-seed and hyperparameter sensitivity analysis.
