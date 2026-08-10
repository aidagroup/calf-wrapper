# TD3-Lagrangian baseline specification

## Purpose and scope

This document specifies a standalone CleanRL-style TD3-Lagrangian baseline for
the two off-policy environments in the CALF-Wrapper experiments:

- `UnderwaterDrone-v0`;
- `RobotNavigationConstSpeedCatch-v0`.

The implementation must be contained in `run/train_td3_lagrangian.py`, copied
from the structure of the pinned CleanRL TD3 trainer used by the repository.
Training, replay, evaluation, checkpointing, metric computation, and result
serialization remain in that file. Algorithm code must not be generalized with
the PPO implementation.

TD3-Lagrangian learns a new constrained policy. It neither uses nor modifies the
fallback controller and is not a deployment-time wrapper.

## Finite-horizon constrained problem

Let

\[
\tau=\min\{\text{first environment termination time},T\}.
\]

For AUV and Robot, success is the goal-hitting event by `tau`, equivalently
final goal membership because the AUV freezes at the goal and Robot terminates
there. Define

\[
C(\tau)=\mathbf 1\{\text{the episode ends without reaching }\mathbb G\}.
\]

The optimization problem is

\[
\begin{aligned}
\underset{\pi}{\operatorname{maximize}}\quad
&J_R(\pi)=
\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}\gamma_R^t
r(S_t,A_t)\right],\\
\text{subject to}\quad
&J_C(\pi)=\mathbb E_\pi[C(\tau)]\leq\varepsilon.
\end{aligned}
\]

For the binary terminal cost,

\[
J_C(\pi)=1-\mathbb P_\pi(\text{goal reached by }T).
\]

The default `epsilon = 0.05` therefore targets a goal-reaching rate of 95%.
This is an average finite-horizon constraint under the reset distribution, not
the asymptotic or state-uniform goal-reaching property analyzed for
CALF-Wrapper.

## Environment-specific events

### Contaminated-Zone AUV Navigation

- Environment: `UnderwaterDrone-v0`.
- Horizon: `T = 1500`.
- Physical observation:
  `(x, y, cos(vartheta), sin(vartheta), v_x, v_y, omega)`.
- Goal membership:

  \[
  y\geq 4,\qquad |x|\leq 0.4.
  \]

- The AUV freezes after entering the surface opening. Consequently, final goal
  membership and the recorded `is_in_hole` event agree for normal episodes.
- Time-limit truncation outside the opening produces cost `1`.

The contaminated zone changes reward only. Entering it is not a terminal
constraint violation and must not generate Lagrangian cost.

### Treasure-Collecting Robot

- Environment: `RobotNavigationConstSpeedCatch-v0`.
- Horizon: `T = 1000`.
- The physical observation contains robot position, heading, target position,
  and obstacle data.
- Goal membership:

  \[
  \lVert p_{\mathrm{robot}}-p_{\mathrm{goal}}\rVert_2\leq0.05.
  \]

- Reaching the goal terminates successfully and produces cost `0`.
- Time-limit truncation without reaching the goal produces cost `1`.
- Collision and target-collection terms remain reward components unless the
  environment itself terminates unsuccessfully.

## Time-aware state

Augmenting the state converts the time-inhomogeneous finite-horizon problem into
a stationary Markov representation. Both actor and critics receive

\[
\widetilde S_t=(S_t,t/T).
\]

Without the normalized elapsed-time coordinate, one stationary value function
cannot represent different failure probabilities at different remaining
horizons.

Goal membership always ignores the appended coordinate.

## Replay transition semantics

The replay buffer stores

\[
(\widetilde S_t,A_t,r_t,c_t,\widetilde S_{t+1},d_t),
\]

where

\[
d_t=\texttt{terminated}_t\lor\texttt{truncated}_t
\]

and

\[
c_t=\begin{cases}
1,&d_t=1\text{ and the final state is outside }\mathbb G,\\
0,&\text{otherwise.}
\end{cases}
\]

This differs deliberately from the existing unconstrained TD3 replay behavior,
which can bootstrap reward through time-limit truncations. The constrained
baseline solves a finite-horizon problem, so both reward and cost Bellman
targets stop at every episode end. In particular, cost must never bootstrap
through the autoreset state of the next episode.

The terminal state used for goal membership and stored as the final next state
must come from `final_observation` when vector autoreset is active.

## Lagrangian

The constrained objective uses

\[
\mathcal L(\pi,\lambda)
=J_R(\pi)-\lambda(J_C(\pi)-\varepsilon),\qquad\lambda\geq0.
\]

The multiplier update is

\[
\lambda\leftarrow
\Pi_{[0,\lambda_{\max}]}
\left(\lambda+\alpha_\lambda
(\widehat J_C-\varepsilon)\right).
\]

The constrained policy is the deterministic actor, not the exploratory
behavior policy. Each reset observation is retained once. After
`lambda_update_episodes` new initial observations have accumulated and critic
learning has begun, the current actor and cost critic estimate

\[
\widehat J_C=\frac{1}{m}\sum_{i=1}^{m}
\operatorname{clip}_{[0,1]}
Q_C(\widetilde S_0^{(i)},\pi(\widetilde S_0^{(i)})).
\]

The corresponding initial observations are then removed from the FIFO. Actual
exploratory behavior-policy failures are logged separately and never drive the
multiplier. The raw and clipped critic estimates are both recorded because
this is an approximate off-policy dual step. `lambda_max` is a numerical
diagnostic guard. Hitting it must be reported.

## Networks

The executable copies the existing TD3 architecture:

- deterministic actor with two 256-unit ReLU hidden layers and `tanh` action
  scaling;
- twin reward action-value networks `Q_R1` and `Q_R2`;
- corresponding target networks;
- delayed actor update;
- target-policy smoothing;
- Polyak target updates.

It adds one cost action-value network `Q_C` and its target network. A single
cost critic is the primary specification. Twin cost critics with a maximum may
be studied later as a conservative ablation, but must not silently change the
primary algorithm.

## Reward Bellman targets

For smoothed target action

\[
A'=\operatorname{clip}
(\pi_{\mathrm{target}}(\widetilde S')+\xi),
\]

the TD3 reward target is

\[
y_R=r_{\mathrm{scaled}}
+\gamma_R(1-d)
\min_{i\in\{1,2\}}Q_{Ri,\mathrm{target}}(\widetilde S',A').
\]

Reward critics minimize

\[
L_{Ri}=\mathbb E[(Q_{Ri}(\widetilde S,A)-y_R)^2].
\]

The raw environment reward is retained for all reported returns. A fixed,
positive `reward_scale` may be applied only to critic targets for numerical
conditioning. Positive constant reward scaling does not change feasibility and
preserves the exact constrained argmax. It rescales the corresponding dual
variable: if reward is multiplied by `kappa > 0`, then
`lambda_scaled = kappa * lambda_raw`. Multiplier initialization, learning rate,
upper bound, and reported values are interpreted in scaled-reward units. The
scale must be fixed before training and recorded in every artifact.

The initial recommended scale is `0.01` for both TD3 tasks because their
finite-horizon returns are orders of magnitude larger than the binary cost.
Smoke experiments must verify that reward and cost actor-gradient norms are
both finite and non-negligible. The scale may be changed before full runs only
through a documented, training-only calibration rule, never by selecting on
evaluation performance.

## Cost Bellman target

Cost is undiscounted, `gamma_C = 1`:

\[
y_C=c+(1-d)Q_{C,\mathrm{target}}(\widetilde S',A').
\]

The cost critic minimizes

\[
L_C=\mathbb E[(Q_C(\widetilde S,A)-y_C)^2].
\]

Because there is at most one unit terminal cost, the true action value is a
failure probability in `[0, 1]`. The implementation should log the fraction of
predictions outside this interval, but should not clamp training targets in a
way that hides divergence.

## Actor objective

On delayed policy updates, minimize

\[
L_\pi=
-\mathbb E[Q_{R1}(\widetilde S,\pi(\widetilde S))]
+\lambda\mathbb E[Q_C(\widetilde S,\pi(\widetilde S))].
\]

The multiplier is treated as a constant during the actor gradient. Reward and
cost critics are not updated through the actor loss.

For diagnostics, separately log:

- mean actor reward value;
- mean actor cost value;
- reward contribution to actor loss;
- multiplier-weighted cost contribution;
- gradient norms for actor, reward critics, and cost critic.

## Sparse-cost behavior

At initialization, all episodes may fail, making terminal cost nearly constant.
The cost critic then has limited information about which actions improve
feasibility. The existing shaped rewards are expected to create the first
successful episodes. Once both successes and failures appear, the cost critic
can learn action-dependent failure probability.

The terminal cost must not be replaced with distance-to-goal or time-outside-goal
cost merely to improve learning, because that would define a different CMDP.
If no success occurs during a smoke run, the run is classified as
signal-starved. Permissible responses before full training are:

- longer training;
- more exploratory initial data;
- a declared warm start from the corresponding unconstrained actor as a
  separate fine-tuning baseline.

Warm-start and from-scratch results must never be mixed under one label.

## Training loop

The executable performs:

1. Validate configuration and seed all random generators.
2. Construct a time-aware vector environment.
3. Initialize actor, twin reward critics, cost critic, targets, optimizers, and
   a replay buffer containing reward, cost, and true episode-end flags.
4. Collect random actions until `learning_starts`, then actor actions with
   exploration noise.
5. Recover final observations before autoreset and assign binary terminal cost.
6. Add complete transitions to replay.
7. Track complete episode returns, costs, goal outcomes, and lengths.
8. Sample minibatches and update twin reward critics and the cost critic.
9. On delayed steps, update the actor and all target networks.
10. Update the multiplier from current-actor cost-critic estimates at FIFO
    initial observations; log behavior-policy episode costs separately.
11. Periodically log numerical and constraint diagnostics.
12. Save an atomic inference/provenance checkpoint.
13. Run deterministic evaluation in the same executable.
14. Write machine-readable JSON results.

## Configuration and budgets

The CLI exposes at least:

- environment preset;
- horizon and total timesteps;
- replay capacity, batch size, and learning starts;
- actor/critic learning rate;
- reward discount, reward scale, Polyak coefficient;
- exploration and target-policy noise;
- policy-update frequency;
- `cost_limit`, multiplier initialization, update rate, fixed episode-batch
  size, and upper bound;
- training seed, 200-trial assessment seed, and 30-trial paired-table seed;
- evaluation episode count;
- checkpoint interval, device, and output directory.

Reference full budgets match the existing TD3 experiments:

- AUV: 3,000,000 environment steps;
- Robot: 3,000,000 environment steps.

The frozen full matrix uses AUV training seeds 0--9 and Robot training seeds
1--10, matching the existing TD3 matrix.

## Evaluation and statistics

Constraint assessment uses the deterministic actor on held-out seeds
10000--10199. A separate evaluation on seeds 42--71 is retained only for the
legacy paired table and is not included in the binomial assessment. Evaluation
does not change networks, replay, target networks, exploration state,
optimizers, or multiplier. Raw per-trial seeds, returns, binary successes,
costs, and episode lengths are retained.

The result JSON contains:

- mean episode return, standard deviation, and approximate 95% normal
  confidence-interval half-width;
- binary mean episode cost;
- goal-reaching probability in `[0,1]`, an explicitly named percentage, and a
  95% Wilson interval;
- a one-sided exact 95% upper confidence bound on failure probability and the
  corresponding strict assessment flag;
- evaluation episode count;
- final multiplier and cost limit;
- training steps and elapsed time;
- environment, horizon, seed, reward scale, and checkpoint path;
- empirical constraint status;
- whether the multiplier upper bound was reached;
- replay success/failure composition and latest critic diagnostics.

The identity

\[
\texttt{mean_episode_cost}=1-\texttt{goal_reaching_probability}
\]

must be asserted before writing successful results. Use at least 200 evaluation
episodes for constraint assessment, but do not treat sample count alone as
validation. The strict gate requires the one-sided 95% Clopper--Pearson upper
bound on failure probability to be no greater than `epsilon`. A separate paired
30-episode evaluation may be produced for compatibility with the current
manuscript table.

## Checkpoint contract

The checkpoint is an atomic inference/provenance artifact and includes:

- format version and full configuration;
- actor and target actor;
- both reward critics and targets;
- cost critic and target;
- all optimizer states;
- multiplier and completed-step count;
- replay size, position, and capacity metadata;
- environment, horizon, reward scale, and seeds;
- Python, NumPy, Torch CPU, and Torch CUDA RNG states.

Loading incompatible observation/action dimensions, environment, horizon, or
format must fail clearly. Deterministic actor output must survive a checkpoint
round trip exactly within floating-point tolerance. Exact training resumption
is not supported because replay contents, environment state, and the pending
initial-observation FIFO are not stored.

## Correctness gates

Before full AUV or Robot training:

1. The time coordinate is zero at reset and one at timeout.
2. Goal predicates ignore the time coordinate.
3. Successful terminal transitions have cost zero.
4. Failed truncations have cost one.
5. `done = terminated OR truncated` in both reward and cost targets.
6. `final_observation` is stored instead of the autoreset observation.
7. Cost targets use `gamma_C = 1`.
8. The multiplier changes in the correct direction using clipped initial-state
   cost-critic estimates for the current deterministic actor.
9. At multiplier zero, the actor and reward-critic updates reduce to the copied
   CleanRL TD3 logic, apart from the declared finite-horizon timeout handling
   and fixed reward scaling.
10. Mean episode cost equals one minus goal-reaching probability.
11. Evaluation does not mutate train state.
12. Checkpoint loading reproduces deterministic actor actions.
13. Reward and cost losses, values, actions, and gradient norms remain finite.
14. Short smoke runs complete on both AUV and Robot and contain at least one
    complete episode.
15. A toy replay batch verifies that no target bootstraps through truncation.
16. CLI dry-runs resolve both environment presets and all output paths.

## Known limitations to report

- The finite-horizon chance constraint is not the CALF-Wrapper goal-reaching
  theorem.
- Constraint satisfaction is learned approximately and may fail under neural
  approximation, off-policy distribution shift, or insufficient exploration.
- The terminal cost is sparse, especially for 1000- and 1500-step tasks.
- Naive Lagrangian updates can oscillate; a PID multiplier would constitute a
  different baseline and must be labeled separately.
- Reward scaling affects optimization conditioning even though fixed positive
  scaling does not change the exact constrained optimum.
- TD3-Lagrangian requires millions of additional environment interactions,
  whereas CALF-Wrapper does not retrain the base policy.

## Frozen launch protocol

`scripts/run_lagrangian_matrix.py` renders the complete matrix by default and
executes it only with `--execute`. Full execution refuses dirty or unpushed
source revisions and refuses existing output directories. `--smoke` selects
the short four-environment readiness configuration; `--allow-dirty` is
restricted to those smoke runs.

The displayed reward scale and multiplier parameters are frozen before the
held-out schedule is evaluated. Any calibration must use disjoint training
seeds and training diagnostics only. Seeds 10000--10199 are reserved for final
evaluation. The Clopper--Pearson assessment is made separately for each
trained actor; it is not computed after pooling trials from multiple actors.
