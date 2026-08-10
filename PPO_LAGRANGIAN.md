# PPO-Lagrangian baseline specification

## Purpose and scope

This document specifies a finite-horizon constrained-reinforcement-learning
baseline for the PPO environments used in the CALF-Wrapper experiments:

- `Pendulum-v1`;
- `CartpoleSwingupEnvLong-v0`.

The implementation must be a standalone CleanRL-style executable named
`run/train_ppo_lagrangian.py`. Training, checkpointing, deterministic
evaluation, metric computation, and result serialization must remain in that
single file. Environment registration and the canonical goal-set predicate may
be imported from `src`, but the PPO-Lagrangian algorithm must not be hidden
behind a framework implementation or a shared algorithm abstraction.

PPO-Lagrangian is a training-time constrained-RL baseline. It does not use the
fallback policy and is not a deployment-time wrapper. Its purpose is to test
whether a policy trained specifically against an empirical goal-failure
constraint can match the reward and goal-reaching rate of CALF-Wrapper.

## What the baseline does and does not establish

The baseline targets a finite-horizon chance constraint under the environment's
initial-state distribution. It does not establish the paper's asymptotic,
state-uniform goal-reaching property.

Let

\[
\tau=\min\{\text{first environment termination time},T\}.
\]

Terminal states are treated as absorbing up to `T`. For Pendulum and CartPole,
success means final goal membership at `tau`. Define the terminal failure random
variable

\[
C(\tau)=\mathbf 1\{S_\tau\notin\mathbb G\}.
\]

The constrained problem is

\[
\begin{aligned}
\underset{\pi}{\operatorname{maximize}}\quad
&J_R(\pi)
=\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}\gamma_R^t
r(S_t,A_t)\right],\\
\text{subject to}\quad
&J_C(\pi)=\mathbb E_\pi[C(\tau)]\leq\varepsilon.
\end{aligned}
\]

Because `C` is binary,

\[
J_C(\pi)=\mathbb P_\pi(S_\tau\notin\mathbb G),
\]

so `epsilon = 0.05` corresponds to a target terminal goal-reaching
probability of at least `0.95`.

The probability identity is exact for the specified finite-horizon population
event; its value is estimated empirically. It is not a formal guarantee for
unseen initial states, and neural PPO-Lagrangian does not
provide a general convergence or zero-duality-gap result for this nonconvex
function-approximation problem.

## Environment-specific finite-horizon events

### Pendulum

- Environment: `Pendulum-v1`.
- Horizon: `T = 200`.
- Physical observation: `(cos(vartheta), sin(vartheta), vartheta_dot)`.
- Goal membership uses the canonical predicate in `src.goal_reaching`:

  \[
  |\cos\vartheta-1|<0.05,\qquad
  |\sin\vartheta|<0.05,\qquad
  |\dot\vartheta|<0.3.
  \]

- The episode succeeds iff the final physical observation belongs to
  $\mathbb G$.
- A time-limit truncation outside $\mathbb G$ produces terminal cost `1`.

### CartPole swing-up

- Training and evaluation environment:
  `CartpoleSwingupEnvLong-v0`.
- Horizon: `T = 1000`.
- Physical observation:
  `(x, x_dot, cos(vartheta), sin(vartheta), vartheta_dot)`.
- Goal membership uses the canonical predicate:

  \[
  |x|<0.3,\quad |\dot x|<0.3,\quad
  |\cos\vartheta-1|<0.05,\quad
  |\sin\vartheta|<0.05,\quad
  |\dot\vartheta|<0.05.
  \]

- The episode succeeds iff its final physical observation belongs to
  $\mathbb G$.
- Any early physical termination outside $\mathbb G$ and any time-limit
  truncation outside $\mathbb G$ produce terminal cost `1`.

The original unconstrained CartPole PPO training used 200-step episodes, while
the reported evaluation uses a 1000-step environment. The constrained baseline
uses the 1000-step horizon during both training and evaluation so that its
constraint is the same event as its reported goal-reaching rate. This protocol
difference must be disclosed in any manuscript comparison.

## Time-aware Markov state

Terminal failure depends on the remaining horizon. Augmenting the physical
state converts the time-inhomogeneous finite-horizon problem into a stationary
Markov representation. The policy and both value functions receive

\[
\widetilde S_t=(S_t,t/T).
\]

The extra component is in `[0, 1]`. Goal membership is always evaluated on the
physical part `S_t`, never on the appended time coordinate.

## Transition cost and episode boundaries

The per-transition cost stored in the rollout buffer is

\[
c_t=\begin{cases}
1,&\text{if the transition ends the episode outside }\mathbb G,\\
0,&\text{otherwise.}
\end{cases}
\]

The episode-end flag is

\[
d_t=\texttt{terminated}_t\lor\texttt{truncated}_t.
\]

When a vector environment autoresets, terminal success must be computed from
`info["final_observation"]` and `info["final_info"]`, not from the reset
observation returned as `next_observation`.

Cost is undiscounted:

\[
\gamma_C=1.
\]

Consequently, for every completed episode, the sum of transition costs is
exactly its binary failure indicator. A required runtime invariant is

\[
\texttt{mean_episode_cost}=1-\texttt{goal_reaching_probability}.
\]

## Lagrangian formulation

For multiplier `lambda >= 0`, the Lagrangian is

\[
\mathcal L(\pi,\lambda)
=J_R(\pi)-\lambda\bigl(J_C(\pi)-\varepsilon\bigr).
\]

The policy maximizes the Lagrangian with respect to its parameters. The
multiplier minimizes it through projected ascent on constraint violation:

\[
\lambda\leftarrow
\Pi_{[0,\lambda_{\max}]}
\left(\lambda+\alpha_\lambda
(\widehat J_C-\varepsilon)\right).
\]

`J_C` is estimated from a fixed-size FIFO batch of newly completed episodes,
with every reset episode receiving equal weight. The multiplier must never be
updated from the mean cost per transition. An update occurs only after
`lambda_update_episodes` new outcomes have accumulated; those outcomes are then
removed from the pending batch. This avoids overweighting short early-failure
episodes or repeatedly reusing the same outcomes.

`lambda_max` is a numerical guard, not a theoretical ingredient. Reaching it
must be logged as a possible infeasibility or optimization-failure diagnostic.

## Network architecture

The executable contains one `Agent` module with:

- a tanh-squashed Gaussian actor with two 64-unit `tanh` hidden layers;
- a reward value network `V_R` with two 64-unit `tanh` hidden layers;
- a cost value network `V_C` with two 64-unit `tanh` hidden layers;
- a state-independent trainable action log-standard-deviation.

The actor samples a latent Gaussian action `U`, then executes

\[
A=\operatorname{bias}+\operatorname{scale}\tanh U.
\]

The log probability includes the change-of-variables correction

\[
\log\pi(A\mid S)=\log p_U(U\mid S)
-\sum_j\log\!\left(
\operatorname{scale}_j(1-\tanh^2 U_j)+\epsilon_{\mathrm{num}}
\right).
\]

The rollout buffer stores the latent action, making the PPO likelihood ratio
the ratio for the bounded action distribution that generated the environment
action. The implementation also logs action-saturation rate.

## Reward and cost generalized advantages

Reward GAE uses the environment reward discount `gamma_R` and `lambda_R`:

\[
\delta_t^R=r_t+\gamma_R(1-d_t)V_R(\widetilde S_{t+1})
-V_R(\widetilde S_t),
\]

\[
A_t^R=\delta_t^R+\gamma_R\lambda_R(1-d_t)A_{t+1}^R.
\]

Cost GAE uses `gamma_C = 1` and `lambda_C`:

\[
\delta_t^C=c_t+(1-d_t)V_C(\widetilde S_{t+1})
-V_C(\widetilde S_t),
\]

\[
A_t^C=\delta_t^C+\lambda_C(1-d_t)A_{t+1}^C.
\]

Value targets are

\[
\widehat V_t^R=A_t^R+V_R(\widetilde S_t),\qquad
\widehat V_t^C=A_t^C+V_C(\widetilde S_t).
\]

No bootstrap is allowed through either physical termination or time-limit
truncation for this finite-horizon problem.

## PPO-Lagrangian policy objective

The reward critic and reward GAE use a fixed scaled reward
$r_{\mathrm{scaled}}=\kappa r$ with default $\kappa=0.01$. Raw environment
returns are retained for reporting. Positive scaling preserves the constrained
argmax and rescales the dual variable as
$\lambda_{\mathrm{scaled}}=\kappa\lambda_{\mathrm{raw}}$; multiplier
hyperparameters and reported multiplier values therefore use scaled-reward
units.

For a fixed multiplier during one PPO update, first combine the raw
scaled-reward and cost advantages as

\[
A_t^L=\frac{A_t^R-\lambda A_t^C}{1+\lambda}.
\]

The combined advantage may then be centered and divided by its own positive
standard deviation before the clipped update. This preserves the direction of
the Lagrangian policy gradient up to a positive batch-dependent scale; reward
and cost advantages are never standardized separately.

With probability ratio

\[
q_t(\theta)=
\frac{\pi_\theta(A_t\mid\widetilde S_t)}
{\pi_{\theta_{\mathrm{old}}}(A_t\mid\widetilde S_t)},
\]

the clipped policy objective is

\[
L_{\mathrm{clip}}
=\mathbb E\!\left[
\min\left(
q_t A_t^L,
\operatorname{clip}(q_t,1-\delta,1+\delta)A_t^L
\right)
\right].
\]

The minimized total loss is

\[
L=-L_{\mathrm{clip}}
-c_H\mathcal H(\pi)
+c_R L_{V_R}+c_C L_{V_C},
\]

where both value losses are mean squared errors against their corresponding
GAE targets.

## Training loop

The single executable performs the following sequence:

1. Validate configuration and seed Python, NumPy, Torch, action spaces, and
   vector environments.
2. Construct time-aware vector environments.
3. Allocate rollout tensors for observations, actions, log probabilities,
   rewards, terminal costs, episode-end flags, and both value estimates.
4. Collect `num_steps * num_envs` transitions.
5. Extract final observations before vector autoreset and compute binary
   terminal costs.
6. Track complete episodic costs independently of transition buffers.
7. Compute reward and cost GAE with separate discounts.
8. Freeze the current multiplier and perform PPO minibatch epochs.
9. Add completed outcomes to the FIFO multiplier batch and update only when a
   full batch of newly completed episodes is available.
10. Log losses, KL divergence, clip fraction, multiplier, completed-episode
    failure rate, throughput, and number of complete episodes.
11. Save a self-describing checkpoint.
12. Run stochastic constraint evaluation and a separate deterministic
    mean-action evaluation with the common held-out seeds.
13. Serialize all final metrics to JSON.

## Required configuration

At minimum, the CLI exposes:

- environment preset;
- horizon;
- total timesteps;
- rollout steps and number of vector environments;
- minibatches and update epochs;
- reward discount and both GAE parameters;
- fixed reward scale;
- PPO clip coefficient and optimizer parameters;
- `cost_limit`, `lambda_init`, `lambda_lr`, `lambda_update_episodes`, and
  `lambda_max`;
- training seed, 200-trial assessment seed, and 30-trial paired-table seed;
- evaluation episode count;
- device and output directory.

Default experiment budgets match the existing PPO runs:

- Pendulum: 102,000 environment steps, seeds 9, 10, and 11;
- CartPole: 300,000 environment steps, seeds 42, 43, and 44.

The paired reproduction comparison uses the existing trained-policy seeds:
Pendulum seed 9 and CartPole seed 42. Seeds 10--11 and 43--44 provide the
separate training-seed robustness assessment and are not pooled with the
paired trials.

## Evaluation and statistics

The constrained PPO policy is stochastic, so the primary constraint assessment
samples that same tanh-squashed Gaussian policy. A deterministic mean-action
evaluation is also recorded as a separate descriptive block; constraint
satisfaction of the stochastic policy is not transferred to the mean-action
policy. Evaluation restores model mode and Torch random-generator state and
does not update the actor, critics, optimizer, or multiplier.

The constraint-assessment schedule is the genuinely held-out seed range
10000--10199. A separate deterministic mean-action evaluation on seeds 42--71
is retained only for compatibility with the legacy paired table; it is not
used for the binomial assessment. Every seed, return, binary success, terminal
cost, episode length, and stochastic-policy noise seed is stored in the result
JSON.

The result JSON contains at least:

- mean episode return;
- return standard deviation;
- approximate 95% normal confidence-interval half-width for the mean return;
- mean terminal cost;
- goal-reaching probability in `[0,1]` and an explicitly named percentage;
- 95% Wilson interval for goal-reaching rate;
- number of evaluation episodes;
- final multiplier and cost limit;
- training steps, seed, horizon, environment, and checkpoint path;
- whether the empirical point estimate satisfies the constraint;
- a one-sided exact 95% upper confidence bound on failure probability;
- whether that upper confidence bound is at most `epsilon`;
- whether `lambda_max` was reached.

Thirty paired trials may be reported to match the existing manuscript table,
but they are insufficient to assess a 5% failure constraint. Run-readiness
assessment must use at least 200 episodes. Even 200 episodes do not
automatically validate the constraint: the strict statistical gate is a
one-sided 95% Clopper--Pearson upper bound on failure probability no greater
than `epsilon`. Wilson intervals remain descriptive summaries.

## Checkpoint contract

The checkpoint is an atomic inference/provenance artifact and contains:

- an explicit format version;
- complete CLI configuration;
- model state;
- optimizer state for diagnostics;
- multiplier value;
- completed environment steps;
- observation and action dimensions;
- environment ID, horizon, and seeds.

Loading a checkpoint into an incompatible environment or horizon must fail
with a descriptive error. A round-trip test must show identical deterministic
actions before and after saving. Exact mid-rollout training resumption is not
supported because environment state, rollout buffers, pending multiplier
outcomes, and RNG state are not stored.

## Correctness gates

No full experiment may start before all of the following pass:

1. Terminal cost is zero on success and one on truncation outside $\mathbb G$.
2. `final_observation` is used instead of the autoreset observation.
3. Both GAE recursions stop at `terminated OR truncated`.
4. Cost GAE uses `gamma_C = 1`.
5. The multiplier increases above the limit and decreases below it.
6. With multiplier fixed to zero and the cost-value loss disabled, the actor
   update reduces to ordinary CleanRL PPO.
7. The appended time coordinate is zero after reset and one at the horizon.
8. Mean episodic cost equals one minus goal-reaching probability exactly.
9. Evaluation leaves every trainable parameter unchanged.
10. Checkpoint save/load preserves deterministic actions.
11. Synthetic feasible and violating episode batches move the multiplier in
    opposite directions, with projection at zero.
12. Short smoke training and evaluation complete for Pendulum and CartPole
    without NaNs, invalid actions, missing complete episodes, or malformed
    artifacts.

## Known limitations to report

- The constraint is in expectation over the initial-state distribution.
- The result is finite-horizon and empirical, not the CALF-Wrapper theorem.
- PPO-Lagrangian with neural networks is a nonconvex primal-dual method and may
  oscillate or fail to find a feasible policy.
- Terminal failure is sparse. Reward shaping already present in the
  environments is expected to produce the first successes; the cost definition
  must not be silently replaced by dense distance cost.
- CartPole uses a longer training horizon than the original unconstrained PPO
  run so that training and evaluation constraints coincide.
- Training-time environment interactions and wall-clock cost must be reported
  because CALF-Wrapper requires no additional policy training.

## Frozen launch protocol

`scripts/run_lagrangian_matrix.py` is the source of truth for commands, output
paths, training seeds, the 200 held-out assessment seeds, the 30 legacy paired
seeds, and smoke overrides.
It renders commands by default. `--execute` starts them, refuses to overwrite
an existing output directory, and requires a clean commit contained in a
remote branch. Only smoke runs may bypass that repository check with
`--allow-dirty`.

Every matrix run uses the MLflow server at `http://192.168.1.5:5001` by
default and belongs to `CALF-Wrapper/Lagrangian-Baselines`.
One MLflow run corresponds to one algorithm, environment, and training seed.
It records the complete configuration, source revision, host, training
metrics, final held-out and paired evaluations, raw-trial JSON, JSONL training
records, and checkpoint.

The displayed PPO hyperparameters, including the multiplier update rate, are
frozen before held-out evaluation. Any calibration must use separate training
seeds and may inspect training diagnostics only; held-out seeds 10000--10199 must
not be used to select hyperparameters. Constraint assessment is per trained
policy. Results from independently trained policies are not pooled into one
Clopper--Pearson claim.
