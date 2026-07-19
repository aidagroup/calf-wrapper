# Published reference results

These values are the canonical aggregate inputs used for the paper figures.
The `residual` entries originate from CALF-Enhance; all other entries are
produced by this repository's 30-trial evaluation matrix. Comparison tooling
checks floating-point values without rounding.

The archived PPO checkpoint tensors were generated on CUDA. Two independent
local CUDA runs reproduce all selected checkpoint tensors exactly; CPU training
uses a different floating-point trajectory and is not an equivalent paper
reproduction protocol.

Canonical paper artifact SHA-256 values:

- `cartpole.json`: `fe1e96701a0ee6cc5391716708932f718ff4af9e7f150bf1dc95813e7034a047`
- `pendulum.json`: `bae3a5bbb7bc0063d757f76d5781859dff47cc60342e5c10a93d43a19d940be7`
- `cartpole.pdf`: `e54412068f62b8a32f18e2c6e32cc78e140cb81b569a8debc6ef4dd895e30a82`
- `pendulum.pdf`: `07943e60eab5a9d045b81c27a185bb7c48ce40191e3d189ee0ae14397a77bf90`
