# Deterministic publication plots

The plotting environment is intentionally isolated from the training
environment and exactly pins the versions used for the archived paper plots.
The PDF creation timestamps are fixed to the original source timestamps; this
makes both generated PDFs byte-for-byte reproducible.

```bash
cd publication
uv sync --frozen
uv run plots.py
sha256sum images/cartpole.pdf images/pendulum.pdf
```

Expected hashes are documented in `../reference-results/README.md`.

To render an exported checkpoint replay rather than the bundled reference
files, pass its aggregate JSON directory explicitly:

```bash
uv run plots.py \
  --data-dir /tmp/calf-wrapper-remote-reference-replay \
  --output-dir /tmp/calf-wrapper-remote-reference-replay/plots
```

The archived Residual RL values are a plot-only external baseline. The export
provenance records that they are carried through from the archived JSON because
this repository contains neither the Residual RL implementation nor its
checkpoints. They are excluded from CALF-Wrapper reproducibility verdicts.
