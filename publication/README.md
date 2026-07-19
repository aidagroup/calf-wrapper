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
