# Reproduction

## Maintenance environment

The current source layout targets Python 3.11. [requirements.txt](../requirements.txt) pins direct dependencies; the [Windows/Python3.11/CUDA12.8 lock](../requirements/locks/windows-py311-cu128.txt) pins the transitive core/test dependency set. The lock was assembled from available distribution metadata and checked against dependency specifiers. It does not describe historical Kaggle training.

Validation uses a separate virtual environment that can reuse existing installed dependencies, then installs the complete lock and checks all 69 pinned versions. This is not a clean-machine installation test.

Use an isolated environment and its Python executable:

```powershell
python -m venv .venv
python -m pip install -r requirements/locks/windows-py311-cu128.txt
python -m pip install --no-deps -e .
python -m pip check
python -m pytest -q
```

Choose a separate compatible torch/torchvision build and resolve a new lock for a different OS/CUDA target using the [official PyTorch instructions](https://pytorch.org/get-started/locally/). Do not upgrade an environment used to produce archived scores as a cleanup step.

Package discovery uses the standard [setuptools src layout](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#src-layout). UI and export dependencies are optional: [warning UI](../requirements/warning-ui.txt), [exports](../requirements/export.txt).

## Historical environments

| Context | Recorded versions |
|---|---|
| Combined60 Kaggle training | torch 2.11.0+cu128; torchvision 0.26.0+cu128; NumPy 2.1.3; Ultralytics 8.4.24 |
| Common saved local replay | torch 2.10.0+cu128; torchvision 0.25.0+cu128; NumPy 2.4.6; Ultralytics 8.4.24 |
| Maintenance target | torch 2.10.0+cu128; torchvision 0.25.0+cu128; NumPy 2.2.6; Ultralytics 8.4.24 |

The recorded local replay used NumPy 2.4.6 even though OpenCV 4.12's metadata requires NumPy <2.3 on this Python version. Preserve that fact in [the replay record](../results/combined60/replay-runtime-protocol.json); the maintenance fix does not retroactively establish exact score reproduction.

## Rebuild the combined60 table

```powershell
python scripts/reproduce_table.py --check
python scripts/reproduce_table.py --output runs/rebuilt-combined60.csv
```

The command reads the five saved `local-summary.json` records under [results/combined60/runs](../results/combined60/runs/), rebuilds all 19 columns, and checks against [comparison.csv](../results/combined60/comparison.csv). It refuses an existing output file and does not invent the missing plain seed42 run.

This reproduces table assembly from already scored metrics. Full AP rescoring needs original predictions/ground truth; full detector replay also needs the exact checkpoints, images, and runtime. Those large external inputs are not supplied by this cleanup.

## Training snapshots

[Combined60 snapshots](../experiments/combined60/) contain the plain/SRC, older adaptive, and dense training source/notebooks. Their external image and shared companion inputs remain frozen. The returned-run [effective training arguments](../configs/experiments/combined60/) are preserved separately. Archived paths in those records document the old run; they are not commands that can run unchanged on a new machine.

The [FLAME-inspired snapshot](../experiments/flame_case_study/) is a single-clip adaptation around an existing detector. Its score script still depends on the original reference/prediction assets; it is not presented as a portable full-video benchmark.

See [data](data.md) and [claims](claims.md). No new training, camera capture, held-out evaluation, or model-accuracy rerun is part of the repository restructure.
