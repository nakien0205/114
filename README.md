# Fire and Smoke Early Warning Research

Research code for tiny fire/smoke detection and video warning. The main thesis method remains undecided.

Start with [claims and evidence](docs/claims.md). Each supported statement points to its command, configuration, and saved result. See [data](docs/data.md) for the dataset identity and interpretation limits.

## Layout

```text
configs/       Runtime templates, model definitions, experiment settings
docs/          Methods, claim-to-result guide, reproduction instructions
experiments/   Frozen experiment source and notebooks
requirements/  Platform lock and optional dependencies
results/       Selected machine-readable results and provenance
scripts/       Training, evaluation, verification, and reproduction commands
src/           Maintained dataset, detector, and warning packages
tests/         CPU smoke tests and warning regressions
```

## Setup

Use Python 3.11. [Reproduction instructions](docs/reproducibility.md) distinguish the maintenance environment from historical training/replay environments.

```powershell
python -m venv .venv
# Use the environment's Python executable below.
python -m pip install -r requirements/locks/windows-py311-cu128.txt
python -m pip install --no-deps -e .
```

The supplied lock targets Windows and CUDA 12.8. Other platforms require a separately resolved environment; do not reuse it as the Kaggle training lock.

## Commands

```powershell
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/infer.py --help
python scripts/run_pipeline.py --help
python scripts/verify_dataset.py --help
python scripts/warning.py --help
```

Runtime defaults live in [configs/runtime.yaml](configs/runtime.yaml). Edit [configs/data.yaml](configs/data.yaml) for your dataset or pass an explicit experiment YAML. General defaults are not the recipe for every frozen experiment.

## Verify a result

```powershell
python scripts/reproduce_table.py --check
python scripts/check_evidence.py
python -m pytest -q
```

The first command rebuilds the five-run combined60 table from saved scored summaries. It does not repeat training, inference, or raw-prediction scoring. Full-run inputs and remaining reproduction gaps are documented in [claims and evidence](docs/claims.md).

Use `python scripts/check_evidence.py --require-tracked` before preparing a shared commit. It checks inclusion and file identity, not scientific validity or full dependency closure.

## Research boundaries

Still-image AP does not establish warning delay or video false alarms/hour. Source splits are not verified independent scenes. Previously examined video studies are exploratory. No reliable ultra-early smoke timing or physical edge-device performance is established.

See the [experiment index](docs/research-index.md), [repository map](docs/repository-map.md), and [LICENSE](LICENSE). External dependencies and datasets retain their own licenses. [CITATION.cff](CITATION.cff) intentionally leaves author/institution fields blank for the owner to complete; it is not yet a validated release citation.
