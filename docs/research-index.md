# Experiment index

The main method is undecided. Keep datasets, model scales, training budgets, runtime, and evaluation geometry separate.

| Family | Code / settings | Saved results | Status |
|---|---|---|---|
| Combined Home Fire + D-Fire, 60 epochs | [Frozen notebooks/source](../experiments/combined60/), [archived configs](../configs/experiments/combined60/) | [Five-run table](../results/combined60/comparison.csv), [per-run records](../results/combined60/runs/) | Descriptive development comparison. Plain seed42 failed before training. The SRC notebook is source only; this table has no combined-data SRC outcome. |
| Original versus P2, Home Fire | Original source remains in the local historical archive. | [Table](../results/historical/original-versus-p2-20261030/comparison.csv) | Historical result; a full portable reproduction bundle has not been assembled. |
| Nano SRC, Home Fire | Separate from combined-data SRC preparation. | [Table](../results/historical/nano-src-comparison-20261028/comparison.csv) | Single-seed development comparison. |
| Medium SRC, Home Fire | Separate scale and initialization history. | [Table](../results/historical/medium-src-comparison-20261029/comparison.csv) | Do not pool with nano outcomes. |
| Medium adaptive/fixed gates | Historical controls. | [Adaptive](../results/historical/medium-adaptive-comparison-20261027/comparison.csv), [final comparison](../results/historical/medium-neck-final-comparison-20261027/medium-comparison.csv) | Tables retained; full portable reproduction remains incomplete. |
| DySample variants, Home Fire | Separate architectures. | [DySample](../results/historical/nano-dysample-comparison-20261029/comparison.csv), [adaptive + DySample](../results/historical/nano-adaptive-dysample-comparison-20261029/comparison.csv) | Historical development comparisons. |
| FLAME-inspired video postprocessing | [Frozen case-study source](../experiments/flame_case_study/) | [Evaluation](../results/flame_case_study/evaluation.json) | One previously examined clip; no generalization or no-fire alarm-rate claim. |
| Warning prototype | [Maintained package](../src/fire_warning/), [usage](warning-prototype.md) | Software regression tests, not a new accuracy result. | Main thesis method not selected. |

Older proposals, processing write-ups, continuation plans, and incomplete studies remain local history. Omitting them from the shared file list does not turn their outcomes into successful experiments. [Source provenance](../experiments/source-provenance.json) identifies copied files by original path and hash.

For the statements currently supported by these artifacts, start with [claims and evidence](claims.md).
