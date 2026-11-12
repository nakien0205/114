# Claims and evidence

The thesis method remains undecided. These are statements supported by saved development results, not a declaration of the final contribution.

| ID | Supported statement | Command | Configuration / inputs | Result |
|---|---|---|---|---|
| C01 | In the common combined60 local replay, adaptive seed2026 has higher tiny-fire and overall AP, but lower historical small-fire AP, than plain seed2026. | `python scripts/reproduce_table.py --check` | [Effective training arguments](../configs/experiments/combined60/), [replay runtime/protocol](../results/combined60/replay-runtime-protocol.json), per-run saved summaries | [comparison.csv](../results/combined60/comparison.csv) |
| C02 | In that replay, dense seed2026 has lower tiny-fire, historical small-fire, and overall AP than plain seed2026. | `python scripts/reproduce_table.py --check` | Same combined60 inputs and runtime; dense changes neck/initialization recipe. | [comparison.csv](../results/combined60/comparison.csv) |
| C03 | The combined60 plain baseline has only one completed seed. Plain seed42 failed before training. | `python scripts/check_evidence.py` checks file identities only. | [Saved archive record](../results/combined60/archive-audit.json) | Five completed run summaries; no plain seed42 score is imputed. |
| C04 | The FLAME-inspired confirmation arms produced no confirmed fire localization in the saved single-clip case study. | No portable full-video reproduction is claimed for this snapshot. | [Case-study specification](../experiments/flame_case_study/video12-study.json); original video and prediction caches remain external. | [evaluation.json](../results/flame_case_study/evaluation.json), [summary.json](../results/flame_case_study/summary.json) |

`reproduce_table.py` rebuilds all 19 columns from the saved scored summaries and compares them with the preserved CSV. It does not rerun neural inference or recompute AP from raw predictions. Repeating the full combined60 validation still requires the exact checkpoints, original validation images, raw/runtime support, and the recorded local environment. The [experiment notebooks](research-index.md) are training source snapshots, not evidence that a new run has been performed.

The common seed supports descriptive point comparisons. It does not establish statistical significance, repeatability, independent-event generalization, or earlier reliable warnings. Historical small-fire AP includes tiny. Still-image false boxes and alarm images are not video incidents per hour.

The original 10/12-by-3-seconds video target is not demonstrated by the retained evidence. No smoke-onset timing or physical edge-device performance claim is established here.
