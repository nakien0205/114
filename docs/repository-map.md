# Repository map

| Directory | Contents |
|---|---|
| `src/fire_audit/` | Dataset checks, sequence grouping, and partitioning |
| `src/fire_detection/` | Detector code, `losses/`, `wavelet/`, and `attention/` |
| `src/fire_warning/` | Maintained tracking, warning state, video review, and demo code |
| `src/modules/` | Small legacy import forwarders for older source-checkout checkpoint paths; no maintained implementations |
| `scripts/` | User commands and evidence/table checks |
| `configs/` | The sole home for editable configs and selected archived settings |
| `experiments/` | Frozen source/notebooks; [original-byte provenance](../experiments/source-provenance.json) |
| `results/` | Selected numeric results and run identities; prose belongs in docs |
| `docs/` | Claim-to-command-to-config-to-result guide and necessary methods |
| `tests/` | Dataset/config/model smoke tests and warning regressions |
| `requirements/` | Platform-scoped environment lock and optional UI/export dependencies |

The root contains the README, license, requirements entry files, package/test metadata, and folders. Standalone audit/proposal/planning Markdown is local-only.

Original `research/`, `reports/`, `system_4stage/`, paper downloads, and scratch outputs remain on the owner's computer. The shared source and result selection is explicit; those historical directories are not the required checkout layout. Frozen copies retain their original bytes, while maintained packages use the new paths.

Old `src.modules` paths are retained for source-checkout compatibility. Use the new `fire_detection.losses`, `fire_detection.wavelet`, and `fire_detection.attention` names in maintained code. A synthetic legacy class reference is tested; arbitrary historical checkpoints have not all been loaded.

The selected notebooks still require their external data/runtime companions. Their presence does not certify training completion, and the source-provenance inventory is not a complete dependency closure. Follow [claims](claims.md) and [reproduction](reproducibility.md) for the actual boundaries.
