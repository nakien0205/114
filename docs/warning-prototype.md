# Warning prototype

The maintained code is in [src/fire_warning](../src/fire_warning/). It contains tracking, persistence, a warning state machine, video I/O, annotation review, and benchmark/export tools. It is a separate prototype; its historical baseline checkpoint is not the selected thesis method.

From a source checkout:

```powershell
python scripts/warning.py --help
python scripts/warning.py --weights "path/to/baseline-best.pt" --source "path/to/clip.mp4" --headless --silent
```

After installing the package, module entry points are also available:

```powershell
python -m fire_warning.evaluate_benchmark --help
python -m fire_warning.review_server --help
```

For the optional UI, install [requirements/warning-ui.txt](../requirements/warning-ui.txt). Export dependencies are separate in [requirements/export.txt](../requirements/export.txt). The preserved original Docker launchers have not been migrated or validated for the new package layout; they are local historical material.

The prototype retains the original checkpoint identity check and separate raw, persistence, and full-state-machine variants. Event-timestamp outcomes can be detection proxies; use reviewed boxes for spatial claims. Legacy predicted-size groups are tiny `d <= 16`, small `16 < d <= 32` at square-640 scale. Missed events do not have an observed predicted size.

Recorded-video replay, physical-camera measurement, model latency, and alarm delay are different evaluations. Software tests do not establish detector accuracy or sustained real-time performance. The cleanup does not open a camera or run model evaluation.

The original `system_4stage/` tree and its local environment remain on the owner's computer, outside the shared package. The maintained source was copied before its package references and project-root paths were updated. Regression tests live in [tests/warning](../tests/warning/).
