# Data

The current combined-data experiments use the frozen complete Home Fire + D-Fire input: 18,022 training images, 4,399 validation images, and 5,606 reserved test images. Class IDs are fire=0 and smoke=1.

Dataset identity:

```text
b7ff6c25e1326f9f380b73bef5cbec78fe658618169a8fdef8c6810d6a9d6ad7
```

[Dataset manifest](../configs/datasets/combined60/dataset-manifest.json) and [canonical validation targets](../configs/datasets/combined60/validation-ground-truth.json) identify the data used by the saved combined60 comparison. Source split membership was preserved. The preparation applied 30 Home Fire orientation corrections and removed 14 zero-area D-Fire boxes from train/validation. Original source files remain unchanged.

Related scenes can still occur across source splits; exact-image duplicate checks do not establish scene independence. Reserved test annotations have known coordinate issues requiring an explicit scoring policy. These limitations affect how the results can be interpreted even though the standalone processing reports are omitted from the shared repo.

The image archive and shared runtime companion are external inputs. The historical combined60 notebooks expect the unchanged `home_fire_dfire_20261102` images and `fire_training_shared_v1` companion. No image archive or trained checkpoint is included in Git.

The [dataset template](../configs/data.yaml) must be edited for the local dataset. Use an absolute dataset `path` when passing it to Ultralytics. The standalone verifier also accepts paths relative to the YAML file. General runtime paths in [runtime.yaml](../configs/runtime.yaml) are resolved from the repository root.

For size reporting, `d` is the square root of box area after proportional fit to 640. Current disjoint bins are tiny `d < 16` and small `16 <= d < 32`. Historical tables retain their original small `d < 32` bin, including tiny. The warning prototype's legacy predicted-size bins have different boundary inclusivity; they are not ground-truth recall groups.

The existing 15-video case studies are exploratory. Flame-onset references do not establish smoke-onset timing. Reserved test data and the 20 final-evaluation videos were not used during this cleanup.
