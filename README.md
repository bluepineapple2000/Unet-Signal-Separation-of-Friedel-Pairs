# Training and Testing Pipeline for AI based signal separation of Friedel Pairs in Crystallography

This repository contains U-Net based experiments from a master thesis project on separating overlapping diffraction spots in ESRF DCT data. The code compares three output formulations for the same spot-separation task and includes evaluation notebooks used during thesis work.

## Repository layout

- `one-output/`: one-output experiments. The network predicts one spot and derives the second spot as a residual.
- `two-output/`: two-output experiments. The network predicts two separated spot-intensity channels directly.
- `multi-output/`: multi-output experiments. The network predicts two spot masks and two spot-intensity channels.
- `eva-notebooks/`: evaluation notebooks for model metrics, uncertainty estimates, and visual thesis examples.
- `augment_data/`: augmentation notebook used to create synthetic overlapping spot patches from detected isolated spots.
- `unet_model.py`, `unet_parts.py`: canonical shared U-Net implementation used by all training scripts. The scripts in subdirectories prepend the repository root to `sys.path` so these root files are imported instead of older local copies.
- `architecture_notes.md`: working notes on the training setup and loss design.

## Terminology

Some older file names and notebooks use `pin`. In this project, `pin` and `multi-output` refer to the same model family. The thesis uses `multi-output` because it is the more precise technical term; `pin` remains in some historical names because the method was inspired by that formulation.

## Best model entry point

The best model described for the thesis was trained with:

```bash
uv run python multi-output/train_tversky.py
```

The multi-output Tversky script trains a U-Net with one grayscale input channel and four output channels:

- channels `0:2`: predicted spot masks
- channels `2:4`: predicted spot intensities

The loss is permutation-invariant with respect to the two spots, so the two output spots are not tied to a fixed physical order.

Example training command:

```bash
uv run python multi-output/train_tversky.py \
  --epochs 200 \
  --batch-size 20 \
  --learning-rate 0.0002 \
  --scale 1.0 \
  --base-features 32 \
  --amp
```

By default, training scripts read `data/augmented_spots_train.h5`. The `data/augmented_spots_test.h5` file is reserved for testing/evaluation and is not used as a training default. Adjust batch size, mixed precision, and `--base-features` to match the available GPU memory.

## Data requirements

The code cannot run end-to-end from a fresh clone without the underlying ESRF DCT data and preprocessing outputs. These data are not published in this repository.

The Friedel-pair recognition step is also not included. The training scripts assume that preprocessing and augmentation have already produced HDF5 files with separated target spot images, and for the multi-output model also separated spot masks.

Expected HDF5 sample structure for one-output and two-output training:

```text
sample_group/
  image        [H, W]       overlapped input intensity image
  spot_images  [2, H, W]    separated target spot intensities
```

Expected HDF5 sample structure for multi-output training:

```text
sample_group/
  image        [H, W]       overlapped input intensity image
  spot_images  [2, H, W]    separated target spot intensities
  spot_masks   [2, H, W]    separated binary target spot masks
```

The augmentation notebook in `augment_data/augmentation.ipynb` documents the local workflow that was used to create these files, but it still depends on data and detection results that are not part of this repository.

## Installation

Create a Python environment with Python 3.11 or newer. The project is set up for `uv`:

```bash
uv sync --extra notebooks
```

All `train_*.py` scripts are implemented with PyTorch rather than TensorFlow. They also use PyTorch's TensorBoard writer for training logs.

Run scripts with `uv run ...`, or activate the environment first with `source .venv/bin/activate` and then use `python ...`. If the lock file is stale after dependency changes, regenerate it with `uv lock --upgrade` and rerun `uv sync --extra notebooks`. For GPU training, make sure the PyTorch build in `pyproject.toml` matches the CUDA setup on the machine.

## Path configuration

Training scripts default to project-level paths so they behave the same whether launched from the repository root or from inside a subdirectory. The default input is `data/augmented_spots_train.h5`. Outputs are grouped by script, for example `multi-output/train_tversky.py` writes to:

```text
checkpoints/multi-output/tversky/
runs/multi-output/tversky/
prediction_previews/multi-output/tversky/
loss_diagnostics/multi-output/tversky/
debug_logs/multi-output/tversky/
```

Most scripts accept explicit path arguments such as `--h5-file`, `--checkpoint-dir`, `--log-dir`, `--preview-dir`, and `--debug-dir`. Use those arguments to override the defaults on a new machine or cluster. `multi-output/train_tversky.py` can also read `project_paths.toml`; copy `project_paths.example.toml` and edit `input_h5` if needed:

```bash
cp project_paths.example.toml project_paths.toml
```

`project_paths.toml` and `training_paths.toml` are ignored by Git because they are expected to contain machine-specific paths.

## Evaluation notebooks

The notebooks in `eva-notebooks/` were used on a different server where paths and checkpoint locations differed from this repository layout. Each notebook has a configuration cell near the top where paths such as `DATA_PATH`, `MODEL_PATHS`, `MODEL_SEARCH_ROOTS`, and `OUTPUT_DIR` should be edited.

The notebooks are useful as analysis templates, but their current path defaults should be treated as examples rather than a guaranteed fresh-clone workflow.

## Current limitations

- ESRF DCT data are required but not included.
- Friedel-pair recognition is required upstream but not included.
- Checkpoints for the thesis-best model are not included.
- The code was tested on another server with slightly different paths. The current repository layout has not been fully revalidated end-to-end with public paths.
- Notebooks may still contain server-specific configuration cells and should be reviewed before publication.

## Contact and citation

Author: TODO: Your name

Contact: TODO: your.email@example.com

Institution: TODO: University / institute / beamline group

Thesis: TODO: Thesis title, degree program, year

Citation: Citation metadata are provided in [CITATION.cff](CITATION.cff). Replace the placeholders before public release, and add a DOI if the repository is archived on Zenodo or another repository.

## License

This project is licensed under the [MIT License](LICENSE.md). Before public release, replace the placeholder copyright holder in `LICENSE.md`.
