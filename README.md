# Training and Testing Pipeline for AI based signal separation of Friedel Pairs in Crystallography

This repository contains U-Net based experiments from a master thesis project on separating overlapping diffraction spots in ESRF DCT data. The code compares three output formulations for the same spot-separation task and includes evaluation notebooks used during thesis work.

## Repository layout

- `one-output/`: one-output experiments. The network predicts one spot and derives the second spot as a residual.
- `two-output/`: two-output experiments. The network predicts two separated spot-intensity channels directly.
- `multi-output/`: multi-output experiments. The network predicts two spot masks and two spot-intensity channels.
- `eva-notebooks/`: evaluation notebooks for model metrics, uncertainty estimates, and visual thesis examples.
- `augment_data/`: augmentation notebook used to create synthetic overlapping spot patches from detected isolated spots.
- `unet_model.py`, `unet_parts.py`: shared U-Net implementation used by the root and multi-output/two-output scripts.
- `architecture_notes.md`: working notes on the training setup and loss design.

## Terminology

Some older file names and notebooks use `pin`. In this project, `pin` and `multi-output` refer to the same model family. The thesis uses `multi-output` because it is the more precise technical term; `pin` remains in some historical names because the method was inspired by that formulation.

## Best model entry point

The best model described for the thesis was trained with:

```bash
python multi-output/train_tversky.py
```

The multi-output Tversky script trains a U-Net with one grayscale input channel and four output channels:

- channels `0:2`: predicted spot masks
- channels `2:4`: predicted spot intensities

The loss is permutation-invariant with respect to the two spots, so the two output spots are not tied to a fixed physical order.

Example training command:

```bash
python multi-output/train_tversky.py \
  --h5-file /path/to/augmented_spots_train.h5 \
  --epochs 200 \
  --batch-size 20 \
  --learning-rate 0.0002 \
  --scale 1.0 \
  --base-features 32 \
  --amp \
  --checkpoint-dir checkpoints/multi-output \
  --log-dir runs/multi-output \
  --preview-dir prediction_previews/multi-output \
  --loss-diagnostic-dir loss_diagnostics/multi-output
```

Adjust batch size, mixed precision, and `--base-features` to match the available GPU memory.

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

Create a Python environment with Python 3.11 or newer. For GPU training, install a PyTorch build that matches your CUDA setup. One possible setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks]"
```

If PyTorch installation fails or installs the wrong CUDA build, install PyTorch manually first using the wheel/index recommended for your system, then rerun the editable install command.

This repository contains a `uv.lock` file from the development environment, but it has not been regenerated after the publication cleanup. Regenerate it before relying on locked installs.

## Path configuration

Most scripts accept explicit path arguments such as `--h5-file`, `--checkpoint-dir`, `--log-dir`, `--preview-dir`, and `--debug-dir`. Prefer those arguments when running on a new machine or cluster.

For `multi-output/train_tversky.py`, you can also copy `project_paths.example.toml` to `project_paths.toml` and edit the HDF5 path:

```bash
cp project_paths.example.toml project_paths.toml
```

`project_paths.toml` is ignored by Git because it is expected to contain machine-specific paths.

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
