# Publication Checklist

Use this list before making the repository public.

## Required before release

- Replace the placeholder copyright holder in `LICENSE.md`.
- Confirm what can be said publicly about the ESRF DCT data and add the correct dataset, beamline, and acknowledgement text.
- Add final author, institution, contact, thesis title, and citation information to `README.md`.
- Replace placeholders in `CITATION.cff`, especially author name, email, repository URL, and optional DOI/release date.
- Regenerate `uv.lock` after the `pyproject.toml` cleanup, or remove the lockfile and document a plain `pip` install only.
- Run a fresh-clone smoke test with a small local HDF5 file and the commands from the README.
- Review all notebooks for server-specific paths and clear outputs that expose private paths or bulky embedded images.

## Strongly recommended

- Add a tiny synthetic or anonymized example HDF5 file, if data policy allows, so notebooks can be executed by external users.
- Add a minimal test that builds a mock HDF5 file and checks dataset loading plus one forward/loss pass.
- Publish the thesis-best checkpoint or document why it cannot be redistributed.
- Record exact training metadata for the thesis-best run: commit, hardware, CUDA/PyTorch versions, epochs, batch size, learning rate, scale, `base_features`, and final metrics.
- Decide whether historical `pin` names should remain or be renamed to `multi-output`; if they remain, keep the terminology note in the README.
- Factor duplicated dataset/path/logging code into a small shared module if the project will be maintained further.
- Add screenshots or static result figures if notebooks are published with cleared outputs.
