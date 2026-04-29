# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

EgoAllo estimates 3D human body pose, hand motion, and height from egocentric (first-person) camera data using a diffusion-based motion prior. It combines SLAM pose estimates, hand detection (HaMeR), and a learned motion prior to output full-body and hand motion trajectories. Research code for the paper "Estimating Body and Hand Motion in an Ego-sensed World" (arXiv:2410.03665).

## Setup

Requires Python >= 3.12.

```bash
pip install -e .

# For inference with guidance optimization (JAX + CUDA):
pip install "jax[cuda12]==0.6.1"
pip install git+https://github.com/brentyi/jaxls.git

# For HaMeR hand detection:
pip install git+https://github.com/brentyi/hamer_helper.git
```

SMPL-H model file expected at `./data/smplh/neutral/model.npz` (download from MANO project).

`bash download_checkpoint_and_data.sh` fetches the model checkpoint (unzips to `./egoallo_checkpoint_april13/checkpoints_3000000/`, the default `--checkpoint-dir`) and example trajectories.

## Commands

```bash
# Type checking (this is the CI check)
pyright

# Linting
ruff check .

# Training (uses HuggingFace accelerate)
python 1_train_motion_prior.py

# Inference
python 3_aria_inference.py --traj-root ./egoallo_example_trajectories/coffeemachine

# Visualization (launches Viser web viewer)
python 4_visualize_outputs.py --search-root-dir ./egoallo_example_trajectories
```

No unit test suite exists; validation is manual/visual.

## Architecture

### Pipeline (numbered scripts)

0a/0b: Preprocess AMASS mocap data (NPZ -> HDF5) -> 1: Train diffusion model -> 2: Run HaMeR hand detection on VRS -> 3: Full inference -> 4: Visualize -> 5: Evaluate metrics

### Core modules (`src/egoallo/`)

- **`network.py`** - `EgoDenoiser`: Transformer encoder-decoder diffusion model with rotary positional embeddings. `EgoDenoiseTraj` packs/unpacks body shape, rotations, contacts, and hand rotations into flat tensors.
- **`sampling.py`** - DDIM sampling with windowed denoising and overlap stitching. Integrates guidance optimization during and after diffusion.
- **`training_loss.py`** - Diffusion loss computation with per-component weighting (betas, body rotmats, contacts, hand rotmats). Time-weighted to match epsilon prediction.
- **`fncsmpl.py`** - Stateless SMPL-H body model wrapper. Chain: `load()` -> `with_shape(betas)` -> `with_pose(root, quats)` -> `lbs()` (mesh). `fncsmpl_jax.py` is the JAX equivalent for optimization.
- **`guidance_optimizer_jax.py`** - Levenberg-Marquardt constraint optimization in JAX/jaxls. Applies foot contact, hand position, and hand orientation constraints.
- **`transforms/`** - SO(3) and SE(3) Lie group operations. Quaternion convention: (w,x,y,z).
- **`data/`** - `EgoAmassHdf5Dataset` loads training data from HDF5. `EgoTrainingData` is the main data structure with transforms, quaternions, contacts, shape params.
- **`hand_detection_structs.py`** - Structures for HaMeR and Aria hand detection results.
- **`inference_utils.py`** - `InferenceTrajectoryPaths.find()` resolves the on-disk layout for an inference trajectory; `InferenceInputTransforms` loads VRS + MPS poses downsampled to a target fps.
- **`vis_helpers.py`** - 3D visualization with Viser (Gaussian splats, PLY point clouds, SMPL-H meshes).

### Inference trajectory layout

`InferenceTrajectoryPaths.find()` expects exactly one `mps_*/` directory under `--traj-root`, with SLAM artifacts under `mps_*/slam/` (`closed_loop_trajectory.csv`, `semidense_points.csv.gz`) and hand tracking under `mps_*/hand_tracking/`. There is no longer a fallback to on-device VRS streams — MPS output is required. Optional `hamer_outputs.pkl` lives at the trajectory root.

### Key coordinate frames

- **CPF** (Central Pupil Frame): Head-centered egocentric view
- **device**: Aria VRS device frame
- **world**: Global frame from SLAM

### Type annotations

Uses `jaxtyping` for tensor shape annotations (e.g., `Float[Tensor, "batch time 3"]`) with `typeguard` runtime checking.

### CLI

Numbered scripts parse args via `tyro.cli(Args)` over a `@dataclasses.dataclass`; pass `--help` to any script to see the full options derived from the dataclass fields and docstrings.

## Tool configuration

- **Pyright**: Ignores `**/preprocessing/**` and `0a_preprocess_training_data.py`
- **Ruff**: Selects E, F, PLC, PLE, PLR, PLW rules with various ignores (see pyproject.toml)
