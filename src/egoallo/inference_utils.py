"""Functions that are useful for inference scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from jaxtyping import Float
from projectaria_tools.core import mps  # type: ignore
from projectaria_tools.core.data_provider import create_vrs_data_provider
from safetensors import safe_open
from torch import Tensor

from .fncsmpl import SmplhModel
from .network import EgoDenoiser, EgoDenoiserConfig
from .tensor_dataclass import TensorDataclass
from .transforms import SE3


def load_denoiser(checkpoint_dir: Path) -> EgoDenoiser:
    """Load a denoiser model."""
    checkpoint_dir = checkpoint_dir.absolute()
    experiment_dir = checkpoint_dir.parent

    config = yaml.load(
        (experiment_dir / "model_config.yaml").read_text(), Loader=yaml.Loader
    )
    assert isinstance(config, EgoDenoiserConfig)

    model = EgoDenoiser(config)
    with safe_open(checkpoint_dir / "model.safetensors", framework="pt") as f:  # type: ignore
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    model.load_state_dict(state_dict)

    return model


def betas_from_height(
    body_model: SmplhModel,
    target_height_m: float,
    tol_m: float = 1e-3,
    max_iters: int = 40,
) -> Float[Tensor, "16"]:
    """Solve for a 16-dim SMPL-H beta vector whose T-pose mesh is `target_height_m`
    tall (head-to-foot vertical extent).

    Adjusts beta[0] only (the dominant stature axis); other coefficients stay zero.
    Useful for fitting the rendered body to a known subject height when the
    diffusion model's predicted shape would otherwise default to AMASS-adult.
    """
    device = body_model.v_template.device
    dtype = body_model.v_template.dtype
    num_joints = body_model.get_num_joints()

    identity_quat = torch.zeros(4, dtype=dtype, device=device)
    identity_quat[0] = 1.0
    local_quats = identity_quat.expand(num_joints, 4).contiguous()
    T_world_root = torch.zeros(7, dtype=dtype, device=device)
    T_world_root[0] = 1.0

    def measure_height(beta0: float) -> float:
        betas = torch.zeros(16, dtype=dtype, device=device)
        betas[0] = beta0
        mesh = body_model.with_shape(betas).with_pose(T_world_root, local_quats).lbs()
        verts_z = mesh.verts[..., 2]
        return float(verts_z.max() - verts_z.min())

    lo, hi = -5.0, 5.0
    h_lo = measure_height(lo)
    h_hi = measure_height(hi)
    increases_with_beta = h_hi > h_lo

    mid = 0.5 * (lo + hi)
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        h_mid = measure_height(mid)
        if abs(h_mid - target_height_m) < tol_m:
            break
        too_short = h_mid < target_height_m
        if too_short == increases_with_beta:
            lo = mid
        else:
            hi = mid

    betas = torch.zeros(16, dtype=dtype, device=device)
    betas[0] = mid
    return betas


@dataclass(frozen=True)
class InferenceTrajectoryPaths:
    """Paths for running EgoAllo on a single sequence from Project Aria.

    Expected directory layout::

        traj_root/
            video.vrs
            mps_video_vrs/
                slam/
                    closed_loop_trajectory.csv
                    semidense_points.csv.gz   # or global_points.csv.gz
                    ...
                hand_tracking/
                    hand_tracking_results.csv
                    ...
            hamer_outputs.pkl          # optional, for HaMeR guidance
            splat.ply / scene.splat    # optional, for visualization
    """

    vrs_file: Path
    slam_root_dir: Path
    points_path: Path
    hamer_outputs: Path | None
    wrist_and_palm_poses_csv: Path | None
    splat_path: Path | None

    @staticmethod
    def find(traj_root: Path) -> InferenceTrajectoryPaths:
        vrs_files = tuple(traj_root.glob("*.vrs"))
        assert len(vrs_files) == 1, (
            f"Expected exactly one VRS file in {traj_root}, found {len(vrs_files)}"
        )

        # Locate the MPS output root (e.g. mps_video_vrs/).
        mps_dirs = [
            d
            for d in traj_root.iterdir()
            if d.is_dir() and d.name.startswith("mps_")
        ]
        assert len(mps_dirs) == 1, (
            f"Expected exactly one mps_* directory in {traj_root}, found {mps_dirs}"
        )
        mps_root = mps_dirs[0]

        # SLAM artifacts live under mps_*/slam/.
        slam_dir = mps_root / "slam"
        assert slam_dir.is_dir(), f"Missing SLAM directory: {slam_dir}"

        points_path = slam_dir / "semidense_points.csv.gz"
        if not points_path.exists():
            points_path = slam_dir / "global_points.csv.gz"
        assert points_path.exists(), (
            f"No points file (semidense_points.csv.gz or global_points.csv.gz) "
            f"found in {slam_dir}"
        )

        hamer_outputs = traj_root / "hamer_outputs.pkl"
        if not hamer_outputs.exists():
            hamer_outputs = None

        # Hand tracking CSV lives under mps_*/hand_tracking/.
        wrist_and_palm_poses_csv: Path | None = None
        ht_dir = mps_root / "hand_tracking"
        ht_csv = ht_dir / "hand_tracking_results.csv"
        if not ht_csv.exists():
            ht_csv = ht_dir / "wrist_and_palm_poses.csv"
        if ht_csv.exists():
            wrist_and_palm_poses_csv = ht_csv

        splat_path = traj_root / "splat.ply"
        if not splat_path.exists():
            splat_path = traj_root / "scene.splat"
        if not splat_path.exists():
            print("No scene splat found.")
            splat_path = None
        else:
            print("Found splat at", splat_path)

        return InferenceTrajectoryPaths(
            vrs_file=vrs_files[0],
            slam_root_dir=slam_dir,
            points_path=points_path,
            hamer_outputs=hamer_outputs,
            wrist_and_palm_poses_csv=wrist_and_palm_poses_csv,
            splat_path=splat_path,
        )


class InferenceInputTransforms(TensorDataclass):
    """Some relevant transforms for inference."""

    Ts_world_cpf: Float[Tensor, "timesteps 7"]
    Ts_world_device: Float[Tensor, "timesteps 7"]
    pose_timesteps: tuple[float, ...]

    @staticmethod
    def load(
        vrs_path: Path,
        slam_root_dir: Path,
        fps: int = 30,
    ) -> InferenceInputTransforms:
        """Read some useful transforms via MPS + the VRS calibration."""
        closed_loop_path = slam_root_dir / "closed_loop_trajectory.csv"
        if not closed_loop_path.exists():
            # Aria digital twins.
            closed_loop_path = slam_root_dir / "aria_trajectory.csv"
        closed_loop_traj = mps.read_closed_loop_trajectory(str(closed_loop_path))  # type: ignore

        provider = create_vrs_data_provider(str(vrs_path))
        device_calib = provider.get_device_calibration()
        T_device_cpf = device_calib.get_transform_device_cpf().to_matrix()

        # Print sensor extrinsics for debugging Gen 1 vs Gen 2 differences.
        print("=== Device calibration extrinsics ===")
        print(f"T_device_cpf:\n{T_device_cpf}")
        for sensor_label in ["camera-rgb", "camera-slam-left", "camera-slam-right"]:
            t = device_calib.get_transform_device_sensor(sensor_label)
            if t is not None:
                print(f"T_device_{sensor_label}:\n{t.to_matrix()}")
        print("=====================================")

        # Get downsampled CPF frames.
        aria_fps = len(closed_loop_traj) / (
            closed_loop_traj[-1].tracking_timestamp.total_seconds()
            - closed_loop_traj[0].tracking_timestamp.total_seconds()
        )
        num_poses = len(closed_loop_traj)
        print(f"Loaded {num_poses=} with {aria_fps=}, visualizing at {fps=}")
        Ts_world_device = []
        Ts_world_cpf = []
        out_timestamps_secs = []
        for i in range(0, num_poses, int(aria_fps // fps)):
            T_world_device = closed_loop_traj[i].transform_world_device.to_matrix()
            assert T_world_device.shape == (4, 4)
            Ts_world_device.append(T_world_device)
            Ts_world_cpf.append(T_world_device @ T_device_cpf)
            out_timestamps_secs.append(
                closed_loop_traj[i].tracking_timestamp.total_seconds()
            )

        return InferenceInputTransforms(
            Ts_world_device=SE3.from_matrix(torch.from_numpy(np.array(Ts_world_device)))
            .parameters()
            .to(torch.float32),
            Ts_world_cpf=SE3.from_matrix(torch.from_numpy(np.array(Ts_world_cpf)))
            .parameters()
            .to(torch.float32),
            pose_timesteps=tuple(out_timestamps_secs),
        )

