"""Functions that are useful for inference scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from jaxtyping import Float
from projectaria_tools.core import mps  # type: ignore
from projectaria_tools.core.data_provider import create_vrs_data_provider
from projectaria_tools.core.sensor_data import TrackingQuality, VioStatus  # type: ignore
from safetensors import safe_open
from torch import Tensor

from .network import EgoDenoiser, EgoDenoiserConfig
from .tensor_dataclass import TensorDataclass
from .transforms import SE3

TrackingSource = Literal["mps", "vrs"]
"""Where SLAM poses and on-device hand tracking are loaded from.

- ``"mps"``: read MPS sidecar CSVs (``closed_loop_trajectory.csv``,
  ``hand_tracking_results.csv``). Works on any Aria recording that has been
  processed with Machine Perception Services.
- ``"vrs"``: read on-device VIO + on-device hand tracking directly from streams
  baked into the VRS file. Aria Gen 2 only.
"""


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
    def find(
        traj_root: Path,
        tracking_source: TrackingSource = "mps",
    ) -> InferenceTrajectoryPaths:
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

        # Hand tracking CSV lives under mps_*/hand_tracking/. Only used when
        # tracking_source == "mps"; for "vrs" we read from the VRS file.
        wrist_and_palm_poses_csv: Path | None = None
        if tracking_source == "mps":
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
        *,
        source: TrackingSource = "mps",
    ) -> InferenceInputTransforms:
        """Read SLAM device poses + the VRS calibration.

        ``source="mps"`` reads ``closed_loop_trajectory.csv`` from MPS.
        ``source="vrs"`` reads on-device VIO from a stream baked into the VRS
        file (Aria Gen 2 only). The "world" frame for the VRS source is the
        VIO odometry frame, which differs from the MPS world frame, but the
        downstream model only consumes relative transforms so this is fine.
        """
        if source == "mps":
            return InferenceInputTransforms._load_from_mps(
                vrs_path, slam_root_dir, fps
            )
        elif source == "vrs":
            return InferenceInputTransforms._load_from_vrs(vrs_path, fps)
        else:
            raise ValueError(f"Unknown tracking source: {source!r}")

    @staticmethod
    def _load_from_mps(
        vrs_path: Path,
        slam_root_dir: Path,
        fps: int,
    ) -> InferenceInputTransforms:
        """Read device poses from MPS closed-loop trajectory CSV."""
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

    @staticmethod
    def _load_from_vrs(
        vrs_path: Path,
        fps: int,
    ) -> InferenceInputTransforms:
        """Read on-device VIO poses from the ``vio`` stream in the VRS file.

        VIO poses are in the device's odometry frame, not the MPS world frame,
        but EgoAllo only uses relative transforms between consecutive CPF
        frames so the choice of world is irrelevant. Aria Gen 2 only.
        """
        provider = create_vrs_data_provider(str(vrs_path))
        device_calib = provider.get_device_calibration()
        T_device_cpf = device_calib.get_transform_device_cpf().to_matrix()

        vio_stream_id = provider.get_stream_id_from_label("vio")
        if vio_stream_id is None:
            raise RuntimeError(
                f"VRS file {vrs_path} has no on-device 'vio' stream. "
                "On-device VIO is Aria Gen 2 only — fall back to "
                "--tracking-source mps for Gen 1 recordings."
            )

        num_records = provider.get_num_data(vio_stream_id)
        if num_records == 0:
            raise RuntimeError(f"VIO stream in {vrs_path} contains no records.")

        # Pull every VIO record once so we can compute the actual stream rate
        # before subsampling. The full list is small (~20Hz × minutes).
        all_records = [
            provider.get_vio_data_by_index(vio_stream_id, i) for i in range(num_records)
        ]
        valid_records = [
            r
            for r in all_records
            if r.status == VioStatus.VALID and r.pose_quality == TrackingQuality.GOOD
        ]
        if len(valid_records) < 2:
            raise RuntimeError(
                f"VIO stream in {vrs_path} has fewer than 2 valid+good records."
            )

        duration_s = (
            valid_records[-1].capture_timestamp_ns - valid_records[0].capture_timestamp_ns
        ) / 1e9
        aria_fps = len(valid_records) / duration_s
        print(
            f"Loaded num_poses={len(valid_records)} on-device VIO records with "
            f"{aria_fps=:.2f}, visualizing at {fps=}"
        )
        stride = max(1, int(aria_fps // fps))

        Ts_world_device: list[np.ndarray] = []
        Ts_world_cpf: list[np.ndarray] = []
        out_timestamps_secs: list[float] = []
        for i in range(0, len(valid_records), stride):
            r = valid_records[i]
            # T_odometry_device = T_odometry_bodyimu @ T_bodyimu_device.
            T_world_device = (
                r.transform_odometry_bodyimu @ r.transform_bodyimu_device
            ).to_matrix()
            assert T_world_device.shape == (4, 4)
            Ts_world_device.append(T_world_device)
            Ts_world_cpf.append(T_world_device @ T_device_cpf)
            out_timestamps_secs.append(r.capture_timestamp_ns / 1e9)

        return InferenceInputTransforms(
            Ts_world_device=SE3.from_matrix(torch.from_numpy(np.array(Ts_world_device)))
            .parameters()
            .to(torch.float32),
            Ts_world_cpf=SE3.from_matrix(torch.from_numpy(np.array(Ts_world_cpf)))
            .parameters()
            .to(torch.float32),
            pose_timesteps=tuple(out_timestamps_secs),
        )
