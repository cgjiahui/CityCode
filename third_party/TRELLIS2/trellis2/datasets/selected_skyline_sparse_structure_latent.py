from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SelectedSkylineMultiImageSparseStructureLatent(Dataset):
    """SelectedSkyline Stage-1 latents with prescribed 2-4 image conditions."""

    def __init__(
        self,
        roots: str,
        *,
        split_file: str,
        latent_root: str,
        transforms_root: str,
        image_size: int = 512,
        min_views: int = 2,
        max_views: int = 4,
        deterministic_views: bool = False,
        fixed_num_views: int = 4,
        single_view_candidates: list[str] | None = None,
        max_instances: int | None = None,
        include_footprint: bool = False,
        asset_root: str = "selected_skyline/assets",
        footprint_grid_size: int = 64,
        footprint_volume_resolution: int = 64,
        pretrained_ss_dec: str = (
            "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
        ),
    ):
        Dataset.__init__(self)
        self.root = Path(roots)
        self.split_file = self._resolve(split_file)
        self.latent_root = self._resolve(latent_root)
        self.transforms_root = self._resolve(transforms_root)
        self.image_size = image_size
        self.min_views = min_views
        self.max_views = max_views
        self.deterministic_views = deterministic_views
        self.fixed_num_views = fixed_num_views
        self.single_view_candidates = tuple(single_view_candidates or ())
        self.include_footprint = bool(include_footprint)
        self.asset_root = self._resolve(asset_root)
        self.footprint_grid_size = int(footprint_grid_size)
        self.footprint_volume_resolution = int(footprint_volume_resolution)
        self.normalization = None
        self.value_range = (0, 1)

        if (min_views, max_views) not in ((1, 1), (2, 4)):
            raise ValueError(
                "SelectedSkyline view policy requires either top-oblique-only "
                "(1, 1) or prescribed multi-view (2, 4)"
            )
        valid_fixed_views = (1,) if (min_views, max_views) == (1, 1) else (2, 3, 4)
        if fixed_num_views not in valid_fixed_views:
            raise ValueError(f"fixed_num_views must be one of {valid_fixed_views}")
        if self.single_view_candidates and (min_views, max_views) != (1, 1):
            raise ValueError(
                "single_view_candidates requires the single-view policy (1, 1)"
            )
        if len(set(self.single_view_candidates)) != len(self.single_view_candidates):
            raise ValueError("single_view_candidates must not contain duplicates")
        if self.footprint_grid_size <= 0:
            raise ValueError("footprint_grid_size must be positive")
        if self.footprint_volume_resolution <= 0:
            raise ValueError("footprint_volume_resolution must be positive")

        instances = [
            line.strip()
            for line in self.split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if max_instances is not None:
            instances = instances[:max_instances]
        self.instances = instances

        # SparseStructureLatentVisMixin state, initialized without invoking the
        # StandardDatasetBase constructor used by the public dataset classes.
        self.ss_dec = None
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = None
        self.ss_dec_ckpt = None

    def _loading_ss_dec(self) -> None:
        if self.ss_dec is not None:
            return
        from .. import models

        self.ss_dec = models.from_pretrained(self.pretrained_ss_dec).cuda().eval()

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor, batch_size: int = 4) -> torch.Tensor:
        self._loading_ss_dec()
        decoded = [
            self.ss_dec(z[index:index + batch_size])
            for index in range(0, z.shape[0], batch_size)
        ]
        return torch.cat(decoded, dim=0)

    @torch.no_grad()
    def visualize_sample(self, x_0: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        from ..representations import Voxel
        from ..renderers import VoxelRenderer
        from ..utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0["x_0"]
        occupancy = self.decode_latent(x_0.cuda())
        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.ssaa = 4
        yaw = [value - 16 / 180 * np.pi for value in (0, np.pi / 2, np.pi, 3 * np.pi / 2)]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaw, pitch, 2, 30
        )

        images = []
        for sample in occupancy:
            coords = torch.nonzero(sample[0] > 0, as_tuple=False)
            resolution = sample.shape[-1]
            color = coords / resolution
            representation = Voxel(
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / resolution,
                coords=coords,
                attrs=color,
                layout={"color": slice(0, 3)},
            )
            canvas = torch.zeros(3, 1024, 1024, device="cuda")
            for view_index, (ext, intr) in enumerate(zip(extrinsics, intrinsics)):
                rendered = renderer.render(
                    representation,
                    ext,
                    intr,
                    colors_overwrite=color,
                )["color"]
                row, column = divmod(view_index, 2)
                canvas[:, row * 512:(row + 1) * 512, column * 512:(column + 1) * 512] = rendered
            images.append(canvas)
        return torch.stack(images)

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __len__(self) -> int:
        return len(self.instances)

    def __str__(self) -> str:
        policy = (
            f"fixed {self.fixed_num_views} views"
            if self.deterministic_views
            else (
                "top-oblique only"
                if (self.min_views, self.max_views) == (1, 1)
                else "random 2-4 views"
            )
        )
        if self.single_view_candidates:
            mode = "deterministic" if self.deterministic_views else "random"
            policy = (
                f"{mode} single view from "
                f"{', '.join(self.single_view_candidates)}"
            )
        return (
            f"{self.__class__.__name__}\n"
            f"  - Total instances: {len(self)}\n"
            f"  - Split: {self.split_file}\n"
            f"  - Condition policy: {policy}"
        )

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            result[key] = values if key == "cond" else torch.stack(values, dim=0)
        return result

    def _num_views(self) -> int:
        if self.deterministic_views:
            return self.fixed_num_views
        return int(np.random.randint(self.min_views, self.max_views + 1))

    def _extra_side(self, instance: str, image_paths: dict[str, str]) -> str:
        candidates = sorted(
            key
            for key, value in image_paths.items()
            if key.startswith("side_perspective_")
            and key != "side_perspective_00"
            and Path(value).is_file()
        )
        if not candidates:
            raise FileNotFoundError(f"No additional side view for {instance}")
        if self.deterministic_views:
            digest = hashlib.sha256(instance.encode("utf-8")).digest()
            return candidates[int.from_bytes(digest[:4], "little") % len(candidates)]
        return str(np.random.choice(candidates))

    def _view_names(
        self,
        instance: str,
        image_paths: dict[str, str],
    ) -> list[str]:
        num_views = self._num_views()
        if num_views == 1:
            if self.single_view_candidates:
                missing = [
                    name
                    for name in self.single_view_candidates
                    if name not in image_paths or not Path(image_paths[name]).is_file()
                ]
                if missing:
                    raise FileNotFoundError(
                        f"{instance}: missing single-view candidates: {missing}"
                    )
                if self.deterministic_views:
                    digest = hashlib.sha256(instance.encode("utf-8")).digest()
                    index = int.from_bytes(digest[:4], "little") % len(
                        self.single_view_candidates
                    )
                    return [self.single_view_candidates[index]]
                return [str(np.random.choice(self.single_view_candidates))]
            return ["top_oblique_perspective_40_60"]
        names = ["top_oblique_perspective_40_60", "side_perspective_00"]
        if num_views >= 3:
            names.append("top_perspective")
        if num_views == 4:
            names.append(self._extra_side(instance, image_paths))
        return names

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        pixels = np.asarray(image)

        # The dataset uses a nearly uniform dark background. Crop around pixels
        # that differ from that background to match TRELLIS.2's object-centric
        # image conditioning without running a segmentation network per epoch.
        border = np.concatenate(
            [pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]], axis=0
        )
        background = np.median(border.astype(np.float32), axis=0)
        foreground = np.max(
            np.abs(pixels.astype(np.float32) - background[None, None, :]), axis=2
        ) > 18.0
        ys, xs = np.nonzero(foreground)
        if len(xs) > 0:
            left, right = int(xs.min()), int(xs.max()) + 1
            top, bottom = int(ys.min()), int(ys.max()) + 1
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            half_size = max(right - left, bottom - top) * 0.55
            box = (
                int(center_x - half_size),
                int(center_y - half_size),
                int(center_x + half_size),
                int(center_y + half_size),
            )
            image = image.crop(box)

        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0

    @staticmethod
    def _canonical_camera(
        c2w: list,
        canonical_transform: np.ndarray,
        sphere_radius: float,
        box_scale: float,
    ) -> np.ndarray:
        """Move a rendered source camera into the saved Stage-1 frame."""
        camera = np.asarray(c2w, dtype=np.float64).copy()
        camera[:3, 3] /= sphere_radius
        camera = canonical_transform @ camera
        camera[:3, 3] *= box_scale
        return camera

    @staticmethod
    def _resolve_ground_source_z(metadata: dict, asset_info: dict) -> float:
        box = asset_info["box"]
        box_min = np.asarray(box["min"], dtype=np.float64)
        box_max = np.asarray(box["max"], dtype=np.float64)
        requested = metadata.get("ground_plane_source_z")
        if requested is None:
            return float(box_min[2])

        requested = float(requested)
        height = max(float(box_max[2] - box_min[2]), 1e-6)
        tolerance = max(1.0, 0.05 * height)
        if requested < box_min[2] - tolerance or requested > box_max[2] + tolerance:
            return float(box_min[2])
        return requested

    @staticmethod
    def _source_to_canonical(
        point: np.ndarray,
        metadata: dict,
        canonical_transform: np.ndarray,
        sphere_radius: float,
        box_scale: float,
    ) -> np.ndarray:
        point = np.asarray(point, dtype=np.float64)
        point = point * float(metadata["scale"]) + np.asarray(
            metadata["offset"], dtype=np.float64
        )
        point /= sphere_radius
        point = canonical_transform @ np.r_[point, 1.0]
        return point[:3] * box_scale

    def _ground_sample_grid(
        self,
        instance: str,
        metadata: dict,
        scale_info: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map pure-top footprint pixels into the Stage-1 O-Voxel frame.

        The top camera and ``canonical_transform`` are the same pair used to
        create ``view00.npz``. The mask is ray-cast onto its true source-ground
        plane to preserve image X/Y and perspective scale. If that plane lies
        just below the normalized O-Voxel cube, only the query Z is moved to
        the lowest voxel center; X/Y are never rotated, scaled, or searched.
        """
        canonical_transform = np.asarray(
            scale_info["canonical_transform"], dtype=np.float64
        )
        sphere_radius = float(scale_info["sphere_radius"])
        box_scale = float(scale_info["box_scale"])
        asset_info_path = self.asset_root / instance / "geometry" / "asset_info.json"
        asset_info = json.loads(asset_info_path.read_text(encoding="utf-8"))
        ground_z = self._resolve_ground_source_z(metadata, asset_info)

        source_view = metadata["source_views"]["top_perspective"]
        top_camera = self._canonical_camera(
            source_view["transform_matrix"],
            canonical_transform,
            sphere_radius,
            box_scale,
        )
        rotation_error = float(
            np.max(np.abs(top_camera[:3, :3] - np.eye(3, dtype=np.float64)))
        )
        if rotation_error > 1e-4:
            raise ValueError(
                f"{instance}: top camera is not canonical "
                f"(max rotation error {rotation_error:.6g})"
            )

        # The source ground is source-Z constant. Transform its normal without
        # the tiny source scale, then orient it toward the pure-top camera.
        source_ground_normal = np.cross(
            canonical_transform[:3, 0], canonical_transform[:3, 1]
        )
        normal_norm = float(np.linalg.norm(source_ground_normal))
        if normal_norm < 1e-10:
            raise ValueError(f"{instance}: degenerate canonical ground axes")
        source_ground_normal /= normal_norm
        top_normal = top_camera[:3, 2].copy()
        top_normal /= max(float(np.linalg.norm(top_normal)), 1e-12)
        if float(source_ground_normal @ top_normal) < 0:
            source_ground_normal *= -1.0
        alignment = float(source_ground_normal @ top_normal)
        if alignment < 1.0 - 1e-5:
            raise ValueError(
                f"{instance}: source ground and canonical top disagree "
                f"(dot={alignment:.8f})"
            )
        normal = top_normal
        p0 = self._source_to_canonical(
            np.asarray([0.0, 0.0, ground_z]),
            metadata,
            canonical_transform,
            sphere_radius,
            box_scale,
        )
        plane_d = -float(normal @ p0)

        intrinsics = np.asarray(source_view["intrinsics_K"], dtype=np.float64)
        source_width, source_height = source_view["resolution"]
        size = self.footprint_grid_size
        x = (np.arange(size) + 0.5) * source_width / size - 0.5
        y = (np.arange(size) + 0.5) * source_height / size - 0.5
        yy, xx = np.meshgrid(y, x, indexing="ij")
        directions = np.stack(
            [
                (xx - intrinsics[0, 2]) / intrinsics[0, 0],
                -(yy - intrinsics[1, 2]) / intrinsics[1, 1],
                -np.ones_like(xx),
            ],
            axis=-1,
        )
        directions = directions @ top_camera[:3, :3].T
        origin = top_camera[:3, 3]
        denominator = directions @ normal
        distance = -(origin @ normal + plane_d) / np.where(
            np.abs(denominator) > 1e-10, denominator, np.nan
        )
        points = origin + distance[..., None] * directions
        valid = np.isfinite(points).all(axis=-1) & (distance > 0)

        # The footprint was rendered at physical ground z=0, which may be a
        # few normalized units below the mesh bounding box. Preserve ray-cast
        # X/Y, but sample the closest representable O-Voxel layer in Z.
        half_voxel = 0.5 / self.footprint_volume_resolution
        query_points = points.copy()
        query_points[..., 2] = np.clip(
            query_points[..., 2],
            -0.5 + half_voxel,
            0.5 - half_voxel,
        )
        valid &= (np.abs(query_points[..., :2]) <= 0.50001).all(axis=-1)

        # grid_sample orders a 3-D grid as W, H, D. Decoder tensor axes map
        # from physical canonical x, y, z to z, y, x respectively.
        grid = np.stack(
            [
                2.0 * query_points[..., 2],
                2.0 * query_points[..., 1],
                2.0 * query_points[..., 0],
            ],
            axis=-1,
        )
        grid[~valid] = 2.0
        return (
            grid.astype(np.float32),
            valid.astype(np.float32),
            normal.astype(np.float32),
        )

    def _load_footprint(
        self,
        instance: str,
        metadata: dict,
        latent_dir: Path,
    ) -> dict[str, torch.Tensor]:
        scale_info = json.loads(
            (latent_dir / "view00_scale.json").read_text(encoding="utf-8")
        )
        mask = Image.open(metadata["footprint_mask"]).convert("L")
        mask = mask.resize(
            (self.footprint_grid_size, self.footprint_grid_size),
            Image.Resampling.NEAREST,
        )
        grid, valid, normal = self._ground_sample_grid(
            instance, metadata, scale_info
        )
        return {
            "footprint_cond": torch.from_numpy(
                (np.asarray(mask) >= 128).astype(np.float32)
            )[None],
            "footprint_ground_grid": torch.from_numpy(grid),
            "footprint_ground_valid": torch.from_numpy(valid),
            "ground_plane_normal": torch.from_numpy(normal),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        for offset in range(min(16, len(self.instances))):
            instance = self.instances[(index + offset) % len(self.instances)]
            try:
                latent_dir = self.latent_root / instance
                latent_path = latent_dir / "view00.npz"
                transforms_path = self.transforms_root / instance / "transforms.json"
                latent = np.load(latent_path)
                x_0 = torch.from_numpy(latent["z"].astype(np.float32, copy=False))
                metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
                image_paths = metadata["image_paths"]
                names = self._view_names(instance, image_paths)
                images = [self._load_image(Path(image_paths[name])) for name in names]
                result = {"x_0": x_0, "cond": torch.stack(images, dim=0)}
                if self.include_footprint:
                    result.update(self._load_footprint(instance, metadata, latent_dir))
                return result
            except (FileNotFoundError, KeyError, OSError, ValueError) as error:
                last_error = error
        raise RuntimeError(f"Unable to load dataset item near index {index}") from last_error
