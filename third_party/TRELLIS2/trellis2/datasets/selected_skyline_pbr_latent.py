from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .selected_skyline_sparse_structure_latent import (
    SelectedSkylineMultiImageSparseStructureLatent,
)
from .structured_latent_svpbr import SLatPbr, SLatPbrVisMixin
from ..modules.sparse import SparseTensor


class SelectedSkylineMultiImageSLatPbr(SLatPbrVisMixin, Dataset):
    """Top-canonical no-baseplate Stage-3 PBR targets with 2-4 images."""

    def __init__(
        self,
        roots: str,
        *,
        split_file: str,
        pbr_latent_root: str,
        shape_latent_root: str,
        structure_latent_root: str,
        transforms_root: str,
        predicted_shape_root: str | None = None,
        image_size: int = 1024,
        min_views: int = 2,
        max_views: int = 4,
        deterministic_views: bool = False,
        fixed_num_views: int = 4,
        single_view_candidates: list[str] | None = None,
        pbr_slat_normalization: dict[str, list[float]] | None = None,
        shape_slat_normalization: dict[str, list[float]] | None = None,
        max_instances: int | None = None,
        pretrained_pbr_slat_dec: str = "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16",
        pretrained_shape_slat_dec: str = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
    ):
        Dataset.__init__(self)
        self.root = Path(roots)
        self.split_file = self._resolve(split_file)
        self.pbr_latent_root = self._resolve(pbr_latent_root)
        self.shape_latent_root = self._resolve(shape_latent_root)
        self.predicted_shape_root = (
            self._resolve(predicted_shape_root)
            if predicted_shape_root is not None
            else None
        )
        self.resolution = 1024
        self.value_range = (0, 1)
        self.attrs = ["base_color", "metallic", "roughness", "alpha"]
        self.layout = {
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }

        instances = [
            line.strip()
            for line in self.split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.instances = instances[:max_instances] if max_instances is not None else instances
        self.loads = [1] * len(self.instances)
        self.condition_source = SelectedSkylineMultiImageSparseStructureLatent(
            roots,
            split_file=split_file,
            latent_root=structure_latent_root,
            transforms_root=transforms_root,
            image_size=image_size,
            min_views=min_views,
            max_views=max_views,
            deterministic_views=deterministic_views,
            fixed_num_views=fixed_num_views,
            single_view_candidates=single_view_candidates,
        )

        self.pbr_slat_normalization = pbr_slat_normalization
        self.shape_slat_normalization = shape_slat_normalization
        self.pbr_slat_mean, self.pbr_slat_std = self._normalization(pbr_slat_normalization)
        self.shape_slat_mean, self.shape_slat_std = self._normalization(shape_slat_normalization)
        self.pbr_slat_dec = None
        self.pretrained_pbr_slat_dec = pretrained_pbr_slat_dec
        self.pbr_slat_dec_path = None
        self.pbr_slat_dec_ckpt = None
        self.shape_slat_dec = None
        self.pretrained_shape_slat_dec = pretrained_shape_slat_dec
        self.shape_slat_dec_path = None
        self.shape_slat_dec_ckpt = None

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _normalization(config):
        if config is None:
            return None, None
        return (
            torch.tensor(config["mean"], dtype=torch.float32).reshape(1, -1),
            torch.tensor(config["std"], dtype=torch.float32).reshape(1, -1),
        )

    def __len__(self) -> int:
        return len(self.instances)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}\n"
            f"  - Total instances: {len(self)}\n"
            f"  - Split: {self.split_file}\n"
            "  - Target: top-canonical no-baseplate PBR latent\n"
            f"  - Shape condition: {self.predicted_shape_root or self.shape_latent_root}"
        )

    @staticmethod
    def _align_target_to_prediction(
        target_coords: np.ndarray,
        target_feats: np.ndarray,
        predicted_coords: np.ndarray,
        *,
        sample_name: str = "unknown sample",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Order the PBR target on exactly the sparse support predicted by Stage 2."""
        base = 64
        target_keys = (
            target_coords[:, 0].astype(np.int64) * base * base
            + target_coords[:, 1].astype(np.int64) * base
            + target_coords[:, 2].astype(np.int64)
        )
        predicted_keys = (
            predicted_coords[:, 0].astype(np.int64) * base * base
            + predicted_coords[:, 1].astype(np.int64) * base
            + predicted_coords[:, 2].astype(np.int64)
        )
        order = np.argsort(target_keys)
        positions = np.searchsorted(target_keys[order], predicted_keys)
        if (
            len(predicted_keys) == 0
            or np.any(positions >= len(order))
            or np.any(target_keys[order[positions]] != predicted_keys)
        ):
            raise ValueError(
                f"{sample_name}: Stage-2 prediction support is not contained in PBR target"
            )
        selected = order[positions]
        return target_coords[selected], target_feats[selected]

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]], split_size: int | None = None):
        return SLatPbr.collate_fn(batch, split_size=split_size)

    @staticmethod
    def _load_sparse(path: Path, mean, std) -> SparseTensor:
        with np.load(path) as latent:
            coords = torch.from_numpy(latent["coords"].astype(np.int32, copy=False))
            feats = torch.from_numpy(latent["feats"].astype(np.float32, copy=False))
        if mean is not None and std is not None:
            feats = (feats - mean) / std
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
        return SparseTensor(feats, coords)

    def __getitem__(self, index: int) -> dict[str, Any]:
        instance = self.instances[index]
        metadata = json.loads(
            (self.condition_source.transforms_root / instance / "transforms.json").read_text(
                encoding="utf-8"
            )
        )
        image_paths = metadata["image_paths"]
        names = self.condition_source._view_names(instance, image_paths)

        if self.predicted_shape_root is None:
            pbr_z = self._load_sparse(
                self.pbr_latent_root / instance / "view00.npz",
                self.pbr_slat_mean,
                self.pbr_slat_std,
            )
            shape_z = self._load_sparse(
                self.shape_latent_root / instance / "view00.npz",
                self.shape_slat_mean,
                self.shape_slat_std,
            )
        else:
            if len(names) != 1:
                raise ValueError("Predicted Stage-2 shape requires one image per sample")
            with np.load(self.pbr_latent_root / instance / "view00.npz") as latent:
                pbr_coords = latent["coords"].astype(np.int32, copy=False)
                pbr_feats = latent["feats"].astype(np.float32, copy=False)
            prediction_path = self.predicted_shape_root / instance / f"{names[0]}.npz"
            with np.load(prediction_path) as prediction:
                predicted_coords = prediction["coords"].astype(np.int32, copy=False)
                predicted_feats = prediction["feats"].astype(np.float32, copy=False)
                selected_view = str(prediction["selected_view"].item())
            if selected_view != names[0]:
                raise ValueError(
                    f"{instance}: cached Stage-2 view {selected_view!r} does not "
                    f"match selected view {names[0]!r}"
                )
            pbr_coords, pbr_feats = self._align_target_to_prediction(
                pbr_coords,
                pbr_feats,
                predicted_coords,
                sample_name=f"{instance}/{names[0]}",
            )
            if self.pbr_slat_mean is not None and self.pbr_slat_std is not None:
                pbr_feats = (
                    torch.from_numpy(pbr_feats) - self.pbr_slat_mean
                ) / self.pbr_slat_std
            else:
                pbr_feats = torch.from_numpy(pbr_feats)
            batch_column = torch.zeros((len(predicted_coords), 1), dtype=torch.int32)
            coords = torch.from_numpy(predicted_coords)
            pbr_z = SparseTensor(pbr_feats, torch.cat([batch_column, coords], dim=1))
            shape_z = SparseTensor(
                torch.from_numpy(predicted_feats),
                torch.cat([batch_column, coords], dim=1),
            )
        if not torch.equal(shape_z.coords, pbr_z.coords):
            raise ValueError(
                f"{instance}: shape/PBR latent coordinates differ: "
                f"{shape_z.coords.shape} vs {pbr_z.coords.shape}"
            )

        images = [
            self.condition_source._load_image(Path(image_paths[name])) for name in names
        ]
        return {
            "x_0": pbr_z,
            "concat_cond": shape_z,
            "cond": [torch.stack(images, dim=0)],
        }
