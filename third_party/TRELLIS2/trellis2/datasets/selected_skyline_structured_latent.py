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
from .structured_latent import SLat, SLatVisMixin


class SelectedSkylineMultiImageSLatShape(SLatVisMixin, Dataset):
    """SelectedSkyline Stage-2 shape latents with the Stage-1 2-4 view policy."""

    def __init__(
        self,
        roots: str,
        *,
        split_file: str,
        latent_root: str,
        transforms_root: str,
        structure_latent_root: str,
        predicted_structure_root: str | None = None,
        image_size: int = 1024,
        min_views: int = 2,
        max_views: int = 4,
        deterministic_views: bool = False,
        fixed_num_views: int = 4,
        single_view_candidates: list[str] | None = None,
        normalization: dict[str, list[float]] | None = None,
        max_instances: int | None = None,
        pretrained_slat_dec: str = (
            "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
        ),
    ):
        Dataset.__init__(self)
        self.root = Path(roots)
        self.split_file = self._resolve(split_file)
        self.latent_root = self._resolve(latent_root)
        self.transforms_root = self._resolve(transforms_root)
        self.predicted_structure_root = (
            self._resolve(predicted_structure_root)
            if predicted_structure_root is not None
            else None
        )
        self.normalization = normalization
        self.value_range = (0, 1)

        instances = [
            line.strip()
            for line in self.split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if max_instances is not None:
            instances = instances[:max_instances]
        self.instances = instances
        # Batch size is one per GPU, so load balancing is immaterial; the
        # distributed sampler still requires one positive load per sample.
        self.loads = [1] * len(self.instances)

        # Reuse the exact Stage-1 image crop and prescribed view selection.
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

        self.mean = None
        self.std = None
        if normalization is not None:
            self.mean = torch.tensor(normalization["mean"], dtype=torch.float32).reshape(1, -1)
            self.std = torch.tensor(normalization["std"], dtype=torch.float32).reshape(1, -1)

        self.slat_dec = None
        self.pretrained_slat_dec = pretrained_slat_dec
        self.slat_dec_path = None
        self.slat_dec_ckpt = None

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __len__(self) -> int:
        return len(self.instances)

    def __str__(self) -> str:
        policy = str(self.condition_source).split("Condition policy: ", 1)[-1]
        return (
            f"{self.__class__.__name__}\n"
            f"  - Total instances: {len(self)}\n"
            f"  - Split: {self.split_file}\n"
            f"  - Condition policy: {policy}\n"
            f"  - Latents: {self.latent_root}\n"
            f"  - Predicted structure: {self.predicted_structure_root}"
        )

    @staticmethod
    def _restrict_to_predicted_structure(
        coords: torch.Tensor,
        feats: torch.Tensor,
        predicted_coords: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep supervised shape tokens supported by the Stage-1 prediction."""
        predicted = torch.from_numpy(
            predicted_coords.astype(np.int64, copy=False)
        )
        target = coords.to(torch.int64)
        base = 64
        predicted_keys = (
            predicted[:, 0] * base * base
            + predicted[:, 1] * base
            + predicted[:, 2]
        )
        target_keys = (
            target[:, 0] * base * base
            + target[:, 1] * base
            + target[:, 2]
        )
        keep = torch.isin(target_keys, predicted_keys)
        if int(keep.sum()) == 0:
            raise ValueError(
                "Stage-1 prediction has no supervised Stage-2 tokens"
            )
        return coords[keep], feats[keep]

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]], split_size: int | None = None):
        return SLat.collate_fn(batch, split_size=split_size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        for offset in range(min(16, len(self.instances))):
            instance = self.instances[(index + offset) % len(self.instances)]
            try:
                latent_path = self.latent_root / instance / "view00.npz"
                transforms_path = self.transforms_root / instance / "transforms.json"
                with np.load(latent_path) as latent:
                    coords = torch.from_numpy(
                        latent["coords"].astype(np.int32, copy=False)
                    )
                    feats = torch.from_numpy(
                        latent["feats"].astype(np.float32, copy=False)
                    )
                if self.mean is not None and self.std is not None:
                    feats = (feats - self.mean) / self.std

                metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
                image_paths = metadata["image_paths"]
                if self.predicted_structure_root is not None:
                    names = self.condition_source._view_names(instance, image_paths)
                    if len(names) != 1:
                        raise ValueError(
                            "Predicted Stage-1 support requires one image per sample"
                        )
                    prediction_path = (
                        self.predicted_structure_root / instance / f"{names[0]}.npz"
                    )
                    with np.load(prediction_path) as prediction:
                        predicted_coords = prediction["coords"]
                        selected_view = str(prediction["selected_view"].item())
                    if selected_view != names[0] or selected_view not in image_paths:
                        raise KeyError(
                            f"{instance}: cached view {selected_view!r} does not match "
                            f"selected view {names[0]!r}"
                        )
                    coords, feats = self._restrict_to_predicted_structure(
                        coords, feats, predicted_coords
                    )
                else:
                    names = self.condition_source._view_names(instance, image_paths)
                images = [
                    self.condition_source._load_image(Path(image_paths[name]))
                    for name in names
                ]
                return {
                    "coords": coords,
                    "feats": feats,
                    # SLat.collate_fn flattens one outer list per sample while
                    # preserving each sample's (V, C, H, W) image tensor.
                    "cond": [torch.stack(images, dim=0)],
                }
            except (FileNotFoundError, KeyError, OSError, ValueError) as error:
                last_error = error
        raise RuntimeError(f"Unable to load dataset item near index {index}") from last_error
