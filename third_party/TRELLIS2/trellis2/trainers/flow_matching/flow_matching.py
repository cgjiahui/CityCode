from typing import *
import copy
import json
import os
import time
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...pipelines import samplers 
from ...utils.data_utils import recursive_to_device
from ...utils.general_utils import dict_reduce
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin, MultiImageConditionedMixin


class FlowMatchingTrainer(BasicTrainer):
    """
    Trainer for diffusion model with flow matching objective.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
        snapshot_sampler_steps: int = 50,
        snapshot_guidance_strength: float = 3.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min
        self.snapshot_sampler_steps = snapshot_sampler_steps
        self.snapshot_guidance_strength = snapshot_guidance_strength

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        return {'cond': cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerSampler(self.sigma_min)
    
    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps.
        """
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        snapshot_dataset = getattr(self, 'validation_dataset', None)
        if snapshot_dataset is None:
            snapshot_dataset = copy.deepcopy(self.dataset)
            shuffle = True
        else:
            indices = range(self.rank, len(snapshot_dataset), self.world_size)
            snapshot_dataset = Subset(snapshot_dataset, indices)
            shuffle = False
        collate_source = snapshot_dataset.dataset if isinstance(snapshot_dataset, Subset) else snapshot_dataset
        dataloader = DataLoader(
            snapshot_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_source.collate_fn if hasattr(collate_source, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            noise = torch.randn_like(data['x_0'])
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=self.snapshot_sampler_steps,
                guidance_strength=self.snapshot_guidance_strength,
                verbose=verbose,
            )
            sample.append(res.samples)

        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        return sample_dict

    
class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass


class TextConditionedFlowMatchingCFGTrainer(TextConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        text_cond_model(str): Text conditioning model.
    """
    pass


class ImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass


class MultiImageConditionedFlowMatchingCFGTrainer(
    MultiImageConditionedMixin,
    FlowMatchingCFGTrainer,
):
    """Dense flow-matching trainer with a variable number of image conditions."""

    def __init__(
        self,
        *args,
        validation_data_dir: str | None = None,
        validation_dataset_args: dict | None = None,
        validation_num_timesteps: int = 4,
        validation_seed: int = 20260827,
        steps_per_epoch: int | None = None,
        **kwargs,
    ):
        self.validation_dataset = None
        if validation_data_dir is not None and validation_dataset_args is not None:
            from ...datasets.selected_skyline_sparse_structure_latent import (
                SelectedSkylineMultiImageSparseStructureLatent,
            )

            self.validation_dataset = SelectedSkylineMultiImageSparseStructureLatent(
                validation_data_dir,
                **validation_dataset_args,
            )
        self.validation_num_timesteps = int(validation_num_timesteps)
        self.validation_seed = int(validation_seed)
        self.steps_per_epoch = int(steps_per_epoch or 1)
        self._checkpoint_train_loss_sum = 0.0
        self._checkpoint_train_loss_count = 0
        if self.validation_num_timesteps <= 0:
            raise ValueError("validation_num_timesteps must be positive")
        super().__init__(*args, **kwargs)

    def run_step(self, data_list):
        step_log = super().run_step(data_list)
        train_loss = step_log.get("loss", {}).get("loss")
        if isinstance(train_loss, (int, float)):
            self._checkpoint_train_loss_sum += float(train_loss)
            self._checkpoint_train_loss_count += 1
        return step_log

    @staticmethod
    def _stack_encoded_conditions(cond: list[torch.Tensor]) -> torch.Tensor:
        lengths = {item.shape[0] for item in cond}
        if len(lengths) != 1:
            raise ValueError(
                "Dense multi-image conditioning requires equal token lengths "
                "inside a batch; use batch_size_per_gpu=1 for random 2-4 views."
            )
        return torch.stack(cond, dim=0)

    def get_cond(self, cond, **kwargs):
        encoded = self._stack_encoded_conditions(self.encode_images(cond))
        neg_cond = torch.zeros_like(encoded[:, :1, :])
        return FlowMatchingCFGTrainer.get_cond(
            self,
            encoded,
            neg_cond=neg_cond,
            **kwargs,
        )

    def get_inference_cond(self, cond, **kwargs):
        encoded = self._stack_encoded_conditions(self.encode_images(cond))
        neg_cond = torch.zeros_like(encoded[:, :1, :])
        supervision_keys = {
            "footprint_cond",
            "footprint_ground_grid",
            "footprint_ground_valid",
            "ground_plane_normal",
        }
        kwargs = {key: value for key, value in kwargs.items() if key not in supervision_keys}
        return FlowMatchingCFGTrainer.get_inference_cond(
            self,
            encoded,
            neg_cond=neg_cond,
            **kwargs,
        )

    def _validation_timesteps(self) -> list[float]:
        quantiles = (
            torch.arange(self.validation_num_timesteps, dtype=torch.float64) + 0.5
        ) / self.validation_num_timesteps
        if self.t_schedule["name"] == "uniform":
            values = quantiles
        elif self.t_schedule["name"] == "logitNormal":
            normal = torch.distributions.Normal(0.0, 1.0)
            values = torch.sigmoid(
                normal.icdf(quantiles) * float(self.t_schedule["args"]["std"])
                + float(self.t_schedule["args"]["mean"])
            )
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return [float(value) for value in values]

    def _validation_objective(
        self,
        *,
        x_0: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        noise: torch.Tensor,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        flow_mse = F.mse_loss(pred, target)
        return {"flow_mse": flow_mse, "loss": flow_mse}

    def _write_validation_row(self, row: dict[str, Any]) -> None:
        path = os.path.join(self.output_dir, "validation_history.json")
        history: list[dict[str, Any]] = []
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as file:
                    loaded = json.load(file)
                if isinstance(loaded, list):
                    history = loaded
            except (OSError, ValueError):
                history = []
        history = [item for item in history if int(item.get("step", -1)) != self.step]
        history.append(row)
        history.sort(key=lambda item: int(item.get("step", 0)))
        temporary = f"{path}.tmp"
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2, ensure_ascii=False)
        os.replace(temporary, path)

    @torch.no_grad()
    def run_full_validation(self) -> dict[str, Any] | None:
        """Evaluate every validation object with fixed views, t and noise."""
        if self.validation_dataset is None:
            return None

        denoiser = self.models["denoiser"]
        denoiser_was_training = denoiser.training
        denoiser.eval()
        metric_names: list[str] | None = None
        metric_sums: dict[str, float] = {}
        local_count = 0
        timesteps = self._validation_timesteps()
        amp_context = (
            lambda: torch.autocast(
                device_type="cuda", dtype=self.mix_precision_dtype
            )
            if self.mix_precision_mode == "amp"
            else nullcontext()
        )

        if self.is_master:
            print(
                f"\nValidating all {len(self.validation_dataset)} instances "
                f"at {len(timesteps)} fixed timesteps...",
                flush=True,
            )

        for sample_index in range(self.rank, len(self.validation_dataset), self.world_size):
            sample = self.validation_dataset[sample_index]
            data = self.validation_dataset.collate_fn([sample])
            data = recursive_to_device(data, self.device)
            x_0 = data["x_0"]
            encoded = self._stack_encoded_conditions(self.encode_images(data["cond"]))

            for timestep_index, timestep in enumerate(timesteps):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(
                    self.validation_seed + sample_index * 1009 + timestep_index
                )
                noise = torch.randn(
                    x_0.shape,
                    generator=generator,
                    device=x_0.device,
                    dtype=x_0.dtype,
                )
                t = torch.full(
                    (x_0.shape[0],), timestep, device=x_0.device, dtype=torch.float32
                )
                x_t = self.diffuse(x_0, t, noise=noise)
                with amp_context():
                    pred = denoiser(x_t, t * 1000, encoded)
                    target = self.get_v(x_0, noise, t)
                    terms = self._validation_objective(
                        x_0=x_0,
                        pred=pred,
                        target=target,
                        noise=noise,
                        data=data,
                    )
                if metric_names is None:
                    metric_names = sorted(terms)
                for name in metric_names:
                    metric_sums[name] = metric_sums.get(name, 0.0) + float(
                        terms[name].detach().float().item()
                    )
                local_count += 1

        metric_names = metric_names or ["flow_mse", "loss"]
        packed = torch.tensor(
            [metric_sums.get(name, 0.0) for name in metric_names]
            + [float(local_count)],
            device=self.device,
            dtype=torch.float64,
        )
        if self.world_size > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        train_packed = torch.tensor(
            [
                self._checkpoint_train_loss_sum,
                float(self._checkpoint_train_loss_count),
            ],
            device=self.device,
            dtype=torch.float64,
        )
        if self.world_size > 1:
            dist.all_reduce(train_packed, op=dist.ReduceOp.SUM)
        total_count = max(float(packed[-1].item()), 1.0)
        row = {
            "step": int(self.step),
            "epoch": float(self.step / self.steps_per_epoch),
            "metric_scope": "validation",
            "validation_protocol": "full_set_fixed_views_fixed_t_fixed_noise_v1",
            "num_instances": len(self.validation_dataset),
            "num_timesteps": len(timesteps),
            "timesteps": timesteps,
            "num_evaluations": int(packed[-1].item()),
            "timestamp": time.time(),
        }
        for index, name in enumerate(metric_names):
            row[f"val_{name}"] = float(packed[index].item() / total_count)
        if train_packed[1].item() > 0:
            row["train_loss"] = float(train_packed[0].item() / train_packed[1].item())
            row["train_num_steps"] = int(train_packed[1].item())
            row["train_loss_protocol"] = (
                "ddp_mean_of_observed_training_steps_since_previous_checkpoint_v1"
            )

        if self.is_master:
            self._write_validation_row(row)
            summary = ", ".join(
                f"{key}={value:.6f}"
                for key, value in row.items()
                if key.startswith("val_")
            )
            print(f"Full validation complete: {summary}", flush=True)
        if self.world_size > 1:
            dist.barrier()
        self._checkpoint_train_loss_sum = 0.0
        self._checkpoint_train_loss_count = 0
        if denoiser_was_training:
            denoiser.train()
        return row if self.is_master else None

    @torch.no_grad()
    def snapshot(self, *args, **kwargs):
        self.run_full_validation()
        return super().snapshot(*args, **kwargs)


class FootprintConstrainedMultiImageFlowMatchingCFGTrainer(
    MultiImageConditionedFlowMatchingCFGTrainer
):
    """Stage-1 flow matching supervised by a fixed top-canonical footprint."""

    def __init__(
        self,
        *args,
        footprint_decoder_path: str,
        footprint_bce_weight: float = 0.02,
        footprint_dice_weight: float = 0.05,
        footprint_outside_weight: float = 0.10,
        footprint_occupancy_temperature: float = 1.0,
        footprint_warmup_steps: int = 0,
        **kwargs,
    ):
        self.footprint_bce_weight = float(footprint_bce_weight)
        self.footprint_dice_weight = float(footprint_dice_weight)
        self.footprint_outside_weight = float(footprint_outside_weight)
        self.footprint_occupancy_temperature = float(
            footprint_occupancy_temperature
        )
        self.footprint_warmup_steps = int(footprint_warmup_steps)
        super().__init__(*args, **kwargs)

        from ... import models as model_registry

        self.footprint_decoder = model_registry.from_pretrained(
            footprint_decoder_path
        ).cuda().eval()
        self.footprint_decoder.requires_grad_(False)

    def _project_vertical_occupancy(
        self,
        occupancy_logits: torch.Tensor,
        grid: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Project surface O-Voxels along canonical top-view columns.

        A sparse structure represents object surfaces rather than a filled
        ground slab. The footprint therefore supervises whether each ground
        X/Y column contains a surface, while its X/Y mapping still comes from
        the exact footprint camera ray cast performed by the dataset.
        """
        resolution = occupancy_logits.shape[-1]
        sample_grid = grid.float()[:, None].expand(
            -1, resolution, -1, -1, -1
        ).clone()
        z_centers = (
            2.0
            * (
                torch.arange(
                    resolution,
                    device=occupancy_logits.device,
                    dtype=torch.float32,
                )
                + 0.5
            )
            / resolution
            - 1.0
        )
        # grid_sample's W coordinate is physical canonical Z for this decoder.
        sample_grid[..., 0] = z_centers[None, :, None, None]
        logits = F.grid_sample(
            occupancy_logits.float(),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[:, 0]
        occupancy = torch.sigmoid(
            logits / self.footprint_occupancy_temperature
        ).amax(dim=1)
        return occupancy * valid.float()

    def _footprint_terms(
        self,
        predicted_x0: torch.Tensor,
        data: dict[str, Any],
        *,
        warmup: float,
    ) -> dict[str, torch.Tensor]:
        required = (
            "footprint_cond",
            "footprint_ground_grid",
            "footprint_ground_valid",
            "ground_plane_normal",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise KeyError(f"Footprint-constrained training is missing: {missing}")

        occupancy_logits = self.footprint_decoder(predicted_x0)
        predicted = self._project_vertical_occupancy(
            occupancy_logits,
            data["footprint_ground_grid"],
            data["footprint_ground_valid"],
        )
        target_mask = F.interpolate(
            data["footprint_cond"].float(),
            size=predicted.shape[-2:],
            mode="nearest",
        )[:, 0]
        valid_pixels = data["footprint_ground_valid"].float()

        with torch.autocast(device_type="cuda", enabled=False):
            bce_map = F.binary_cross_entropy(
                predicted.float().clamp(1e-5, 1.0 - 1e-5),
                target_mask.float(),
                reduction="none",
            )
            footprint_bce = (
                bce_map * valid_pixels
            ).sum() / valid_pixels.sum().clamp_min(1.0)
        intersection = (predicted * target_mask * valid_pixels).sum(dim=(1, 2))
        denominator = (predicted * valid_pixels).sum(dim=(1, 2)) + (
            target_mask * valid_pixels
        ).sum(dim=(1, 2))
        footprint_dice = (
            1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        ).mean()
        outside = (1.0 - target_mask) * valid_pixels
        footprint_outside = (
            (predicted * outside).sum(dim=(1, 2))
            / outside.sum(dim=(1, 2)).clamp_min(1.0)
        ).mean()
        footprint_valid_fraction = valid_pixels.mean()
        footprint_target_coverage = (
            (target_mask * valid_pixels).sum(dim=(1, 2))
            / target_mask.sum(dim=(1, 2)).clamp_min(1.0)
        ).mean()
        footprint_total = float(warmup) * (
            self.footprint_bce_weight * footprint_bce
            + self.footprint_dice_weight * footprint_dice
            + self.footprint_outside_weight * footprint_outside
        )
        return {
            "footprint_bce": footprint_bce,
            "footprint_dice": footprint_dice,
            "footprint_outside": footprint_outside,
            "footprint_total": footprint_total,
            "footprint_opacity": predicted.detach().mean(),
            "footprint_valid_fraction": footprint_valid_fraction.detach(),
            "footprint_target_coverage": footprint_target_coverage.detach(),
        }

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        model_cond = self.get_cond(cond)
        pred = self.training_models["denoiser"](x_t, t * 1000, model_cond)
        target = self.get_v(x_0, noise, t)
        flow_mse = F.mse_loss(pred, target)

        predicted_x0 = (1.0 - self.sigma_min) * noise - pred
        warmup = (
            min(1.0, float(self.step + 1) / self.footprint_warmup_steps)
            if self.footprint_warmup_steps > 0
            else 1.0
        )
        footprint = self._footprint_terms(
            predicted_x0, kwargs, warmup=warmup
        )
        terms = edict()
        terms["mse"] = flow_mse
        terms.update(footprint)
        terms["footprint_warmup"] = torch.tensor(
            warmup, device=flow_mse.device
        )
        terms["loss"] = flow_mse + footprint["footprint_total"]
        return terms, {}

    def _validation_objective(
        self,
        *,
        x_0: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        noise: torch.Tensor,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        flow_mse = F.mse_loss(pred, target)
        predicted_x0 = (1.0 - self.sigma_min) * noise - pred
        footprint = self._footprint_terms(predicted_x0, data, warmup=1.0)
        return {
            "flow_mse": flow_mse,
            "footprint_bce": footprint["footprint_bce"],
            "footprint_dice": footprint["footprint_dice"],
            "footprint_outside": footprint["footprint_outside"],
            "footprint_total": footprint["footprint_total"],
            "footprint_opacity": footprint["footprint_opacity"],
            "footprint_valid_fraction": footprint["footprint_valid_fraction"],
            "footprint_target_coverage": footprint["footprint_target_coverage"],
            "loss": flow_mse + footprint["footprint_total"],
        }
