from typing import *
import os
import copy
import functools
import json
import time
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin, MultiImageConditionedMixin


class SparseFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective.
    
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
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
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
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=num_samples,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        data = next(iter(dataloader))

        # inference
        sampler = self.get_sampler()
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch_data = {k: v[i:i+batch_size] for k, v in data.items()}
            batch_data = recursive_to_device(batch_data, 'cuda')
            noise = batch_data['x_0'].replace(torch.randn_like(batch_data['x_0'].feats))
            cond_vis.append(self.vis_cond(**batch_data))
            del batch_data['x_0']
            args = self.get_inference_cond(**batch_data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=12, guidance_strength=3.0, verbose=verbose,
            )
            sample.append(res.samples)
        sample = sp.sparse_cat(sample)
        
        sample_gt = {k: v for k, v in data.items()}
        sample = {k: v if k != 'x_0' else sample for k, v in data.items()}
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        return sample_dict


class SparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective and classifier-free guidance.
    
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


class TextConditionedSparseFlowMatchingCFGTrainer(TextConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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


class ImageConditionedSparseFlowMatchingCFGTrainer(ImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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


class MultiImageConditionedSparseFlowMatchingCFGTrainer(MultiImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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


class SelectedSkylineMultiImageSparseFlowMatchingCFGTrainer(
    MultiImageConditionedSparseFlowMatchingCFGTrainer
):
    """Stage-2 sparse flow matching with checkpoint-aligned full validation."""

    def __init__(
        self,
        *args,
        validation_data_dir: str | None = None,
        validation_dataset_args: dict | None = None,
        validation_num_timesteps: int = 4,
        validation_seed: int = 20260828,
        steps_per_epoch: int | None = None,
        parent_stage1_checkpoint: str | None = None,
        **kwargs,
    ):
        self.validation_dataset = None
        if validation_data_dir is not None and validation_dataset_args is not None:
            from ...datasets.selected_skyline_structured_latent import (
                SelectedSkylineMultiImageSLatShape,
            )

            self.validation_dataset = SelectedSkylineMultiImageSLatShape(
                validation_data_dir,
                **validation_dataset_args,
            )
        self.validation_num_timesteps = int(validation_num_timesteps)
        self.validation_seed = int(validation_seed)
        self.steps_per_epoch = int(steps_per_epoch or 1)
        self.parent_stage1_checkpoint = parent_stage1_checkpoint
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
        if self.validation_dataset is None:
            return None

        denoiser = self.models["denoiser"]
        denoiser_was_training = denoiser.training
        denoiser.eval()
        metric_sum = 0.0
        local_count = 0
        timesteps = self._validation_timesteps()
        amp_context = (
            lambda: torch.autocast(device_type="cuda", dtype=self.mix_precision_dtype)
            if self.mix_precision_mode == "amp"
            else nullcontext()
        )

        if self.is_master:
            print(
                f"\nValidating all {len(self.validation_dataset)} Stage-2 instances "
                f"at {len(timesteps)} fixed timesteps...",
                flush=True,
            )

        for sample_index in range(self.rank, len(self.validation_dataset), self.world_size):
            sample = self.validation_dataset[sample_index]
            data = self.validation_dataset.collate_fn([sample])
            data = recursive_to_device(data, self.device)
            x_0 = data["x_0"]
            encoded = self.encode_images(data["cond"])

            for timestep_index, timestep in enumerate(timesteps):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(
                    self.validation_seed + sample_index * 1009 + timestep_index
                )
                noise = x_0.replace(
                    torch.randn(
                        x_0.feats.shape,
                        generator=generator,
                        device=x_0.device,
                        dtype=x_0.dtype,
                    )
                )
                t = torch.full(
                    (x_0.shape[0],), timestep, device=x_0.device, dtype=torch.float32
                )
                x_t = self.diffuse(x_0, t, noise=noise)
                with amp_context():
                    pred = denoiser(x_t, t * 1000, encoded)
                    target = self.get_v(x_0, noise, t)
                    loss = F.mse_loss(pred.feats, target.feats)
                metric_sum += float(loss.detach().float().item())
                local_count += 1

        packed = torch.tensor(
            [metric_sum, float(local_count)],
            device=self.device,
            dtype=torch.float64,
        )
        train_packed = torch.tensor(
            [self._checkpoint_train_loss_sum, float(self._checkpoint_train_loss_count)],
            device=self.device,
            dtype=torch.float64,
        )
        if self.world_size > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_packed, op=dist.ReduceOp.SUM)
        total_count = max(float(packed[1].item()), 1.0)
        row = {
            "step": int(self.step),
            "epoch": float(self.step / self.steps_per_epoch),
            "metric_scope": "validation",
            "validation_protocol": "full_set_fixed_views_fixed_t_fixed_noise_v1",
            "num_instances": len(self.validation_dataset),
            "num_timesteps": len(timesteps),
            "timesteps": timesteps,
            "num_evaluations": int(packed[1].item()),
            "val_flow_mse": float(packed[0].item() / total_count),
            "val_loss": float(packed[0].item() / total_count),
            "parent_stage1_checkpoint": self.parent_stage1_checkpoint,
            "timestamp": time.time(),
        }
        if train_packed[1].item() > 0:
            row["train_loss"] = float(train_packed[0].item() / train_packed[1].item())
            row["train_num_steps"] = int(train_packed[1].item())
            row["train_loss_protocol"] = (
                "ddp_mean_of_observed_training_steps_since_previous_checkpoint_v1"
            )

        if self.is_master:
            self._write_validation_row(row)
            print(
                f"Full Stage-2 validation complete: val_loss={row['val_loss']:.6f}",
                flush=True,
            )
        if self.world_size > 1:
            dist.barrier()
        self._checkpoint_train_loss_sum = 0.0
        self._checkpoint_train_loss_count = 0
        if denoiser_was_training:
            denoiser.train()
        return row if self.is_master else None

    @torch.no_grad()
    def snapshot_dataset(self, *args, **kwargs):
        # The no-base GT decoder exposes geometry but no color render. Loss
        # monitoring and final post-selection GLB generation are handled
        # separately, so the generic color-preview hook is intentionally off.
        return None

    @torch.no_grad()
    def snapshot(self, *args, **kwargs):
        # Stage-2 checkpoint monitoring needs the complete loss, not a costly
        # stochastic mesh decode. Final GLBs are generated after model selection.
        return self.run_full_validation()


class SelectedSkylineMultiImagePbrSparseFlowMatchingCFGTrainer(
    SelectedSkylineMultiImageSparseFlowMatchingCFGTrainer
):
    """Stage-3 PBR flow matching with shape concat-conditioning validation."""

    def __init__(
        self,
        *args,
        validation_data_dir: str | None = None,
        validation_dataset_args: dict | None = None,
        parent_stage1_checkpoint: str | None = None,
        parent_stage2_checkpoint: str | None = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            validation_data_dir=None,
            validation_dataset_args=None,
            parent_stage1_checkpoint=parent_stage1_checkpoint,
            **kwargs,
        )
        self.parent_stage2_checkpoint = parent_stage2_checkpoint
        if validation_data_dir is not None and validation_dataset_args is not None:
            from ...datasets.selected_skyline_pbr_latent import (
                SelectedSkylineMultiImageSLatPbr,
            )

            self.validation_dataset = SelectedSkylineMultiImageSLatPbr(
                validation_data_dir,
                **validation_dataset_args,
            )

    @torch.no_grad()
    def run_full_validation(self) -> dict[str, Any] | None:
        if self.validation_dataset is None:
            return None

        denoiser = self.models["denoiser"]
        was_training = denoiser.training
        denoiser.eval()
        metric_sum = 0.0
        local_count = 0
        timesteps = self._validation_timesteps()
        amp_context = (
            lambda: torch.autocast(device_type="cuda", dtype=self.mix_precision_dtype)
            if self.mix_precision_mode == "amp"
            else nullcontext()
        )

        if self.is_master:
            print(
                f"\nValidating all {len(self.validation_dataset)} Stage-3 instances "
                f"at {len(timesteps)} fixed timesteps...",
                flush=True,
            )

        for sample_index in range(self.rank, len(self.validation_dataset), self.world_size):
            sample = self.validation_dataset[sample_index]
            data = self.validation_dataset.collate_fn([sample])
            data = recursive_to_device(data, self.device)
            x_0 = data["x_0"]
            shape_z = data["concat_cond"]
            encoded = self.encode_images(data["cond"])

            for timestep_index, timestep in enumerate(timesteps):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(
                    self.validation_seed + sample_index * 1009 + timestep_index
                )
                noise = x_0.replace(torch.randn(
                    x_0.feats.shape,
                    generator=generator,
                    device=x_0.device,
                    dtype=x_0.dtype,
                ))
                t = torch.full(
                    (x_0.shape[0],), timestep, device=x_0.device, dtype=torch.float32
                )
                x_t = self.diffuse(x_0, t, noise=noise)
                with amp_context():
                    pred = denoiser(x_t, t * 1000, encoded, concat_cond=shape_z)
                    target = self.get_v(x_0, noise, t)
                    loss = F.mse_loss(pred.feats, target.feats)
                metric_sum += float(loss.detach().float().item())
                local_count += 1

        packed = torch.tensor(
            [metric_sum, float(local_count)], device=self.device, dtype=torch.float64
        )
        train_packed = torch.tensor(
            [self._checkpoint_train_loss_sum, float(self._checkpoint_train_loss_count)],
            device=self.device,
            dtype=torch.float64,
        )
        if self.world_size > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_packed, op=dist.ReduceOp.SUM)
        count = max(float(packed[1].item()), 1.0)
        row = {
            "step": int(self.step),
            "epoch": float(self.step / self.steps_per_epoch),
            "metric_scope": "validation",
            "validation_protocol": "full_set_fixed_views_fixed_t_fixed_noise_stage3_v1",
            "num_instances": len(self.validation_dataset),
            "num_timesteps": len(timesteps),
            "timesteps": timesteps,
            "num_evaluations": int(packed[1].item()),
            "val_flow_mse": float(packed[0].item() / count),
            "val_loss": float(packed[0].item() / count),
            "parent_stage1_checkpoint": self.parent_stage1_checkpoint,
            "parent_stage2_checkpoint": self.parent_stage2_checkpoint,
            "timestamp": time.time(),
        }
        if train_packed[1].item() > 0:
            row["train_loss"] = float(train_packed[0].item() / train_packed[1].item())
            row["train_num_steps"] = int(train_packed[1].item())
            row["train_loss_protocol"] = (
                "ddp_mean_of_observed_training_steps_since_previous_checkpoint_v1"
            )
        if self.is_master:
            self._write_validation_row(row)
            print(f"Full Stage-3 validation complete: val_loss={row['val_loss']:.6f}", flush=True)
        if self.world_size > 1:
            dist.barrier()
        self._checkpoint_train_loss_sum = 0.0
        self._checkpoint_train_loss_count = 0
        if was_training:
            denoiser.train()
        return row if self.is_master else None
