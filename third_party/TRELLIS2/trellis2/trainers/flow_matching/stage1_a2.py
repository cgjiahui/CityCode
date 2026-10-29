"""Image-only A2: balanced BCE, soft IoU and scale-invariant H/sqrt(A)."""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .flow_matching import FootprintConstrainedMultiImageFlowMatchingCFGTrainer
from ...datasets.selected_skyline_sparse_structure_latent import SelectedSkylineMultiImageSparseStructureLatent


class A2SkylineDataset(SelectedSkylineMultiImageSparseStructureLatent):
    def _load_footprint(self, instance, metadata, latent_dir):
        data = super()._load_footprint(instance, metadata, latent_dir)
        scale = json.loads((latent_dir / 'view00_scale.json').read_text())
        # Trusted local preprocessing output: vertices are already normalized.
        path = self.root / 'selected_skyline_pixal3d_nobase/mesh_dumps' / (instance + '.pickle')
        with path.open('rb') as f:
            objects = pickle.load(f)['objects']
        v = np.concatenate([o['vertices'] for o in objects]).astype(np.float64)
        matrix = np.asarray(scale['canonical_transform'])
        v = (v / scale['sphere_radius']) @ matrix[:3, :3].T + matrix[:3, 3]
        v *= scale['box_scale']
        height = float(np.ptp(v[:, 2]))
        grid = data['footprint_ground_grid'].numpy()[..., [2, 1]] / 2
        dx = grid[:, 1:] - grid[:, :-1]
        dy = grid[1:] - grid[:-1]
        valid = data['footprint_ground_valid'].numpy().astype(bool)
        dx = dx[valid[:, 1:] & valid[:, :-1]]
        dy = dy[valid[1:] & valid[:-1]]
        if not len(dx) or not len(dy):
            raise ValueError('Insufficient valid camera area: ' + instance)
        area_per_pixel = float(abs(np.linalg.det(np.stack([dx.mean(0), dy.mean(0)]))))
        # Count all mask pixels: validity clipping must not alter GT area.
        area = float(data['footprint_cond'].sum()) * area_per_pixel
        if not (height > 0 and area > 0 and np.isfinite(height / np.sqrt(area))):
            raise ValueError('Invalid ratio target: ' + instance)
        data['ratio_gt'] = torch.tensor(height / np.sqrt(area), dtype=torch.float32)
        data['footprint_pixel_area'] = torch.tensor(area_per_pixel, dtype=torch.float32)
        return data


def soft_height(logits):
    """Smooth 1%-99% vertical mass extent; soft quantiles retain gradients."""
    mass = logits.float().sigmoid().sum(dim=(1, 2, 3))
    cdf = mass.cumsum(-1) / mass.sum(-1, keepdim=True).clamp_min(1e-6)
    low = torch.sigmoid(1000 * (0.01 - cdf)).sum(-1)
    high = torch.sigmoid(1000 * (0.99 - cdf)).sum(-1)
    return ((high - low) / logits.shape[-1]).clamp_min(1e-4)


class A2Stage1Trainer(FootprintConstrainedMultiImageFlowMatchingCFGTrainer):
    def __init__(self, *args, ratio_weight=0.02, iou_weight=0.05, **kwargs):
        self.ratio_weight = float(ratio_weight)
        self.iou_weight = float(iou_weight)
        super().__init__(*args, **kwargs)
        if kwargs.get('validation_data_dir'):
            self.validation_dataset = A2SkylineDataset(kwargs['validation_data_dir'], **kwargs['validation_dataset_args'])

    def get_inference_cond(self, cond, **kwargs):
        # Supervision must NEVER be forwarded to the sampling backbone.
        return super().get_inference_cond(cond)

    def _footprint_terms(self, predicted_x0, data, *, warmup):
        logits = self.footprint_decoder(predicted_x0)
        p = self._project_vertical_occupancy(logits, data['footprint_ground_grid'], data['footprint_ground_valid']).float()
        g = data['footprint_cond'][:, 0].float()
        valid = data['footprint_ground_valid'].float()
        axes = (1, 2)
        inter = (p * g * valid).sum(axes)
        union = ((p + g - p * g) * valid).sum(axes)
        iou_loss = (1 - (inter + 1e-6) / (union + 1e-6)).mean()
        with torch.autocast('cuda', enabled=False):
            bce = F.binary_cross_entropy(p.clamp(1e-5, 1 - 1e-5), g, reduction='none')
        pos, neg = g * valid, (1 - g) * valid
        balanced = (0.5 * (bce * pos).sum(axes) / pos.sum(axes).clamp_min(1)
                    + 0.5 * (bce * neg).sum(axes) / neg.sum(axes).clamp_min(1)).mean()
        # Native XY columns cover equal canonical area, independent of camera.
        area = logits.float().sigmoid().amax(-1).sum((1, 2, 3)) / logits.shape[-1] ** 2
        ratio = soft_height(logits) / area.clamp_min(1e-6).sqrt()
        ratio_loss = F.smooth_l1_loss(ratio.clamp_min(1e-6).log(), data['ratio_gt'].float().log())
        total = warmup * (self.iou_weight * iou_loss + self.footprint_bce_weight * balanced + self.ratio_weight * ratio_loss)
        return dict(footprint_bce=balanced, footprint_dice=iou_loss,
                    footprint_outside=p.new_zeros(()), footprint_total=total,
                    footprint_opacity=p.detach().mean(),
                    footprint_valid_fraction=valid.mean(),
                    footprint_target_coverage=(pos.sum(axes) / g.sum(axes).clamp_min(1)).mean(),
                    ratio_loss=ratio_loss, ratio_pred=ratio.detach().mean())

    def _validation_objective(self, **kwargs):
        fp = self._footprint_terms((1 - self.sigma_min) * kwargs['noise'] - kwargs['pred'], kwargs['data'], warmup=1)
        mse = F.mse_loss(kwargs['pred'], kwargs['target'])
        result = dict(fp, flow_mse=mse, loss=mse + fp['footprint_total'])
        return result

    @torch.no_grad()
    def run_full_validation(self):
        row = super().run_full_validation()
        # Full independent generation, NOT teacher-forced one-step predictions.
        model = self.models['denoiser']
        previous = model.training
        model.eval()
        records = []
        for index in range(self.rank, len(self.validation_dataset), self.world_size):
            sample = self.validation_dataset[index]
            gen = torch.Generator(device=self.device).manual_seed(self.validation_seed + index * 1009)
            noise = torch.randn(sample['x_0'].shape, generator=gen, device=self.device)[None]
            args = self.get_inference_cond([sample['cond'].to(self.device)])
            with torch.autocast('cuda', dtype=self.mix_precision_dtype, enabled=self.mix_precision_mode == 'amp'):
                z = self.get_sampler().sample(model, noise=noise, **args,
                    steps=self.snapshot_sampler_steps, guidance_strength=self.snapshot_guidance_strength, verbose=False).samples
                logits = self.footprint_decoder(z)
            pred = self._project_vertical_occupancy(logits,
                    sample['footprint_ground_grid'][None].to(self.device),
                    sample['footprint_ground_valid'][None].to(self.device))[0] >= 0.5
            target = sample['footprint_cond'][0].to(self.device) >= 0.5
            # Full GT mask in denominator: out-of-volume GT is not discarded.
            intersection = (pred & target).sum().item()
            union = (pred | target).sum().item()
            records.append({'id': self.validation_dataset.instances[index],
                            'iou_2d_canonical': intersection / union if union else 0.0})
        gathered = [None] * self.world_size
        if self.world_size > 1:
            dist.all_gather_object(gathered, records)
        else:
            gathered = [records]
        if self.is_master:
            all_records = sorted([r for part in gathered for r in part], key=lambda r: r['id'])
            if len(all_records) != len(self.validation_dataset):
                raise RuntimeError('Incomplete validation IoU')
            row['val_sample_iou_2d_canonical'] = float(np.mean([r['iou_2d_canonical'] for r in all_records]))
            row['iou_protocol'] = 'full270_image_only_12step_cfg3_canonical_threshold0.5_v1'
            row['iou_num_instances'] = len(all_records)
            folder = Path(self.output_dir) / 'validation_iou'
            folder.mkdir(exist_ok=True)
            (folder / f'step{self.step:07d}.json').write_text(json.dumps({'summary': row, 'rows': all_records}, indent=2))
            self._write_validation_row(row)
            print('A2 sampled validation 2D IoU:', row['val_sample_iou_2d_canonical'], flush=True)
        if self.world_size > 1:
            dist.barrier()
        model.train(previous)
        return row
