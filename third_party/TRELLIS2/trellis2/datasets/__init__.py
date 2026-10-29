import importlib

__attributes = {
    'FlexiDualGridDataset': 'flexi_dual_grid',
    'SparseVoxelPbrDataset':'sparse_voxel_pbr',

    'SparseStructureLatent': 'sparse_structure_latent',
    'TextConditionedSparseStructureLatent': 'sparse_structure_latent',
    'ImageConditionedSparseStructureLatent': 'sparse_structure_latent',
    'SelectedSkylineMultiImageSparseStructureLatent': 'selected_skyline_sparse_structure_latent',
    'SelectedSkylineMultiImageSLatShape': 'selected_skyline_structured_latent',
    'SelectedSkylineMultiImageSLatPbr': 'selected_skyline_pbr_latent',
    
    'SLat': 'structured_latent',
    'ImageConditionedSLat': 'structured_latent',
    'SLatShape': 'structured_latent_shape',
    'ImageConditionedSLatShape': 'structured_latent_shape',
    'SLatPbr': 'structured_latent_svpbr',
    'ImageConditionedSLatPbr': 'structured_latent_svpbr',
}

__submodules = []

def _a2_dataset():
    from ..trainers.flow_matching.stage1_a2 import A2SkylineDataset
    return A2SkylineDataset

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name == 'A2SkylineDataset':
        return _a2_dataset()
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':    
    from .flexi_dual_grid import FlexiDualGridDataset
    from .sparse_voxel_pbr import SparseVoxelPbrDataset
    
    from .sparse_structure_latent import SparseStructureLatent, ImageConditionedSparseStructureLatent
    from .selected_skyline_sparse_structure_latent import SelectedSkylineMultiImageSparseStructureLatent
    from .selected_skyline_structured_latent import SelectedSkylineMultiImageSLatShape
    from .selected_skyline_pbr_latent import SelectedSkylineMultiImageSLatPbr
    from .structured_latent import SLat, ImageConditionedSLat
    from .structured_latent_shape import SLatShape, ImageConditionedSLatShape
    from .structured_latent_svpbr import SLatPbr, ImageConditionedSLatPbr
