"""降维算法模块"""
from .pca import PCAReduction
from .tsne import TSNEReduction
from .umap_reduction import UMAPReduction
from .lda import LDAReduction

__all__ = [
    'PCAReduction',
    'TSNEReduction',
    'UMAPReduction',
    'LDAReduction',
]
