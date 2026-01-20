"""Build script for CUDA preprocessing kernels."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name='cuda_preprocessing',
    ext_modules=[
        CUDAExtension(
            'cuda_ops',
            [os.path.join(ROOT, 'kernels', 'preprocessing_kernels.cu')],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
