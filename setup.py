from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attn_v100',
    version='1.0.0',
    description='High-performance LLM inference library with V100-optimized Flash Attention',
    packages=find_packages(include=['flash_attn_llm', 'flash_attn_llm.*']),
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12.0',
        'safetensors',
        'transformers',
        'fastapi',
        'uvicorn',
        'pydantic',
    ],
    ext_modules=[
        CUDAExtension('flash_attn_v100', [
            'flash_attn.cpp',
            'flash_attn_kernel.cu',
            'continuous_batching_kernel.cu'
        ],extra_compile_args={'cxx': ['-O3', '-std=c++17'],
                              'nvcc': [
                                '-O3', 
                                '--use_fast_math', 
                                '-lineinfo', 
                                '-gencode=arch=compute_70,code=sm_70',
                                '-std=c++17']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
