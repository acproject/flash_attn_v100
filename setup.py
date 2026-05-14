from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    
setup(
    name='flash_attn_v100',
    ext_modules=[
        CUDAExtension('flash_attn_v100', [
            'flash_attn.cpp',
            'flash_attn_kernel.cu'
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
