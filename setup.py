from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(
    name='auto_smoothquant',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='auto_smoothquant._CUDA',
            sources=[
                'layers/int8gemm/cuda_utils.cc',
                'layers/int8gemm/cublasINT8MMWrapper.cc',
                'layers/int8gemm/cublasAlgoMap.cc',
                'layers/int8gemm/bindings.cpp',
            ],
            extra_link_args=['-lcublas_static', '-lcublasLt_static',
                             '-lculibos', '-lcudart', '-lcudart_static',
                             '-lrt', '-lpthread', '-ldl', '-L/usr/lib/x86_64-linux-gnu/'],
            extra_compile_args={'cxx': ['-std=c++14', '-O3'],
                                'nvcc': ['-O3', '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']},
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests']),
)