# from setuptools import setup, find_packages
# from torch.utils import cpp_extension

# setup(
#     name='ftgemm',
#     ext_modules=[
#         cpp_extension.CUDAExtension(
#             name='ftgemm',
#             sources=[
#                 # 'linear.cu',
#                 'bindings.cpp'
#             ],
#             # include_dirs=['include'],
#             include_dirs=['./'],
#             extra_link_args=['-lcublas_static', '-lcublasLt_static',
#                              '-lculibos', '-lcudart', '-lcudart_static',
#                              '-lrt', '-lpthread', '-ldl', '-L/usr/lib/x86_64-linux-gnu/'],
#             extra_compile_args={'cxx': ['-std=c++14', '-O3'],
#                                 'nvcc': ['-O3', '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']},
#         ),
#     ],
#     cmdclass={
#         'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
#     },
#     packages=find_packages(
#         exclude=['notebook', 'scripts', 'tests']),
# )

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import subprocess

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="", *args, **kwargs):
        Extension.__init__(self, name, sources=[], *args, **kwargs)
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # cmake_args = [
        #     # fixed for lightseq.inference
        #     "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + os.path.join(extdir, "lightseq"),
        #     "-DPYTHON_EXECUTABLE=" + sys.executable,
        # ]

        # cfg = "Release"
        # build_args = ["--config", cfg]

        # if platform.system() == "Windows":
        #     cmake_args += [
        #         "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
        #     ]
        #     if sys.maxsize > 2**32:
        #         cmake_args += ["-A", "x64"]
        #     build_args += ["--", "/m"]
        # else:
        #     cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        #     cmake_args += ["-DFP16_MODE=OFF"] if ENABLE_FP32 else ["-DFP16_MODE=ON"]
        #     cmake_args += ["-DDEBUG_MODE=ON"] if ENABLE_DEBUG else ["-DDEBUG_MODE=OFF"]
        #     cmake_args += (
        #         ["-DUSE_NEW_ARCH=ON"] if ENABLE_NEW_ARCH else ["-DUSE_NEW_ARCH=OFF"]
        #     )
        #     cmake_args += ["-DDEVICE_ARCH={}".format(DEVICE_ARCH)]
        #     cmake_args += ["-DDYNAMIC_API=OFF"]
        #     build_args += ["--target", "lightseq"]
        #     build_args += ["--", "-j{}".format(multiprocessing.cpu_count())]

        env = os.environ.copy()
        # env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
        #     env.get("CXXFLAGS", ""), self.distribution.get_version()
        # )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            # ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
            ["cmake", ext.sourcedir], cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            # ["cmake", "--build", "."] + build_args, cwd=self.build_temp
            ["cmake", "--build", "."], cwd=self.build_temp
        )


# 使用setup函数来定义模块的信息和编译命令
setup(
    name='ftgemm',
    version='0.1',
    description='ftgemm module',
    ext_modules=[CMakeExtension("ftgemm")],
    cmdclass={'build_ext': CMakeBuild},
    packages=setuptools.find_packages(
        exclude=['notebook', 'scripts', 'tests'])
)
