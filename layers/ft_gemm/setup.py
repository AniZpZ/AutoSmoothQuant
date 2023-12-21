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
