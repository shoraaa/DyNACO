from setuptools import setup, Extension
import pybind11
import os
import sys

# Platform specific flags
if sys.platform == "darwin":
    # MacOS usually uses clang which might need different openmp flags
    # But user is on Linux, so we focus on GCC/Clang with -fopenmp
    extra_compile_args = ['-O3', '-std=c++17', '-Xpreprocessor', '-fopenmp']
    extra_link_args = ['-lomp']
else:
    extra_compile_args = ['-O3', '-std=c++17', '-fopenmp']
    extra_link_args = ['-fopenmp']

ext_modules = [
    Extension(
        "faco_opt",
        ["binding.cpp", "mfaco_train.cpp"],
        include_dirs=[
            pybind11.get_include(),
            ".",
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++',
    ),
]

setup(
    name="faco_opt",
    version="1.0.0",
    ext_modules=ext_modules,
)
