from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np
import sys

def omp_flags():
    if sys.platform.startswith("win"):
        return (["/openmp"], [])
    elif sys.platform == "darwin":
        return (["-Xpreprocessor", "-fopenmp"], ["-lomp"])
    else:
        return (["-fopenmp"], ["-fopenmp"])

cxx_omp, link_omp = omp_flags()

ext_modules = [
    Pybind11Extension(
        "faco_cvrp",
        sources=[
            "binding.cpp",
            "mfaco_train.cpp",
            # "kd_tree.cpp",   # include only if you actually have it
        ],
        include_dirs=["include", np.get_include()],
        cxx_std=17,
        extra_compile_args=["-O3", "-ffast-math", *cxx_omp],
        extra_link_args=[*link_omp],
    ),
]

setup(
    name="faco-cvrp",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
