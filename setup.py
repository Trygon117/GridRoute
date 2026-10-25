from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="gridroute",
    version="0.1.0",
    author="Abraham Furlan",
    description="Dynamic C++ Memory Routing Engine for Recurrent Transformer Architectures",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="gridroute._C",
            sources=[
                "csrc/router.cpp",
                "csrc/bindings.cpp",
            ],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"]
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)