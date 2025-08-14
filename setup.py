# Copyright (c) 2023, Albert Gu, Tri Dao.
import os, re, ast
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PACKAGE_NAME = "mamba_ssm"
this_dir = os.path.dirname(os.path.abspath(__file__))

# Force strictly local build (ignore any prebuilt wheel logic)
os.environ["MAMBA_FORCE_BUILD"] = "TRUE"

# Hardcode GPU arch for GTX 1080 Ti (Pascal SM 6.1)
cc_flag = [
    "-gencode", "arch=compute_61,code=sm_61",
    "-gencode", "arch=compute_61,code=compute_61"  # PTX fallback
]

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3", "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=-v",
        "-lineinfo",
    ] + cc_flag
}

def get_package_version():
    with open(Path(this_dir) / PACKAGE_NAME / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("MAMBA_LOCAL_VERSION")
    return f"{public_version}+{local_version}" if local_version else str(public_version)

ext_modules = [
    CUDAExtension(
        name="selective_scan_cuda",
        sources=[
            "csrc/selective_scan/selective_scan.cpp",
            "csrc/selective_scan/selective_scan_fwd_fp32.cu",
            "csrc/selective_scan/selective_scan_fwd_fp16.cu",
            "csrc/selective_scan/selective_scan_fwd_bf16.cu",
            "csrc/selective_scan/selective_scan_bwd_fp32_real.cu",
            "csrc/selective_scan/selective_scan_bwd_fp32_complex.cu",
            "csrc/selective_scan/selective_scan_bwd_fp16_real.cu",
            "csrc/selective_scan/selective_scan_bwd_fp16_complex.cu",
            "csrc/selective_scan/selective_scan_bwd_bf16_real.cu",
            "csrc/selective_scan/selective_scan_bwd_bf16_complex.cu",
        ],
        extra_compile_args=extra_compile_args,
        include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
    )
]

setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(exclude=("build", "csrc", "include", "tests", "dist", "docs", "benchmarks", "mamba_ssm.egg-info")),
    author="Tri Dao, Albert Gu",
    author_email="tri@tridao.me, agu@cs.cmu.edu",
    description="Mamba state-space model (local build for GTX 1080 Ti SM 6.1)",
    long_description="Local build for GTX 1080 Ti (Pascal SM 6.1) with PyTorch CUDA 12.6 toolchain assumptions.",
    long_description_content_type="text/markdown",
    url="https://github.com/state-spaces/mamba",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "packaging",
        "ninja",
        "einops",
        "triton",
        "transformers",
    ],
)
