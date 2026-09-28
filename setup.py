"""Setup configuration for the tokentrie package.

This module handles compilation and packaging of the TokenTrieModel C-extension.
It integrates platform-specific C compiler optimizations (MSVC on Windows, 
GCC/Clang on Unix/macOS) with Cython's compilation pipeline to maximize 
inference throughput and online learning speed.
"""

import os
import sys
from typing import Dict, List, Tuple, Union

from setuptools import Extension, find_packages, setup

# Conditional import of Cython to support environments building from pre-generated C sources
try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


def read_long_description(file_path: str = "README.md") -> str:
    """Reads the long description from a file for PyPI package metadata.

    Args:
        file_path (str): Relative path to the markdown documentation file.

    Returns:
        str: Contents of the documentation file, or a brief fallback description.
    """
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file_handle:
            return file_handle.read()
    return (
        "High-performance, adaptive sequence prediction engine based on "
        "Context Mixing and Variable Order Markov Models (VOMM)."
    )


def get_compiler_optimization_flags() -> Tuple[List[str], List[str]]:
    """Determines platform-specific compilation and linking optimization flags.

    Enables maximum hardware-level mathematical throughput (/O2, /fp:fast for MSVC;
    -O3, -ffast-math for GCC and Clang) to accelerate libc.math evaluations in core.pyx.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing:
            - extra_compile_args: Flags passed to the C compiler.
            - extra_link_args: Flags passed to the linker.
    """
    compile_args: List[str] = []
    link_args: List[str] = []

    if sys.platform == "win32":
        # Flags for Microsoft Visual C++ (MSVC)
        compile_args.extend(["/O2", "/fp:fast"])
    else:
        # Flags for GCC / Clang on Linux and macOS
        compile_args.extend(["-O3", "-ffast-math"])

    return compile_args, link_args


def build_extension_modules() -> Union[List[Extension], List[object]]:
    """Configures the C/Cython extension modules with optimal compiler directives.

    Directives match the constraints established in tokentrie/core.pyx (disabling
    bounds checking, wraparound checks, and enabling native C division).

    Returns:
        Union[List[Extension], List[object]]: Cythonized extension objects or raw Extensions.
    """
    source_ext = ".pyx" if HAS_CYTHON else ".c"
    source_path = os.path.join("tokentrie", f"core{source_ext}")

    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Source file '{source_path}' not found. Ensure Cython is installed "
            "to compile 'core.pyx', or pre-generate 'core.c'."
        )

    compile_args, link_args = get_compiler_optimization_flags()

    extensions = [
        Extension(
            name="tokentrie.core",
            sources=[source_path],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ]

    compiler_directives: Dict[str, Union[str, bool]] = {
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "nonecheck": False,
        "cdivision": True,
    }

    if HAS_CYTHON:
        return cythonize(extensions, compiler_directives=compiler_directives)
    return extensions


setup(
    name="tokentrie",
    version="1.0.0",
    author="Ivan Kholodilo",
    author_email="iholodilo2008@gmail.com",
    description=(
        "High-performance, adaptive sequence prediction engine based on "
        "Context Mixing and Variable Order Markov Models (VOMM)."
    ),
    long_description=read_long_description("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/Icold21/token-trie",
    license="MIT",
    packages=find_packages(exclude=["tests*", "examples*"]),
    ext_modules=build_extension_modules(),
    python_requires=">=3.8",
    install_requires=[
        "tqdm>=4.60.0",
    ],
    extras_require={
        "full": [
            "numpy>=1.20.0",
            "matplotlib>=3.5.0",
            "tiktoken>=0.5.0",
            "pypdf>=3.0.0",
            "jupyter>=1.0.0",
            "pytest>=7.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)