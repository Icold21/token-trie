"""Setup configuration for the tokentrie package.

Builds the native Cython C-extension module with platform-specific
compiler optimizations (MSVC on Windows, GCC/Clang on POSIX/macOS).
"""

import os
import sys
from typing import Dict, List, Tuple, Union

from setuptools import Extension, find_packages, setup

try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


def get_compiler_optimization_flags() -> Tuple[List[str], List[str]]:
    """Determines platform-specific compilation and linking optimization flags.

    Returns:
        Tuple[List[str], List[str]]: (extra_compile_args, extra_link_args).
    """
    compile_args: List[str] = []
    link_args: List[str] = []

    if sys.platform == "win32":
        compile_args.extend(["/O2", "/fp:fast"])
    elif sys.platform == "darwin":
        compile_args.extend(["-O3", "-ffast-math"])
    else:
        compile_args.extend(["-O3", "-ffast-math", "-fPIC"])

    return compile_args, link_args


def build_extension_modules() -> Union[List[Extension], List[object]]:
    """Configures C/Cython extension modules with optimal compiler directives."""
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
        "embedsignature": True,
    }

    if HAS_CYTHON:
        return cythonize(
            extensions, 
            compiler_directives=compiler_directives, 
            force=True
        )
    return extensions


setup(
    name="tokentrie",
    version="1.1.0",
    description="High-Throughput Variable Order Markov Models via Reverse Suffix Tries",
    packages=find_packages(),
    ext_modules=build_extension_modules(),
    python_requires=">=3.8",
)