"""Setup configuration for Black Diamond."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="black-diamond",
    version="0.1.0",
    author="Black Quantum Diamond Ltd",
    author_email="contact@blackquantumdiamond.com",
    description=(
        "Numerical solvers for classical mathematics, physics, and chemistry "
        "with controlled error and polynomial cost."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/blackquantumdiamond/black-diamond",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.8",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
