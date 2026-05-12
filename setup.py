#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

import setuptools

version = re.search(
    '^__version__\\s*=\\s*"(.*)"', open("src/periodicity/__init__.py").read(), re.M
).group(1)

with open("README.md", "r") as f:
    long_description = f.read()

install_requires = [
    "bottleneck",
    "celerite2 >= 0.2.0, < 0.3.3",
    "emcee >= 3.0",
    "george",
    "matplotlib",
    "numpy < 2",
    "pandas >= 1.2, < 1.5",
    "PyWavelets >= 0.5",
    "scipy >= 1.1",
    "tqdm",
    "xarray >= 0.20, < 2022",
]

extras_require = {
    "docs": [
        "jupyter >= 1.0",
        "myst-nb >= 0.17",
        "numpydoc",
        "pydata-sphinx-theme",
    ],
    "test": [
        "black == 22.3.0",
        "flake8",
        "isort",
        "pytest",
        "pytest-cov",
        "tox",
    ],
}

setuptools.setup(
    name="periodicity",
    version=version,
    author="Eduardo Nunes",
    author_email="dioph@pm.me",
    license="MIT",
    description="Useful tools for periodicity analysis in time series data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dioph/periodicity",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8, <3.11",
)
