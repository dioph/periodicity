import re

import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

version = re.search(
    '^__version__\\s*=\\s*"(.*)"',
    open('periodicity/__init__.py').read(),
    re.M
).group(1)

setuptools.setup(
    name="periodicity",
    version=version,
    author="Eduardo Nunes",
    author_email="dioph@pm.me",
    license="MIT",
    description="Useful tools for analysis of periodicities in time series data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dioph/periodicity",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'astropy', 'scipy>=1.2.0',
                      'emcee', 'tqdm', 'autograd'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ),
)
