import re

import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

version = re.search(
    '^__version__\\s*=\\s*"(.*)"',
    open("periodicity/__init__.py").read(),
    re.M
).group(1)

install_requires = [
    "astropy >= 3.2",
    "autograd",
    "celerite",
    "emcee >= 3.0",
    "george",
    "numpy",
    "PyWavelets >= 0.5",
    "scipy >= 1.1",
    "tqdm"
]

extras_require = {
    "test": ["pytest"],
    "docs": ["jupyter", "numpydoc", "pandoc", "sphinx_rtd_theme"]
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
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ),
)
