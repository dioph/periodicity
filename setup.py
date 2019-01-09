import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="periodicity",
    version="0.1.0b2",
    author="Eduardo Nunes",
    author_email="diofanto.nunes@gmail.com",
    license="MIT",
    description="Useful tools for analysis of periodicities in time series data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dioph/periodicity",
    packages=setuptools.find_packages(),
    install_requires=['numpy>=1.11', 'astropy>=1.3', 'scipy>=0.19.0',
                      'emcee', 'tqdm', 'autograd'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ),
)
