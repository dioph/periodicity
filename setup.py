import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="periodicity",
    version="0.0.1",
    author="Eduardo Nunes",
    author_email="diofanto.nunes@gmail.com",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dioph/periodicity",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ),
)
