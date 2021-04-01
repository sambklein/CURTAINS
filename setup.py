from setuptools import find_packages, setup

with open("README.md", "r") as fh:

    long_description = fh.read()

setup(
    name="implicitBIBae",
    version='0.01',
    description="Normalizing flows in PyTorch.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/sambklein/implicitBIBae",
    author="Sam Klein",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    install_requires=[
        "matplotlib",
        "numpy",
        "tensorboard",
        "torch",
        "tqdm",
    ],
    dependency_links=[],
)
