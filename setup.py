from setuptools import find_packages, setup

with open("README.md", "r") as fh:

    long_description = fh.read()

setup(
    name="curtains",
    version='0.01',
    description="Continuously Updated Retrained Transport model for Anomaly detection using Invertible Networks.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    # url="https://github.com/sambklein/implicitBIBae",
    # author="Sam Klein, Johnny Raine",
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
