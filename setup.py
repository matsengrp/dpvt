from setuptools import setup, find_packages

setup(
    name="dpvt",
    version="0.1.0",
    url="https://github.com/matsengrp/dpvt.git",
    author="Matsen Group",
    author_email="ematsen@gmail.com",
    description="Deep neural networks for Phylogenetics Via Traversals",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "click",
        "ete3 >= 3.0.0",
        "lightning >= 2.2.0",
        "pytest >= 7.3",
        "torch >= 2.0.0",
    ],
    python_requires="==3.9.*",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
)
