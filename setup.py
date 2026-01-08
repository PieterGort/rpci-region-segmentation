"""Setup script for rpci-region-segmentation."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rpci-region-segmentation",
    version="1.0.0",
    author="Pieter Gort",
    author_email="your.email@tue.nl",
    description="CT segmentation of peritoneal regions using SwinUNETR and nnU-Net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PieterGort/rpci-region-segmentation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "nnunet": ["nnunetv2>=2.2"],
        "dev": ["pytest", "black", "ruff"],
    },
    entry_points={
        "console_scripts": [
            "rpci-train=swinunetr.main:main",
            "rpci-predict=swinunetr.predict:main",
        ],
    },
)

