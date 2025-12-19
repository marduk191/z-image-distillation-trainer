"""
Z-Image Distillation Trainer
Setup script for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="z-image-distillation-trainer",
    version="1.0.0",
    author="marduk191",
    description="Single-file Python implementation of Decoupled-DMD distillation for Z-Image models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marduk191/z-image-distillation-trainer",
    py_modules=[
        "z_image_distillation_trainer",
        "z_image_distillation_gui",
        "launch_gui",
        "test_distilled_model",
        "batch_inference",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "safetensors>=0.4.0",
        "accelerate>=0.25.0",
        "transformers>=4.36.0",
        "tqdm>=4.66.0",
        "datasets>=2.14.0",
        "peft>=0.7.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "lpips": ["lpips>=0.1.4"],
        "flash-attn": ["flash-attn>=2.5.0"],
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "pytest>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "z-image-train=z_image_distillation_trainer:main",
            "z-image-gui=launch_gui:main",
            "z-image-test=test_distilled_model:main",
            "z-image-batch=batch_inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["examples/*.json", "examples/*.yaml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/marduk191/z-image-distillation-trainer/issues",
        "Source": "https://github.com/marduk191/z-image-distillation-trainer",
        "Documentation": "https://github.com/marduk191/z-image-distillation-trainer/blob/main/docs/",
        "Z-Image Official": "https://github.com/Tongyi-MAI/Z-Image",
    },
)
