from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="flame-mind",
    version="1.1.0",
    packages=find_packages(exclude=['ext']),
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.22",
        "torchaudio>=2.7",
        "scikit-learn>=1.7",
        "numpy>=2.2",
        "pandas>=2.3",
        "matplotlib>=3.10",
        "opencv-python>=4.11",
        "tqdm>=4.67",
        "pyyaml>=6.0",
        "requests>=2.32",
        "h5py>=3.14",
        "scipy>=1.15",
        "joblib>=1.5",
        "packaging>=25.0",
    ],
    author="Zebulon Zhang",
    author_email="zhangbiao02@spic.com",
    description="Real-time flame status prediction for gas turbines using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhangbiao1231/FlameMind",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
    python_requires=">=3.6",
    include_package_data=True,
)