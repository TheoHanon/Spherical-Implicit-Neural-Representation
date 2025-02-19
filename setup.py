from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spherical_inr",
    version="0.1.2",
    author="Theo Hanon",
    author_email="theo.hanon@student.uclouvain.be",
    description="A package for spherical positional encoding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheoHanon/spherical_inr",  # Update with your repo URL if applicable.
    packages=find_packages(include=["spherical_inr", "spherical_inr.*"]),
    install_requires=[
        "torch>=1.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
