from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spherical_inr",
<<<<<<< HEAD
    version="0.3.15",
=======
    version="0.3.9",
>>>>>>> 4c29566 (Bump version to 0.3.9 in __init__.py, pyproject.toml, and setup.py)
    author="Theo Hanon",
    author_email="theo.hanon@student.uclouvain.be",
    description="A package for spherical positional encoding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheoHanon/Spherical-Implicit-Neural-Representation",
    packages=find_packages(include=["spherical_inr", "spherical_inr.*"]),
    install_requires=[
        "torch>=1.7.0",
    ],
    project_urls={
        "Documentation": "https://spherical-implicit-neural-representation.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/TheoHanon/Spherical-Implicit-Neural-Representation",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
