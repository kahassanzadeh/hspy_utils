from setuptools import setup, find_packages

setup(
    name="hspy_utils",
    version="0.1.0",
    py_modules=["CondAns", "HspyPrep"],
    description="A utility package including CondAns and HspyPrep modules.",
    packages=find_packages(),
    author="Mohammadreza Hassanzadeh",
    author_email="mohammadreza.hassanzadeh@epfl.ch",
    url="https://example.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "lmfit",
        "lumispy",
        "hyperspy"
    ],
)
