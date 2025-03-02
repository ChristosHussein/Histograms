from setuptools import setup, find_packages

setup(
    name="star-histogram-analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.2.0",
        "openpyxl>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "star-histogram=main:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Interactive tool for visualizing astrophysical data with Chi-squared weighted histograms",
    keywords="astronomy, visualization, histogram, chi-squared",
    url="https://github.com/yourusername/star-histogram-analyzer",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.7",
)
