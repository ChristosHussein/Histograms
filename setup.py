# setup.py

from setuptools import setup, find_packages

# Define the package metadata
setup(
    name="StarHistogramAnalysis",  # Name of your project
    version="1.0.0",  # Version number
    author="Christos Hussein",  # Your name (or team name)
    author_email="chris-houssein@windowslive.com",  # Your email address
    description="A project to analyze star data and generate interactive histograms.",  # Short description
    long_description=open("README.md").read(),  # Long description from README.md
    long_description_content_type="text/markdown",  # Specify that the long description is in Markdown
    url="https://github.com/ChristosHussein/StarHistogramAnalysis",  # GitHub repository URL
    packages=find_packages(),  # Automatically discover all packages and sub-packages
    include_package_data=True,  # Include additional files specified in MANIFEST.in
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Update with your chosen license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',  # Minimum Python version required
    install_requires=[
        "pandas>=1.3.0",  # For data manipulation
        "matplotlib>=3.4.0",  # For plotting
        "numpy>=1.20.0",  # For numerical computations
        "openpyxl>=3.0.0",  # For reading Excel files
        "ipywidgets>=7.0.0",  # For interactive widgets
    ],  # List of dependencies
    entry_points={
        'console_scripts': [
            'star_analysis=main_script:interactive_search',  # Optional: Create a CLI command
        ],
    },
)
