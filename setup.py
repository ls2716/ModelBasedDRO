from setuptools import setup, find_packages

setup(
    name="mbdro",                     # Name of your package
    version="0.1.0",                       # Initial version
    author="Lukasz S",
    author_email="lukasz@ls314.com",
    description="code for model-based DRO numerical experiments",
    packages=find_packages(),             # Automatically finds all packages
    install_requires=[],                  # List of dependencies
    python_requires=">=3.7",              # Minimum Python version
)