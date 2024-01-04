import setuptools
from setuptools import find_packages

setuptools.setup(
    packages=find_packages(),
    include_package_data=True,  # for the URDFs in cloth_tools/resources
)
