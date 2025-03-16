from setuptools import setup
import os
import sys

if sys.version_info[0] < 3:
    with open('README.rst') as f:
        long_description = f.read()
else:
    with open('README.rst', encoding='utf-8') as f:
        long_description = f.read()


setup(
    name='AxisLabeling',
    version='0.0.2',
    description='Implementation of axis-labeling algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',  # This is important!
    author='StatguyUser',
    url='https://github.com/StatguyUser/AxisLabeling',
    install_requires=['numpy'],
    download_url='https://github.com/StatguyUser/AxisLabeling.git',
    py_modules=["AxisLabeling"],
    package_dir={'':'src'},
)
