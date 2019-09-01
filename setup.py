# Copyright (c) 2018-present, Ahmed H. Al-Ghidani.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__author__ = "Ahmed H. Al-Ghidani"
__copyright__ = "Copyright 2018, The mleus Project, https://github.com/AhmedHani/mleus"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Ahmed H. Al-Ghidani"
__email__ = "ahmed.hani.ibrahim@gmail.com"

from setuptools import setup, find_packages


with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='mleus',
    version='bce',
    description='A personal library for machine learning projects',
    long_description_content_type="text/markdown",
    author='Ahmed H. Al-Ghidani',
    author_email='ahmed.hani.ibrahim@gmail.com',
    url='https://github.com/AhmedHani/mleus',
    license='BSD 3-Clause License',
    install_requires=required,
    packages=find_packages(exclude="tests")
)
