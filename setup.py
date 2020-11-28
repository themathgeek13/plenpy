# -*- coding: utf-8 -*-1.3
# Copyright (C) 2018-2020  The Plenpy Authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import setuptools
from setuptools import setup
import subprocess

# Get Version from Pipeline, ignore leading 'v'
if os.environ.get('CI_COMMIT_TAG'):
    version = os.environ['CI_COMMIT_TAG'][1:]
elif os.environ.get('CI_JOB_ID'):
    version = os.environ['CI_JOB_ID']

# For local builds:
else:
    try:
        # Get latest git tag
        result = subprocess.run("git describe --tags", shell=True, stdout=subprocess.PIPE)
        version = result.stdout.decode('utf-8')[1:-1] + "-local"
    except:
        version = "local"

# Write version file for module to import
with open("plenpy/_version.py", "w") as f:
    f.write(f"__version__='{version}'")

# Load README as full description
short_description = "A plenoptic processing library for Python."
try:
    with open("README.md", "r") as f:
        long_description = f.read()
except:
    long_description = short_description



setup(
    python_requires='>=3.6',
    name='plenpy',
    version=version,
    description=short_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://iiit-public.gitlab.io/plenpy/intro.html',
    project_urls={
        'Documentation': 'https://iiit-public.gitlab.io/plenpy/',
        'Source': 'https://gitlab.com/iiit-public/plenpy',
        'Research Group': 'https://www.iiit.kit.edu/english/3127.php'
    },
    author='Maximilian Schambach',
    author_email='schambach@kit.edu',
    install_requires=[
        'h5py >= 2.0',
        'numPy >= 1.18',
        'scipy >= 1.1',
        'scikit-image >= 0.14',
        'matplotlib',
        'imageio >= 2.3.0',
        'colour-science >= 0.3.15',
        'colour-demosaicing',
        'joblib',
        'jsmin',
    ],
    license='GNU General Public License v3 (GPLv3)',
    packages=setuptools.find_packages(),
    package_data={'plenpy': ['data/*.npy']},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
    zip_safe=True)
