# plenpy - A plenoptic processing library for Python.

[![build status](https://gitlab.com/iiit-public/plenpy/badges/master/pipeline.svg)](https://gitlab.com/iiit-public/plenpy/commits/master)
[![coverage report](https://gitlab.com/iiit-public/plenpy/badges/master/coverage.svg)](https://gitlab.com/iiit-public/plenpy/commits/master)
[![PyPI](https://img.shields.io/pypi/v/plenpy.svg)](https://pypi.org/project/plenpy/#description)
[![PyPI](https://img.shields.io/pypi/pyversions/plenpy.svg)](https://pypi.org/project/plenpy/#description)
[![PyPI](https://img.shields.io/pypi/status/plenpy.svg)](https://pypi.org/project/plenpy/#description)


This is a Python package to calibrate, process and analyse
(hyperspectral) light field images as well as (hyper)spectral images
from either real cameras (e.g. Lytro) or synthetic/rendered images.


>**Note:** The package is still undergoing API altering changes with each minor release.  
     

## License and Usage

This software is licensed under the GNU GPLv3 license (see below).

If you use this software in your scientific research, please cite [our paper](https://doi.org/10.1109/TCI.2020.2964257):


    @Article{Schambach2020,
      author  = {Schambach, Maximilian and Puente León, Fernando},
      title   = {Microlens array grid estimation, light field decoding, and calibration},
      journal = {IEEE Transactions on Computational Imaging},
      volume  = {6},
      pages   = {591--603},
      year    = {2020},
      doi     = {10.1109/TCI.2020.2964257},
    }


## Quick Start
Have a look at our [User Documentation](https://iiit-public.gitlab.io/plenpy/user-doc.html) 
for notes on usage and some examples to get you started.

For a quick tryout of plenpy, you can use our latest 
[Docker Image](https://hub.docker.com/r/plenpy/plenpy).



## Installation

You can install ``plenpy`` directly from PyPi via ``pip``:


    $ pip install plenpy


That's it!


### Dependencies
Plenpy requires `python >= 3.6` as it relies on Python syntax that has
been introduced in Python 3.6 such as f-strings or type hinting.
Plenpy is currently tested on Python 3.6, 3.7, and 3.8.

The package dependencies are resolved automatically upon installation using `pip`.
For development and testing dependencies, see the ``requirements.txt`` file.
The package dependencies are stated in ``setup.py``.
 

### Manual Installation on Unix / Linux / macOS

If you want to install from source, the installation using `make` is straightforward and installs
`plenpy` and its runtime dependencies automatically.
If ``make`` is not available, or if you are running Windows, see below.


**Caution**: A system wide installation using ``sudo`` is easy and possible but
discouraged. Installing in a environment is recommended.

To install `plenpy`, first clone the project's git repository to
a location of your desire and change directory to the project:

    $ cd <path-to-plenpy>/
    $ git clone git@gitlab.com:iiit-public/plenpy.git
    $ cd plenpy

Then, install the library via:

    $ make

Or, to have an editable install of plenpy, using

    $ make editable

If no errors occur, you can check if the installation was successful
by running the unit tests:

    $ make test

That's it! The package should now be available.


### Manual Installation on Windows

If `make` is not available on your system, the installation via `pip`
is also straightforward. Instead of invoking `make`,
install by calling (e.g. from the Anaconda prompt)

    $ pip install -r requirements.txt .

Please note the `.` at the end, referring to the current folder
`<path-to-plenpy>/plenpy`. 


### Testing

You can manually run the tests using `pytest`:

    $ pytest <path-to-plenpy>/test/



### Uninstallation
Uninstall ``plenpy`` using

    $ pip uninstall plenpy



## Documentation

The documentation can be found [here](https://iiit-public.gitlab.io/plenpy).

You can also build the documentation yourself:

### Dependencies and Building

The documentation is build using [Sphinx](http://www.sphinx-doc.org/en/stable/).
To install all necessary dependencies for the documentation, run

    $ cd <path-to-plenpyr>
    $ make
    $ pip install -r docs/requirements.txt
    $ cd docs
    $ sphinx-apidoc -f -o ./ ../plenpy/
    $ make html
    
This will create the full `plenpy` documentation in the 
`docs/_build/html` folder.



## Contribute
If you are interested in contributing to ``plenpy``, feel free to create an issue or
fork the project and submit a merge request. As this project is still undergoing
restructuring and extension, help is always welcome!


### For Programmers

Please stick to the 
[PEP 8 Python coding styleguide](https://www.python.org/dev/peps/pep-0008/).

The docstring coding style of the reStructuredText follows the 
[googledoc style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).



## License

Copyright (C) 2018-2020  The Plenpy Authors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
