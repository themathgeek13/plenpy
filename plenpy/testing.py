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


"""Testing module of the plenpy package.

This module should only be imported in tests of the plenpy packages.
The module provides methods to read and store large binary data that is needed
for testing and which is not stored in the plenpy repository but the extra
plenpy-binaries repository. The test files are downloaded and saved
in a plenpy app data folder, e.g. on Unix-like machines ``~/.plenpy``.

This module is mostly adapted from the ``imageio`` package by Almar Klein.

"""

import os
import shutil
import socket
import sys
import time
import uuid
from math import log
from pathlib import Path
from typing import Optional, Union, Tuple
from urllib.request import urlopen, Request
from urllib.response import addinfourl

import matplotlib as mpl
import pytest

import plenpy.logg

mpl.use('Agg')

# Logging settings
plenpy.logg.set_level("warning")
logger = plenpy.logg.get_logger()


class InternetNotAllowedError(IOError):
    """ Tests that need resources can just use get_remote_file(), but
    should catch this error and silently ignore it.

    """
    pass


class NeedDownloadError(IOError):
    """ Is raised when a remote file is requested that is not locally
    available, but which needs to be explicitly downloaded by the user.

    """


def has_internet(host: str = "8.8.8.8",
                 port: int = 53,
                 timeout: int = 3) -> bool:
    """Check if internet connection is available.

    Args:
        host: DNS server. Default: 8.8.8.8 (google-public-dns-a.google.com).
        port: Open port. Default: 53 (tcp)
        timeout: Timeout in seconds.

    Returns:
        ``True`` if Internet connection is available.

    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception as ex:
        logger.warning(ex)
        return False


def needs_internet():
    """Specify if a tests needs an Internet connection.

    If a test needs an Internet connection to download binary test files
    but no connection is available during testing, the according test will
    be skipped.

    """
    if os.getenv('PLENPY_NO_INTERNET', '').lower() in ('1', 'true', 'yes'):
        pytest.skip('No internet connection to download test data.')


def get_tmp_folder() -> Tuple[Path, uuid.UUID]:
    """Get a unique temporary folder in the appdata directory"""

    # Create a unique id
    id = uuid.uuid4()

    path = appdata_dir(plenpy.__name__) / str(id)

    try:
        path.mkdir()
    except FileExistsError:
        raise FileExistsError("Cannot create temporary test directory.")

    return path, id


def remove_tmp_folder(id: uuid.UUID):
    """Remove a temporarily created test directory."""

    path = appdata_dir(plenpy.__name__) / str(id)
    shutil.rmtree(path)

    return


# From pyzolib/paths.py (https://bitbucket.org/pyzo/pyzolib/src/tip/paths.py)
def appdata_dir(appname: Optional[str] = None,
                roaming: bool = False) -> Path:
    """Get the path to the application directory, where applications are allowed
    to write user specific files (e.g. configurations).

    Args:
        appname: If specified, a subdir is appended (and created if necessary).
        roaming: If ``True``, will prefer a roaming directory (Windows 7).

    """

    # Define default user directory
    user_dir = os.getenv('PLENPY_USERDIR', None)
    if user_dir is None:
        user_dir = os.path.expanduser('~')
        if not os.path.isdir(user_dir):  # pragma: no cover
            user_dir = '/var/tmp'  # issue #54

    # Get system app data dir
    path = None
    if sys.platform.startswith('win'):
        path1, path2 = os.getenv('LOCALAPPDATA'), os.getenv('APPDATA')
        path = (path2 or path1) if roaming else (path1 or path2)
    elif sys.platform.startswith('darwin'):
        path = os.path.join(user_dir, 'Library', 'Application Support')
    # On Linux and as fallback
    if not (path and os.path.isdir(path)):
        path = user_dir

    # Maybe we should store things local to the executable (in case of a
    # portable distro or a frozen application that wants to be portable)
    prefix = sys.prefix
    if getattr(sys, 'frozen', None):
        prefix = os.path.abspath(os.path.dirname(sys.executable))
    for reldir in ('settings', '../settings'):
        localpath = os.path.abspath(os.path.join(prefix, reldir))
        if os.path.isdir(localpath):  # pragma: no cover
            try:
                open(os.path.join(localpath, 'test.write'), 'wb').close()
                os.remove(os.path.join(localpath, 'test.write'))
            except IOError:
                pass  # We cannot write in this directory
            else:
                path = localpath
                break

    # Get path specific for this app
    if appname:
        if path == user_dir:
            appname = '.' + appname.lstrip('.')  # Make it a hidden directory
        path = os.path.join(path, appname)
        if not os.path.isdir(path):  # pragma: no cover
            os.mkdir(path)

    # Done
    return Path(path)


def get_remote_file(file_name: str,
                    directory: Optional[str] = None,
                    force_download: Union[bool, str] = False,
                    auto: bool = True) -> Path:
    """ Get a the filename for the local version of a file from the web.

    Args:
        file_name: The relative filename on the remote data repository to
            download. These correspond to paths on
            ``https://gitlab.com/iiit-public/plenpy-binaries/raw/master/``.

        directory: The directory where the file will be cached if a download
            was required to obtain the file. By default, the appdata directory
            is used. This is also the first directory that is checked for
            a local version of the file. If the directory does not exist,
            it will be created.

        force_download: If ``True``, the file will be downloaded even if a
            local copy exists (and this copy will be overwritten).
            Can also be a YYYY-MM-DD date to ensure a file is up-to-date
            (modified date of a file on disk, if present, is checked).

        auto: Whether to auto-download the file if its not present locally.
            Default is ``True``.
            If ``False`` and a download is needed, raises NeedDownloadError.

    Returns:
        The path to the file on the local system.
    """

    _url_root = 'http://gitlab.com/iiit-public/plenpy-binaries/-/raw/master/'
    url = _url_root + file_name
    nfname = os.path.normcase(file_name)  # convert to native

    # Get dirs to look for the resource
    given_directory = directory
    directory = given_directory or appdata_dir(plenpy.__name__)

    # Try to find the resource locally
    # for dir in dirs:
    filename = os.path.join(directory, nfname)
    if os.path.isfile(filename):

        if not force_download:  # we're done
            if given_directory and given_directory != directory:
                filename2 = os.path.join(given_directory, nfname)
                # Make sure the output directory exists
                if not os.path.isdir(os.path.dirname(filename2)):
                    os.makedirs(os.path.abspath(os.path.dirname(filename2)))
                shutil.copy(filename, filename2)
                return Path(filename2)
            return Path(filename)

        if isinstance(force_download, str):
            ntime = time.strptime(force_download, '%Y-%m-%d')
            ftime = time.gmtime(os.path.getctime(filename))

            if ftime >= ntime:
                if given_directory and given_directory != directory:
                    filename2 = os.path.join(given_directory, nfname)
                    # Make sure the output directory exists
                    if not os.path.isdir(os.path.dirname(filename2)):
                        os.makedirs(os.path.abspath(
                            os.path.dirname(filename2)))

                    shutil.copy(filename, filename2)
                    return Path(filename2)

                return Path(filename)

            else:
                logger.info(f"File older than {force_download}, updating...")

    # If we get here, we're going to try to download the file
    needs_internet()
    # if os.getenv('PLENPY_NO_INTERNET', '').lower() in ('1', 'true', 'yes'):
    #     raise InternetNotAllowedError('Will not download resource from the '
    #                                   'internet because enironment variable '
    #                                   'PLENPY_NO_INTERNET is set.')

    # Can we proceed with auto-download?
    if not auto:
        raise NeedDownloadError()

    # Get filename to store to and make sure the dir exists
    filename = os.path.join(directory, nfname)
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.abspath(os.path.dirname(filename)))

    # let's go get the file
    _fetch_file(url, filename)
    return Path(filename)


def _fetch_file(url: str,
                file_name: str,
                print_destination: bool = True):
    """Load requested file, downloading it if needed or requested

    Args:
        url: The url of file to be downloaded.

        file_name: Name, along with the path,
            of where downloaded file will be saved.

        print_destination: If true, destination of where file was saved will
            be printed after download finishes.

        resume: If true, try to resume partially downloaded files.

    """
    # Adapted from NISL:
    # https://github.com/nisl/tutorial/blob/master/nisl/datasets.py

    logger.info(f'File: {os.path.basename(file_name)} was not found '
                ' on your computer. Downloading it now...')

    temp_file_name = file_name + ".part"
    local_file = None
    initial_size = 0
    errors = []
    for tries in range(4):
        try:
            logger.info(f'Try {tries + 1}. Download from {url}')
            # Checking file size and displaying it alongside the download url
            req = Request(url, headers={'User-Agent': 'Magic Browser'})
            remote_file = urlopen(req, timeout=5.)
            local_file = open(temp_file_name, 'wb')

            if 'Content-Length' in [i for i in remote_file.headers]:
                file_size = int(remote_file.headers['Content-Length'].strip())
                size_str = _sizeof_fmt(file_size)

                # Downloading data (can be extended to resume if need be)
                _chunk_read(remote_file, local_file, initial_size=initial_size)

            else:
                local_file.write(remote_file.read())

            # temp file must be closed prior to the move
            if not local_file.closed:
                local_file.close()
            shutil.move(temp_file_name, file_name)
            if print_destination is True:
                sys.stdout.write('File saved as %s.\n' % file_name)
            break

        except Exception as e:
            errors.append(e)
            print('Error while fetching file: %s.' % str(e))

        finally:
            if local_file is not None:
                if not local_file.closed:
                    local_file.close()
    else:
        raise IOError(f'Unable to download {os.path.basename(file_name)}. '
                      'Perhaps there is a no internet connection? '
                      'If there is, please report this problem.')


def _chunk_read(response: addinfourl,
                local_file,
                chunk_size: int = 8192,
                initial_size: int = 0):
    """Download a file chunk by chunk and show advancement.

    Can also be used when resuming downloads over http.

    Args:
        response: Response to the download request in order to get file size.

    local_file (file): Hard disk file where data should be written.

    chunk_size: Size of downloaded chunks. Default: 8192

    initial_size: If resuming, indicate the initial size of the file.

    """
    # Adapted from NISL:
    # https://github.com/nisl/tutorial/blob/master/nisl/datasets.py

    bytes_so_far = initial_size
    # Returns only amount left to download when resuming, not the size of the
    # entire file
    total_size = int(response.headers['Content-Length'].strip())
    total_size += initial_size

    logger.info('Downloading...')

    while True:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)
        if not chunk:
            break

        local_file.write(chunk)
        time.sleep(0.0001)
        logger.info('...done')


def _sizeof_fmt(num):
    """Turn number of bytes into human-readable str"""
    units = ['bytes', 'kB', 'MB', 'GB', 'TB', 'PB']
    decimals = [0, 0, 1, 2, 2, 2]
    """Human friendly file size"""
    if num > 1:
        exponent = min(int(log(num, 1024)), len(units) - 1)
        quotient = float(num) / 1024 ** exponent
        unit = units[exponent]
        num_decimals = decimals[exponent]
        format_string = '{0:.%sf} {1}' % num_decimals
        return format_string.format(quotient, unit)
    return '0 bytes' if num == 0 else '1 byte'


# Set internet environment variable on import
if has_internet():
    os.environ["PLENPY_NO_INTERNET"] = "0"
else:
    os.environ["PLENPY_NO_INTERNET"] = "1"
