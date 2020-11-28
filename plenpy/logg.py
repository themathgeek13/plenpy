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


"""Logging module of the plenpy package.

Load this module at the beginning of a submodule or script to enable logging
across the plenpy package. Logging is enabled by default.
This module controls the name and formatting of the plenpy logger.

Get the logger with :func:`get_logger()`.
Set the logging level with :func:`set_level()`.
Disable or enable with `enable()` or `disable()`
"""

import logging

logger = logging.getLogger("plenpy")
logger.setLevel(logging.INFO)

# Logging Format
FORMAT = "%(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT)


def get_logger():
    """Get the logger of the plenpy package.

    See Also:
        The Python ``logging`` module.

    """
    return logging.getLogger("plenpy")


def enable():
    """Enable the plenpy logger. The logger is enabled by default.
    """
    lg = get_logger()
    lg.propagate = True
    return


def disable():
    """Disable the plenpy logger. The logger is enabled by default.
    """
    lg = get_logger()
    lg.propagate = False
    return


def set_level(level="info"):
    """Set the logging level.

    Args:
        level: Logging level. Available levels: 'info', 'debug', 'warning' and
            'critical'.

    """

    if level.lower() == "info":
        logger.setLevel(logging.INFO)

    elif level.lower() == "debug":
        logger.setLevel(logging.DEBUG)

    elif level.lower() == "warning":
        logger.setLevel(logging.WARNING)

    elif level.lower() == "critical":
        logger.setLevel(logging.CRITICAL)

    else:
        raise ValueError("Option '{}' is not valid.".format(level))

    return


def set_format(format="normal"):

    if format.lower() == "normal":
        format_str = "%(levelname)s: %(message)s"

    elif format.lower() == "detail":
        format_str = "%(levelname)s: [%(filename)s:%(lineno)3s - " \
                     "%(funcName)30s() ] %(message)s"

    elif format.lower() == "time":
        format_str = '%(asctime)-15s %(message)s'

    else:
        raise ValueError("Option '{}' is not valid.".format(format))

    logging.basicConfig(format=format_str)
    return
