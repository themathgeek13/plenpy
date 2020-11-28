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


"""The camera package of plenpy.

The package provides a framework to implement camera models as well as
pre developed cameras such as a basic
:class:`plenpy.cameras.rgb_camera.RgbCamera` or light field cameras such as
the :class:`plenpy.cameras.lytro_illum.LytroIllum` camera.

The framework provides an API to implement own camera modules (or classes).

"""

import plenpy.logg

from plenpy.cameras.lytro_illum import LytroIllum
from plenpy.cameras.rgb_camera import RgbCamera

logger = plenpy.logg.get_logger()


