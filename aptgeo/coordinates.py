"""
Functionality for generating coordinates for passes and images.
"""
import os
from typing import Dict, Union
from functools import lru_cache
from urllib.request import urlopen

import numpy as np
from tletools import TLE

import astropy.coordinates as coords
import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.time import Time
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import propagate

__all__ = ['get_pass_coordinates', 'get_satellite_orbit']


# Cache this so we don't download the file more than once, but also don't cache
# it to disk or else we will have to worry about it being out of date etc.
@lru_cache(maxsize=None)
def get_weather_tle(tle_location: Union[str, os.PathLike]) -> Dict[str, TLE]:
    """
    Get and load a TLE file for weather satellites.
    """
    if isinstance(tle_location, str):
        with urlopen(tle_location) as req:
            weather_tle_bytes = req.read()
    elif isinstance(tle_location, os.PathLike):
        with open(tle_location, "rb") as fobj:
            weather_tle_bytes = fobj.read()
    else:
        raise TypeError("tle_location must be a string if a url or PathLike if a local path")

    weather_tle_lines = weather_tle_bytes.decode().strip().splitlines()

    n_tle = int(len(weather_tle_lines) / 3)
    weather_tles = [weather_tle_lines[i*3:(i+1)*3] for i in range(n_tle)]

    return {str(tle.name): tle for tle in [TLE.from_lines(*tle) for tle in weather_tles]}


def get_satellite_orbit(satellite_name: str,
                        tle_location: Union[str, os.PathLike] = "http://www.celestrak.com/NORAD/elements/weather.txt"
                        ) -> Orbit:
    """
    Given a weather satellite name as found in the TLE file, return a `poliastro.twobody.Orbit` object.
    """
    return get_weather_tle(tle_location)[satellite_name].to_orbit()


def get_pass_coordinates(satellite: Orbit,
                         start_time: Time,
                         pass_length: u.Quantity,
                         sample_time: u.Quantity) -> SkyCoord:
    """
    Given an `~poliastro.twobody.Orbit` object generate the coordinates for a pass.

    The pass is defined based on a start time, a duration and a sample time.
    A coordinate will be generated every sample time for the duration of the
    pass, inclusive of both the start and end time.

    Parameters
    ----------
    satellite
        The orbit to sample for coordinates.

    start_time
        The time to propagate the orbit to for sampling, will be the time of
        the 0th coordinate returned.

    pass_length
        An amount of time to sample the orbit for.

    sample_time
        The time interval to sample the orbit at. ``pass_length / sample_time``
        coordinates will be returned.

    Returns
    -------
    coordinates
        The coordinates of the satellites in an `astropy.coordinates.ITRS` frame.

    """
    start_time = Time(start_time)
    sample_time = sample_time.to(u.s)
    pass_length = pass_length.to(u.s)

    pass_times = np.arange(0,
                           (pass_length + sample_time).to_value(u.s),
                           sample_time.to_value(u.s)) * u.s

    # Propagate the orbit to the pass start time
    satellite = satellite.propagate(start_time)

    # Generate the coordinates of the satellite during the pass
    sc_position = propagate(satellite, pass_times).without_differentials()
    sc_position = coords.SkyCoord(satellite.get_frame().realize_frame(sc_position))
    sc_position = sc_position.transform_to(coords.ITRS())
    sc_position.representation_type = coords.WGS84GeodeticRepresentation

    return sc_position
