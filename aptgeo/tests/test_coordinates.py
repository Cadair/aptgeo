from pathlib import Path

import pytest
from pkg_resources import resource_filename
from tletools.tle import TLE

import astropy.units as u
from astropy.coordinates import ITRS, SkyCoord, WGS84GeodeticRepresentation
from astropy.time import Time
from poliastro.twobody import Orbit

import aptgeo.coordinates as aptcoords


@pytest.fixture
def local_tle():
    return Path(resource_filename("aptgeo.data.test", "weather_tle.txt"))


@pytest.fixture(params=["NOAA 15", "NOAA 18", "NOAA 19"])
def freeflyer_coordinates(request):
    """
    These coordinates were kindly calculated by someone with freeflyer to serve as a reference.
    """
    coords = {
        "NOAA 15": {"time": Time("2021-05-12T00:00:00"),
                    "position": (0*u.deg, 0*u.deg)},
        "NOAA 18": {"time": Time("2021-05-12T00:00:00"),
                    "position": (0*u.deg, 0*u.deg)},
        "NOAA 19": {"time": Time("2021-05-12T00:00:00"),
                    "position": (0*u.deg, 0*u.deg)},
    }
    return request.param, coords[request.param]


def test_local_tle(local_tle):
    tles = aptcoords.get_weather_tle(local_tle)
    assert isinstance(tles, dict)
    for key in ["NOAA 19", "NOAA 18", "NOAA 15"]:
        assert key in tles
        assert isinstance(tles[key], TLE)


def test_get_orbit_local(local_tle):
    for key in ["NOAA 19", "NOAA 18", "NOAA 15"]:
        orbit = aptcoords.get_satellite_orbit(key, tle_location=local_tle)
        assert isinstance(orbit, Orbit)


def test_get_coordinates(local_tle):
    orbit = aptcoords.get_satellite_orbit("NOAA 15", local_tle)

    coords = aptcoords.get_pass_coordinates(orbit, Time("2021-05-12T00:00:00"), 20*u.s, 10*u.s)

    assert coords.shape == (3,)

    assert isinstance(coords, SkyCoord)
    assert isinstance(coords.frame, ITRS)

    assert coords.representation_type == WGS84GeodeticRepresentation


@pytest.mark.skip(reason="Waiting for reference coords")
def test_accuracy_known_values(local_tle, freeflyer_coordinates):
    satellite_name, coords = freeflyer_coordinates

    orbit = aptcoords.get_satellite_orbit(satellite_name, local_tle)

    pass_coord = aptcoords.get_pass_coordinates(orbit, coords["time"], pass_length=1*u.s, sample_time=1*u.s)[0]

    assert u.allclose(coords["position"][0], pass_coord.lon)
    assert u.allclose(coords["position"][1], pass_coord.lat)
