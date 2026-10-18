import numpy as np
import pytest

from etm.domain import Grid


def test_from_extent():
    grid = Grid.from_extent(130, 135, 30, 32, 0.5)
    assert grid.shape == (5, 11)
    assert grid.lons[0, 0] == 130 and grid.lons[0, -1] == 135
    assert grid.lats[0, 0] == 30 and grid.lats[-1, 0] == 32
    assert grid.crs == "EPSG:4326"


def test_from_projected():
    """平面直角座標系第IX系の原点は 北緯36°・東経139°50′"""
    grid = Grid.from_projected(np.array([0.0, 1000.0]), np.array([0.0, 1000.0, 2000.0]), "EPSG:6677")
    assert grid.shape == (3, 2)
    assert grid.lons[0, 0] == pytest.approx(139 + 50 / 60)
    assert grid.lats[0, 0] == pytest.approx(36.0)
    assert grid.lats[2, 0] > grid.lats[0, 0]
