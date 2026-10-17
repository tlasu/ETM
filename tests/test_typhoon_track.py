import numpy as np
import pandas as pd
import pytest

from etm.TyphoonTrack import (
    add_cumulative_distance,
    calculate_moving_velocity,
    interpolate_by_cumulative_distance,
    segment_distances,
)


def test_segment_distances_geodesic():
    """赤道上の経度1°は約111.32km、北緯60°では約半分"""
    df = pd.DataFrame({"LON": [150.0, 151.0, 150.0, 151.0], "LAT": [0.0, 0.0, 60.0, 60.0]})
    d = segment_distances(df)
    assert d[0] == 0
    assert d[1] == pytest.approx(111_320, rel=1e-3)
    assert d[3] == pytest.approx(55_800, rel=1e-2)


def test_moving_velocity(make_track):
    """6時間で緯度1°（約110.7km）北上 -> 約5.1m/s"""
    track = make_track([140.0, 140.0, 140.0], [20.0, 21.0, 22.0])
    df = calculate_moving_velocity(track)
    assert df["moving_velocity"].iloc[0] == 0
    assert df["moving_velocity"].iloc[1] == pytest.approx(110_700 / 21_600, rel=1e-2)
    assert "x" not in df.columns


def test_cumulative_distance_with_non_default_index(make_track):
    """0始まりの連番でないインデックスでも行が増えない"""
    df = make_track([130.0, 131.0, 132.0], [20.0, 21.0, 22.0]).df.copy()
    df.index = [10, 11, 12]
    result = add_cumulative_distance(df)
    assert len(result) == 3
    np.testing.assert_allclose(result["cumulative_distance"], np.cumsum(segment_distances(df)))


def test_interpolate_by_cumulative_distance_end_point(make_track):
    track = make_track([130.0, 131.0, 132.0], [20.0, 21.0, 22.0])
    result = interpolate_by_cumulative_distance(track, constant_velocity=5.0)
    assert result["LON"].iloc[-1] == pytest.approx(132.0)
    assert result["LAT"].iloc[-1] == pytest.approx(22.0)
    np.testing.assert_allclose(np.diff(result["cumulative_distance"].iloc[:-1]), 5.0 * 3600)


def test_set_direction(make_track):
    track = make_track([130.0, 131.0, 131.0], [20.0, 20.0, 21.0])
    track.set_direction()
    direction = track.df["direction"]
    assert direction.iloc[0] == pytest.approx(90, abs=0.5)  # 東進
    assert direction.iloc[1] == pytest.approx(0, abs=1e-6)  # 北進
    assert np.isnan(direction.iloc[2])
