import numpy as np
import pytest

from etm.rotation import rotate_typhoon_track, rotate_typhoon_track_projected


@pytest.fixture
def track(make_track):
    return make_track([125.0, 128.0, 131.0, 134.0], [20.0, 24.0, 28.0, 33.0])


@pytest.mark.parametrize("rotate", [rotate_typhoon_track, rotate_typhoon_track_projected])
@pytest.mark.parametrize("angle", [0, 360])
def test_rotation_identity(track, rotate, angle):
    rotated = rotate(track, 130.0, 30.0, angle)
    np.testing.assert_allclose(rotated.df["LON"], track.df["LON"], atol=1e-6)
    np.testing.assert_allclose(rotated.df["LAT"], track.df["LAT"], atol=1e-6)


@pytest.mark.parametrize("rotate", [rotate_typhoon_track, rotate_typhoon_track_projected])
def test_rotation_keeps_center(track, rotate):
    """回転中心上の点は動かない"""
    rotated = rotate(track, 128.0, 24.0, 45)
    assert rotated.df["LON"].iloc[1] == pytest.approx(128.0)
    assert rotated.df["LAT"].iloc[1] == pytest.approx(24.0)


def test_rotation_does_not_modify_original(track):
    before = track.df[["LON", "LAT"]].copy()
    rotate_typhoon_track(track, 130.0, 30.0, 90)
    np.testing.assert_array_equal(track.df[["LON", "LAT"]], before)
