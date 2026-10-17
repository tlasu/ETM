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


from etm.rotation import rotate_coordinates_around_center  # noqa: E402
from etm.TyphoonTrack import geod_inv  # noqa: E402


@pytest.mark.parametrize("center_lat", [10.0, 35.0, 55.0])
@pytest.mark.parametrize("angle", [30, 90, 200])
def test_rotation_preserves_distance_from_center(center_lat, angle):
    """高緯度でも回転中心からの距離が保たれる"""
    lon = np.array([140.0, 145.0, 150.0])
    lat = np.array([center_lat - 5, center_lat + 3, center_lat])
    rlon, rlat = rotate_coordinates_around_center(lon, lat, 145.0, center_lat, angle)
    c_lon, c_lat = np.full(3, 145.0), np.full(3, center_lat)
    d_before = geod_inv(c_lon, c_lat, lon, lat)[2]
    d_after = geod_inv(c_lon, c_lat, rlon, rlat)[2]
    np.testing.assert_allclose(d_after, d_before, rtol=1e-9)


def test_rotation_counterclockwise():
    """北にある点を90°回すと西に来る（正の角度は反時計回り）"""
    rlon, rlat = rotate_coordinates_around_center([135.0], [40.0], 135.0, 35.0, 90)
    az = geod_inv(135.0, 35.0, rlon[0], rlat[0])[0]
    assert az == pytest.approx(-90, abs=1e-6)


@pytest.mark.parametrize("center_lat", [10.0, 50.0])
def test_projected_matches_geodesic_rotation(track, center_lat):
    """正距方位図法での回転と測地線での回転は一致する"""
    a = rotate_typhoon_track(track, 130.0, center_lat, 60)
    b = rotate_typhoon_track_projected(track, 130.0, center_lat, 60)
    np.testing.assert_allclose(a.df["LON"], b.df["LON"], atol=1e-6)
    np.testing.assert_allclose(a.df["LAT"], b.df["LAT"], atol=1e-6)
