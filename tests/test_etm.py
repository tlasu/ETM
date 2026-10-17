import numpy as np
import pytest

from etm.etm import (
    RHOA,
    cal_gradient_velocity_top,
    cal_pressure,
    cal_utf_velocity,
    coriolis_parameter,
)


@pytest.mark.parametrize("r_km", [30.0, 100.0, 300.0, 800.0])
@pytest.mark.parametrize("lat", [15.0, 35.0])
def test_gradient_wind_balance(r_km, lat):
    """V²/r + fV = (1/ρ)∂P/∂r を満たすこと（SI単位で確認）"""
    pc, r0 = 950.0, 60.0
    v = cal_gradient_velocity_top(pc, r0, r_km, lat)

    h = 1e-3  # km
    dpdr = (cal_pressure(pc, r0, r_km + h) - cal_pressure(pc, r0, r_km - h)) / (2 * h)
    dpdr_si = dpdr * 100 / 1000  # hPa/km -> Pa/m

    r_m = r_km * 1000
    lhs = v**2 / r_m + coriolis_parameter(lat) * v
    assert lhs == pytest.approx(dpdr_si / RHOA, rel=1e-5)


def test_gradient_wind_does_not_modify_input():
    r = np.array([0.0, 10.0, 100.0])
    cal_gradient_velocity_top(960, 50, r)
    assert r[0] == 0.0


def test_coriolis_reduces_gradient_wind_at_large_radius():
    """高緯度ほどコリオリ項が効いて外側の傾度風は弱くなる"""
    assert cal_gradient_velocity_top(960, 50, 500.0, 40.0) < cal_gradient_velocity_top(960, 50, 500.0, 10.0)


@pytest.mark.parametrize(
    "direction, sign_x, sign_y",
    [(0, 0, 1), (90, 1, 0), (180, 0, -1), (270, -1, 0)],
)
def test_utf_velocity_uses_azimuth(direction, sign_x, sign_y):
    """direction は北=0°、時計回りの方位角"""
    ux, uy = cal_utf_velocity(960, 50, np.array([50.0]), 10.0, direction)
    assert np.sign(np.round(ux[0], 8)) == sign_x
    assert np.sign(np.round(uy[0], 8)) == sign_y


def test_set_direction_matches_utf_convention(make_track):
    """北進する台風の移動風は北向き成分を持つ"""
    track = make_track([130.0, 130.0, 130.0], [20.0, 21.0, 22.0])
    track.set_direction()
    direction = track.df["direction"].iloc[0]
    ux, uy = cal_utf_velocity(960, 50, np.array([50.0]), 10.0, direction)
    assert abs(ux[0]) < 1e-6
    assert uy[0] > 0


@pytest.mark.parametrize("x0, y0", [(130.0, 20.0), (140.0, 60.0)])
@pytest.mark.parametrize("dlon, dlat", [(1.0, 0.0), (1.0, 1.0), (-2.0, 0.5), (0.0, -1.0)])
def test_gradient_wind_direction_is_geodesic(x0, y0, dlon, dlat):
    """風向は中心からの方位に対して反時計回り接線から30°内向き（数学角で+120°）"""
    from etm.etm import cal_gradient_velocity
    from etm.TyphoonTrack import GEOD

    x, y = np.array([x0 + dlon]), np.array([y0 + dlat])
    ux, uy = cal_gradient_velocity(960, 50, np.array([100.0]), x, y, x0, y0)
    azimuth = GEOD.inv(x0, y0, x[0], y[0])[0]
    theta = 90 - azimuth
    wind_angle = np.degrees(np.arctan2(uy[0], ux[0]))
    assert (wind_angle - theta - 120 + 180) % 360 - 180 == pytest.approx(0, abs=1e-6)
