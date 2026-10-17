import numpy as np
import pytest
import xarray as xr

from etm.domain import Grid
from etm.field import compute_wind_field, prepare_track


@pytest.fixture
def northward_track(make_track):
    # 6時間ごとに緯度1°北上（約5.1m/s）
    return make_track([130.0] * 5, [20.0, 21.0, 22.0, 23.0, 24.0])


@pytest.fixture
def grid():
    return Grid.from_extent(126, 134, 18, 26, 0.25)


def test_prepare_track(northward_track):
    df = prepare_track(northward_track, dt_hours=1.0, rmax=60)
    assert len(df) == 25
    assert (df["Rmax"] == 60).all()
    np.testing.assert_allclose(df["direction"], 0, atol=1e-6)
    np.testing.assert_allclose(df["moving_velocity"], 110_700 / 21_600, rtol=1e-2)


def test_compute_wind_field(northward_track, grid):
    ds = compute_wind_field(northward_track, grid, dt_hours=6.0, rmax=60)
    assert dict(ds.sizes) == {"time": 5, "y": 33, "x": 33}
    snap = ds.isel(time=2)  # 中心 (130, 22)
    center = snap.sel(x=130, y=22, method="nearest")
    assert float(center.pressure) == pytest.approx(960, abs=0.5)
    assert float(snap.pressure.max()) < 1013

    # 北半球では反時計回り: 中心の東側は北向き、西側は南向き
    east = snap.sel(x=131, y=22, method="nearest")
    west = snap.sel(x=129, y=22, method="nearest")
    assert float(east.v10) > 0 > float(west.v10)
    # 北進する台風は進行方向右側（東側）が強い
    assert float(east.wind_speed) > float(west.wind_speed)


def test_compute_wind_field_period_and_rmax_model(northward_track, grid):
    ds = compute_wind_field(northward_track, grid, dt_hours=1.0, rmax="kokuso",
                            start="2000-08-01 06:00", end="2000-08-01 12:00")
    assert ds.sizes["time"] == 7
    # kokuso: Pc=960hPa -> Rmax = 80 + 1.633*10
    np.testing.assert_allclose(ds.rmax, 96.33)
    assert ds.attrs["rmax_model"] == "kokuso"


def test_netcdf_roundtrip(northward_track, grid, tmp_path):
    ds = compute_wind_field(northward_track, grid, dt_hours=6.0, rmax=60)
    path = tmp_path / "field.nc"
    ds.to_netcdf(path)
    with xr.open_dataset(path) as loaded:
        np.testing.assert_allclose(loaded.wind_speed, ds.wind_speed)
