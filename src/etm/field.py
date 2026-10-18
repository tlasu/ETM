"""
台風経路から気圧・風速場の時系列を計算するモジュール
"""

import numpy as np
import pandas as pd
import xarray as xr

from etm.domain import Grid
from etm.etm import cal_gradient_velocity, cal_pressure, cal_utf_velocity
from etm.TyphoonTrack import TyphoonTrack, cal_rmax, geod_inv, resample_by_time_interval


def prepare_track(track: TyphoonTrack | pd.DataFrame,
                  dt_hours: float | None = 1.0,
                  rmax: str | float = "kokuso") -> pd.DataFrame:
    """
    風速場の計算に使う時系列を作る

    Parameters:
    -----------
    track : TyphoonTrack or pd.DataFrame
        台風トラックデータ
    dt_hours : float or None
        計算の時間間隔[時間]。None の場合はリサンプルしない
    rmax : str or float
        最大風速半径。数値[km]なら一定値、文字列なら cal_rmax のモデル名

    Returns:
    --------
    pd.DataFrame
        ISO_TIME, LON, LAT, PRES, Rmax, moving_velocity[m/s], direction[度] を含むDataFrame
        移動速度・進行方向は次の時刻へ向かう値（最終時刻は直前の値）
    """
    if dt_hours is None:
        df = (track.df if isinstance(track, TyphoonTrack) else track).copy()
        df = df.sort_values("ISO_TIME").reset_index(drop=True)
    else:
        df = resample_by_time_interval(track, dt_hours)
    if len(df) < 2:
        raise ValueError("移動速度を求めるには2時刻以上のデータが必要です")

    if isinstance(rmax, str):
        df["Rmax"] = cal_rmax(df, rmax)
    else:
        df["Rmax"] = float(rmax)

    lons = df["LON"].to_numpy(dtype=float)
    lats = df["LAT"].to_numpy(dtype=float)
    azimuth, _, distance = geod_inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    seconds = np.diff(df["ISO_TIME"].to_numpy()) / np.timedelta64(1, "s")
    df["moving_velocity"] = np.append(distance / seconds, distance[-1] / seconds[-1])
    df["direction"] = np.append(azimuth, azimuth[-1])
    return df


def compute_wind_field(track: TyphoonTrack | pd.DataFrame,
                       grid: Grid,
                       dt_hours: float | None = 1.0,
                       rmax: str | float = "kokuso",
                       start=None,
                       end=None) -> xr.Dataset:
    """
    台風経路から格子上の気圧・風速場の時系列を計算する

    Parameters:
    -----------
    track : TyphoonTrack or pd.DataFrame
        台風トラックデータ
    grid : Grid
        計算格子
    dt_hours : float or None
        計算の時間間隔[時間]。None の場合は元データの時刻で計算
    rmax : str or float
        最大風速半径。数値[km]なら一定値、文字列なら cal_rmax のモデル名
    start, end : 日時（文字列可）
        計算する期間（両端を含む）。省略時は全期間

    Returns:
    --------
    xr.Dataset
        pressure[hPa], u10, v10, wind_speed[m/s]（次元: time, y, x）
        中心気圧が欠損している時刻はNaN
    """
    df = prepare_track(track, dt_hours, rmax)
    if start is not None:
        df = df[df["ISO_TIME"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["ISO_TIME"] <= pd.Timestamp(end)]
    df = df.reset_index(drop=True)

    shape = (len(df),) + grid.shape
    pressure = np.full(shape, np.nan)
    u10 = np.full(shape, np.nan)
    v10 = np.full(shape, np.nan)

    for i, row in enumerate(df.itertuples(index=False)):
        if np.isnan(row.PRES) or np.isnan(row.Rmax):
            continue
        # 台風中心からの距離[km]（中心上の点はゼロ除算を避ける）
        r = np.maximum(geod_inv(row.LON, row.LAT, grid.lons, grid.lats)[2] * 1e-3, 1e-6)
        pressure[i] = cal_pressure(row.PRES, row.Rmax, r)
        ugr_x, ugr_y = cal_gradient_velocity(row.PRES, row.Rmax, r, grid.lons, grid.lats, row.LON, row.LAT)
        utf_x, utf_y = cal_utf_velocity(row.PRES, row.Rmax, r, row.moving_velocity, row.direction, lat=row.LAT)
        u10[i] = ugr_x + utf_x
        v10[i] = ugr_y + utf_y

    dims = ("time", "y", "x")
    name = track.name if isinstance(track, TyphoonTrack) else None
    return xr.Dataset(
        {
            "pressure": (dims, pressure, {"units": "hPa", "long_name": "sea level pressure"}),
            "u10": (dims, u10, {"units": "m s-1", "long_name": "eastward wind"}),
            "v10": (dims, v10, {"units": "m s-1", "long_name": "northward wind"}),
            "wind_speed": (dims, np.hypot(u10, v10), {"units": "m s-1", "long_name": "wind speed"}),
        },
        coords={
            "time": df["ISO_TIME"].to_numpy(),
            "x": grid.x,
            "y": grid.y,
            "lon": (("y", "x"), grid.lons),
            "lat": (("y", "x"), grid.lats),
            "center_lon": ("time", df["LON"].to_numpy(dtype=float)),
            "center_lat": ("time", df["LAT"].to_numpy(dtype=float)),
            "center_pressure": ("time", df["PRES"].to_numpy(dtype=float)),
            "rmax": ("time", df["Rmax"].to_numpy(dtype=float), {"units": "km"}),
        },
        attrs={
            "crs": grid.crs,
            "rmax_model": str(rmax),
            "typhoon_name": name or "",
        },
    )
