# %%
"""
台風1個分の気圧・風速場を計算する例（2019年 台風19号 HAGIBIS）
"""
import matplotlib.pyplot as plt
import numpy as np

import etm.util as util
from etm.domain import Grid
from etm.field import compute_wind_field

# %% 台風経路の読み込み
typhoons = util.read_parquet("../jma-bst/typhoon_all.parquet")
track = next(t for t in typhoons if t.name == "HAGIBIS" and t.season == 2019)

# %% 計算格子（経緯度0.1°）。平面直角座標系で計算する場合は Grid.from_projected(x, y, "EPSG:6677")
grid = Grid.from_extent(130, 150, 25, 45, resolution=0.1)

# %% 1時間間隔で計算（最大風速半径は kokuso モデル）
ds = compute_wind_field(track, grid, dt_hours=1.0, rmax="kokuso",
                        start="2019-10-11", end="2019-10-13")
print(ds)
# ds.to_netcdf("hagibis.nc")  # NetCDFで保存

# %% 計算範囲内で風速が最大となる時刻の風速場
i = int(np.argmax(ds.wind_speed.max(("y", "x")).values))
snap = ds.isel(time=i)
fig, ax = plt.subplots(figsize=(8, 8))
cont = ax.contourf(snap.lon, snap.lat, snap.wind_speed, levels=np.linspace(0, 40, 17), cmap="jet")
fig.colorbar(cont, ax=ax, label="wind speed [m/s]")
ax.contour(snap.lon, snap.lat, snap.pressure, levels=np.arange(940, 1014, 4), colors="k", linewidths=0.5)
step = 8
ax.quiver(snap.lon[::step, ::step], snap.lat[::step, ::step],
          snap.u10[::step, ::step], snap.v10[::step, ::step], scale=800)
ax.plot(ds.center_lon, ds.center_lat, "w.-", lw=1)
ax.set_title(f"{track.name} {str(snap.time.values)[:16]}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect(1 / np.cos(np.deg2rad(35)))
plt.show()

# %% 地点の時系列（東京付近）
point = ds.sel(x=139.8, y=35.6, method="nearest")
fig, ax1 = plt.subplots()
ax1.plot(point.time, point.wind_speed, "b-")
ax1.set_ylabel("wind speed [m/s]", color="b")
ax2 = ax1.twinx()
ax2.plot(point.time, point.pressure, "r-")
ax2.set_ylabel("pressure [hPa]", color="r")
fig.autofmt_xdate()
plt.show()
