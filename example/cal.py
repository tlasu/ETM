#%%
import numpy as np
import etm.util as util
from etm.domain import cal_domain
from etm.TyphoonTrack import TyphoonTrack,resample_by_time_interval
from etm.etm import cal_pressure, cal_gradient_velocity, cal_gradient_velocity_top, cal_utf_velocity
import matplotlib.pyplot as plt


#%%
area = cal_domain()
area.extnt = [120, 125, 30, 35]
area.nlons = 100
area.nlats = 100
delta = 0.1
area.lons = 120 + delta * np.arange(area.nlons) # １次元の経度データ
area.lats = 30 + delta * np.arange(area.nlats) # １次元の緯度データ

area.lons = np.linspace(area.extnt[0], area.extnt[1], area.nlons)
area.lats = np.linspace(area.extnt[2], area.extnt[3], area.nlats)
area.array_lons, area.array_lats = np.meshgrid(area.lons, area.lats)
#%%
class cal_domain:
    def __init__(self) -> None:
        self.array_lons = None
        self.array_lats = None
        self.nlons = 0
        self.nlats = 0
        self.lons = []
        self.lats = []
        self.gridType:str | None = None #"rect" or "polar"
        pass
    def latlons(self):
        if self.gridType == "wgs84":
            lons, lats = np.meshgrid(self.lons, self.lats)
        else:
            transformer = pyproj.Transformer.from_crs(
                "EPSG:6677",    # 平面直角座標系第9系（関東）
                "EPSG:4326",    # WGS84（経度・緯度）
                always_xy=True
            )
            _x, _y = np.meshgrid(self.x, self.y)
            lons, lats = transformer.transform(_x, _y)
        return lons, lats
d = cal_domain()
d.gridType = "grid"
d.extent = [0, 100000, 0, 100000]
d.nx = 50
d.ny = 50
d.x = np.linspace(d.extent[0], d.extent[1], d.nx)
d.y = np.linspace(d.extent[2], d.extent[3], d.ny)
lons, lats = d.latlons()

#%%
plt.scatter(lons, lats)
#d.lons = np.linspace(120, 125, 100)
#d.lats = np.linspace(30, 35, 100)
#d.latlons()
d.x.shape, lats.shape
#%%
transformer = pyproj.Transformer.from_crs(
    "EPSG:6677",    # 平面直角座標系第9系（関東）
    "EPSG:4326",    # WGS84（経度・緯度）
    always_xy=True
)
#%%
lons, lats = transformer.transform(d.x, d.y)

#%%
# # %%
# typhoons = util.read_parquet("../jma-bst/typhoon_all.parquet")
# # %%
# # %%
# # 3次元配列として初期化（時間、緯度、経度）
# cyclone = typhoons[-15]
# cyclone.hourly_data = resample_by_time_interval(cyclone.df, time_interval_hours=1.0)
# print(f"LON: {cyclone.hourly_data.LON.min()}, {cyclone.hourly_data.LON.max()}")
# print(f"LAT: {cyclone.hourly_data.LAT.min()}, {cyclone.hourly_data.LAT.max()}")
# # %%
# p_array = np.zeros((len(cyclone.hourly_data), area.nlats, area.nlons))
# array_grad_velo_u = np.zeros((len(cyclone.hourly_data), area.nlats, area.nlons))
# array_grad_velo_v = np.zeros((len(cyclone.hourly_data), area.nlats, area.nlons))
# array_tf_velo_u = np.zeros((len(cyclone.hourly_data), area.nlats, area.nlons))
# array_tf_velo_v = np.zeros((len(cyclone.hourly_data), area.nlats, area.nlons))
# array_tf_velo = []
# u10 = []
# v10 = []
# %%
x = area.array_lons
y = area.array_lats

import pyproj
grs80 = pyproj.Geod(ellps='GRS80')
r0 = 100.
pc = 960
center_lon = 124.6
center_lat = 35
r_array = grs80.inv(x, y, np.zeros_like(x) + center_lon,
                          np.zeros_like(y) + center_lat)[2]*0.001
#grad_velo = cal_gradient_velocity_top(pc,r0,r_array)
grad_velo_u, grad_velo_v = cal_gradient_velocity(pc,r0,r_array,x,y,center_lon,center_lat)
utf_velo_u, utf_velo_v = cal_utf_velocity(pc, r0, r_array, moving_velocity=40. / 3.6, direction=90)
# x方向とy方向成分から絶対値を計算
grad_velo_abs = np.hypot(grad_velo_u, grad_velo_v)
utf_velo_abs = np.hypot(utf_velo_u, utf_velo_v)
print(f"utf_velo_u.max(), utf_velo_u.min(): {utf_velo_u.max():.2f}, {utf_velo_u.min():.2f}")
print(f"utf_velo_v.max(), utf_velo_v.min(): {utf_velo_v.max():.2f}, {utf_velo_v.min():.2f}")

cb_min, cb_max = 0, 40
cb_div = 16
interval_of_cf = np.linspace(cb_min, cb_max, cb_div+1)

u10_abs = np.hypot(grad_velo_u+utf_velo_u, grad_velo_v+utf_velo_v)
cont = plt.contourf(x,y,u10_abs, interval_of_cf, cmap="jet")
plt.colorbar(cont)
print(grad_velo_abs.max(), grad_velo_abs.min())
print(utf_velo_abs.max(), utf_velo_abs.min())
print(u10_abs.max(), u10_abs.min())
#%%
plt.plot(u10_abs[::-1,55])
plt.plot(u10_abs[::-1,54])
plt.grid()
plt.ylim(0,40)
# 基本的なquiver（ベクトル図）
# %%
grs80.inv(124.6, 35, 125, 35)
# %%
u10_abs.shape
# %%
# 特定の時刻のデータを使用
cnt = 20
u = grad_velo_u  # x方向成分
v = grad_velo_v  # y方向成分
# %%
dx = x - center_lon
dy = y - center_lat
np.arctan(dy / dx)
# %%
# ベクトル図を描画
plt.quiver(x, y, u, v,scale=1000)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Wind Vector Field')
plt.xlim(123,127)
plt.ylim(34,38)
plt.show()
#np.cos(np.arctan(dy / dx) + 2 * np.pi/3)
#np.cos(np.arctan(dy / dx) + np.deg2rad(90 + 30))
# %%

for cnt in range(len(cyclone.hourly_data)):
    r0 = 50
    pc = cyclone.hourly_data.PRES[cnt]
    center_lon = cyclone.hourly_data.LON[cnt]
    center_lat = cyclone.hourly_data.LAT[cnt]
    
    # r0_array = np.zeros_like(area.array_lons) + r0

    r_array = grs80.inv(x, y, np.zeros_like(x) + center_lon,
                              np.zeros_like(y) + center_lat)[2]*0.001
    r_array[r_array == 0] = 1e-10
    p_array[cnt, :, :] = cal_pressure(pc, r0, r=r_array)
    # array_grad_velo_u[cnt,:,:], array_grad_velo_v[cnt,:,:] = 
    array_grad_velo_u[cnt,:,:], array_grad_velo_v[cnt,:,:] = cal_gradient_velocity(pc,r0,r_array,x,y,center_lon,center_lat)
    array_tf_velo_u[cnt,:,:], array_tf_velo_v[cnt,:,:] = cal_utf_velocity(pc,r0,r_array,
                                                                          cyclone.hourly_data.moving_velocity[cnt],
                                                                          cyclone.hourly_data.direction[cnt])
    #u10 += [array_tf_velo[cnt][0] + array_grad_velo[cnt][0]]
    #v10 += [array_tf_velo[cnt][1] + array_grad_velo[cnt][1]]
# %%
plt.plot([array_grad_velo_u[cnt,:,:].max() for cnt in range(len(cyclone.hourly_data))], "-")
# %%
cnt = 20
r0 = 50
pc = cyclone.hourly_data.PRES[cnt]
center_lon = cyclone.hourly_data.LON[cnt]
center_lat = cyclone.hourly_data.LAT[cnt]

# r0_array = np.zeros_like(area.array_lons) + r0
# %%
plt.plot(cyclone.hourly_data.WIND, "-")
# %%
# 時間軸全体での各格子点の最小気圧（2次元配列）
p_min_2d = np.min(p_array, axis=0)
print(f"p_min_2d.shape: {p_min_2d.shape}")

# %%
array_grad_velo = np.sqrt(array_grad_velo_u[:,:,:] ** 2 + array_grad_velo_v[:,:,:] ** 2)
grad_velo_max_2d = np.max(array_grad_velo, axis=0)
# %%
grad_velo_max_2d.max()
# %%
cb_min, cb_max = 0, 30
cb_div = 24
interval_of_cf = np.linspace(cb_min, cb_max, cb_div+1)
cnt = 50
#cont = plt.contourf(x,y,p_array[cnt], cmap="jet")
#cont = plt.contourf(x,y,p_min_2d, cmap="jet")
#cont = plt.contourf(x,y,grad_velo_max_2d, interval_of_cf, cmap="jet")
cont = plt.contourf(x,y,grad_velo_max_2d, interval_of_cf, cmap="jet")
#plt.plot(cyclone.hourly_data.LON, cyclone.hourly_data.LAT, "-")
plt.colorbar(cont)
plt.xlim(125,145)
plt.ylim(10,50)
# %%
u10 = array_grad_velo[cnt][0]
v10 = array_grad_velo[cnt][1]
Wind = np.sqrt(u10**2+v10**2)
#%%
Wind.shape
#%%
cb_min, cb_max = 0, 40
cb_div = 24
interval_of_cf = np.linspace(cb_min, cb_max, cb_div+1)
cnt = 100
cont = plt.contourf(x,y,np.sqrt(u10**2+v10**2),interval_of_cf, cmap="jet")
plt.xlim(125,135)
plt.ylim(10,30)
#%%
from ipywidgets import interact
def plot(cnt):
    cont = plt.contourf(x,y,np.sqrt(u10[cnt]**2+v10[cnt]**2),interval_of_cf, cmap="jet")
    plt.colorbar(cont)
# %%
interact(plot, cnt=(0,len(cyclone.hourly_data)-2, 1))
# %%
len(cyclone.hourly_data)
# %%
# %%
r0 = 100
f = 2 * 7.29 * 10 **-5 * np.sin(np.deg2rad(35))
print(f)
dp = (1013 - 960) * 100
RHOA = 1.22
r_array = np.linspace(0,500,100)
a1 = -0.5 * r_array * f
a2 = a1**2
#Ugr = a1
r_array[r_array == 0] = 1e-10
Ugr = 0.66*(a1 + np.sqrt(a2 + dp / RHOA * r0/r_array * np.exp(-r0/r_array)))  
plt.plot(r_array,Ugr)

#print(Ugr)
# %%
#plt.plot(r_array,r0/r_array * np.exp(-r0/r_array))
# %%
# %%
