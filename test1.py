#%%
import TyphoonTrack
import numpy as np
import pyproj
import matplotlib.pyplot as plt

#%%
P_CONSTANT = 1013
RHOA = 1.22
f = 7.29 * 10 **-5

grs80 = pyproj.Geod(ellps='GRS80')

def cal_pressure(center_pressure, r0=50, r=None):
    dp = P_CONSTANT - center_pressure
    return center_pressure + dp * np.exp(-r0/r)

def cal_gradient_velocity_top(center_pressure, r0=50, r=None, x=None, y=None,x0=0,y0=0):
    dp = (P_CONSTANT - center_pressure)
    Ugr = -0.5*r*f+np.sqrt((r*f *0.5)**2 + dp / RHOA * r0/r * np.exp(r0/r))   
    return Ugr

def cal_gradient_velocity(center_pressure, r0=50, r=None, x=None, y=None,x0=0,y0=0):
    dp = (P_CONSTANT - center_pressure)
    Ugr = -0.5*r*f+np.sqrt((r*f *0.5)**2 + dp / RHOA * r0/r * np.exp(r0/r))  
    c1 = cal_c1(Ugr, r0, r,x0,y0,x,y)
    ugr_x = c1 * Ugr * np.cos(np.arctan((y-y0) / (x-x0)) + 2*np.pi/3)
    ugr_y = c1 * Ugr * np.sin(np.arctan((y-y0) / (x-x0)) + 2*np.pi/3)
    return ugr_x, ugr_y

def cal_c1(urg, r0=50, r=None,x0=0,y0=0,x=0,y=0):
    r0 = 50
    X = r/r0
    Xp = 0.5
    k = 2.5
    c1_inf = 2/3
    c1_XP = 1.2
    c1 = c1_inf + (c1_XP - c1_inf)*(X/Xp)**(k-1.) * np.exp((1.-1/k)*(1.-(X/Xp)**(k)))
    return c1

def cal_utf_velocity(cyclone,center_pressure, r0=50, r=None, x=None, y=None,x0=0,y0=0):
    ugr_r = cal_gradient_velocity_top(pc,r0,r_array,x,y,x0,y0)
    ugr_r0 = cal_gradient_velocity_top(pc,r0,r0_array,x,y,x0,y0)
    moving_velocity = cyclone.hourly_data.moving_velocity[cnt]
    direction = cyclone.hourly_data.direction[cnt]
    utf = ugr_r / ugr_r0 * moving_velocity
    c2 = 0.65
    utf_x = c2 * utf * np.cos(np.deg2rad(direction))
    utf_y = c2 * utf * np.cos(np.deg2rad(direction))
    return utf_x, utf_y


#%%
#解析範囲の作成
class cal_area:
    def __init__(self) -> None:
        self.array_lons = None
        self.array_lats = None
        self.nlons = 0
        self.nlats = 0
        self.lons = []
        self.lats = []
        pass
#%%
area = cal_area()
area.nlons = 240
area.nlats = 180
delta = 0.25
area.lons = 120 + delta * np.arange(area.nlons) # １次元の経度データ
area.lats = 0 + delta * np.arange(area.nlats) # １次元の緯度データ
area.array_lons, area.array_lats = np.meshgrid(area.lons, area.lats) # ２次元の経度・緯度座標の準備

# %%
typhoons = TyphoonTrack.read_bsttxt("./bst_all.txt")
cyclone = typhoons[2]

# %%
cyclone.hourly_data = cyclone.set_moving_velocity(cyclone.hourly_data)
cyclone.hourly_data = cyclone.set_direction(cyclone.hourly_data)
# %%

# %%
p_array = []
array_grad_velo = []
array_tf_velo = []
u10 = []
v10 = []

for cnt in range(len(cyclone.hourly_data)):
    r0 = 50
    center_lon, center_lat = cyclone.hourly_data.lon[cnt], cyclone.hourly_data.lat[cnt]
    pc = cyclone.hourly_data.pc[cnt]
    r0_array = np.zeros_like(x) + r0
    r_array = grs80.inv(x, y, np.zeros_like(x) + center_lon,
                              np.zeros_like(y) + center_lat)[2]*0.001

    p_array += [cal_pressure(pc,r0,r=r_array)]
    array_grad_velo += [cal_gradient_velocity(pc,r0,r_array,x,y,center_lon,center_lat)]
    array_tf_velo += [cal_utf_velocity(cyclone,pc,r0,r_array,x,y,center_lon,center_lat)]
    u10 += [array_tf_velo[cnt][0] + array_grad_velo[cnt][0]]
    v10 += [array_tf_velo[cnt][1] + array_grad_velo[cnt][1]]
# %%
# %%
u10 = array_tf_velo[0][0] + array_grad_velo[0][0]
v10 = array_tf_velo[0][1] + array_grad_velo[0][1]
# %%
array_grad_velo
#%%
cb_min, cb_max = 0, 40
cb_div = 24
interval_of_cf = np.linspace(cb_min, cb_max, cb_div+1)

cnt = 200
cont = plt.contourf(x,y,np.sqrt(u10[cnt]**2+v10[cnt]**2),interval_of_cf, cmap="jet")
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
np.rad2deg(2*np.pi/3)
# %%
