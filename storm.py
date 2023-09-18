#%%
import numpy as np
import matplotlib.pyplot as plt
# %%


track = typhoon_track()
# %%
with open("./bst2023.txt", "r") as f:
    l_strip = [s.rstrip() for s in f.readlines()]
   
# %%
typhoon_data = []
cnt = 0
while cnt < len(l_strip):
    header = l_strip[cnt].split()
    track_num_data = int(header[2])
    cnt += 1
    data = l_strip[cnt:cnt+track_num_data]
    cnt += track_num_data
    typhoon_data += [{"header":header, "data": data}]
#%%

#%%
class typhoon_track:
    def __init__(self, data=None) -> None:
        if type(data) is dict:
            header = data["header"]
            self.typhoon_number = header[1]
            self.num_data = int(header[2])
            self.tropical_cyclone_number = header[3]
            self.typhoon_name = header[7]
            typ_data = data['data']

            self.time = [dt.datetime.strptime(_l[0:8], "%y%m%d%H") for _l  in typ_data]
            self.rank = [int(_l[13]) for _l  in typ_data]
            self.lat = [float(_l[15:18])*0.1 for _l  in typ_data]
            self.lon = [float(_l[19:23])*0.1 for _l  in typ_data]
            self.pc = [float(_l[24:28]) for _l  in typ_data]
        pass
# %%
typhoon = typhoon_data[1]
header = typhoon["header"]
data = typhoon["data"]
typhoon_number = header[1]
track_num_data = int(header[2])
tropical_cyclone_number = header[3]
typhoon_name = header[7]


#%%
vars(typ1)
# %%
line = typhoon["data"][0]

# %%
import datetime as dt
typ1 = typhoon_track(typhoon)

# %%
typ1.lon
# %%

plt.plot(typ1.lon,typ1.lat)
# %%
plt.plot(typ1.pc)

# %%
def cal_pressure(center_pressure, r0=50, r=None):
    P_CONSTANT = 1013
    dp = P_CONSTANT - center_pressure
    return center_pressure + dp * np.exp(-r0/r)

r_array = np.arange(1,100)
cal_pressure(950,50,r=r_array)
# %%
typ1.lon,typ1.lat
# %%
np.array()

# %%
# %%
import pyproj
grs80 = pyproj.Geod(ellps='GRS80')
for cnt in range(7):
    distance = grs80.inv(x, y, np.zeros_like(x) + typ1.lon[cnt],
                            np.zeros_like(y) + typ1.lat[cnt])[2]*0.001
    plt.contourf(x,y,cal_pressure(typ1.pc[cnt],50,r=distance))
# %%
from ipywidgets import interact
# %%
cnt = 40
cb_min, cb_max = 900, 1020
cb_div = 24
interval_of_cf = np.linspace(cb_min, cb_max, cb_div+1)
def tess(cnt):
    distance = grs80.inv(x, y, np.zeros_like(x) + typ1.lon[cnt],
                            np.zeros_like(y) + typ1.lat[cnt])[2]*0.001
    p = cal_pressure(typ1.pc[cnt],50,r=distance)
    cont = plt.contourf(x,y,p,interval_of_cf, cmap="jet")
    plt.colorbar(cont)
# %%
interact(tess, cnt=(0,66, 1))
# %%
r


# %%
# %%
def cal_grad_pressure(center_pressure, r0=50, r=None):
    P_CONSTANT = 1013
    RHOA = 1.22
    f = 7.29 * 10 **-5
    dp = (P_CONSTANT - center_pressure)
    return -0.5*r*f+np.sqrt((r*f *0.5)**2 + dp / RHOA * r0/r * np.exp(r0/r))
# %%
cal_grad_pressure(typ1.pc[cnt],50,r=distance)
# %%
cnt = 40
cb_min, cb_max = 0, 20
cb_div = 24
interval_of_cf = np.linspace(cb_min, cb_max, cb_div+1)
def tess(cnt):
    distance = grs80.inv(x, y, np.zeros_like(x) + typ1.lon[cnt],
                            np.zeros_like(y) + typ1.lat[cnt])[2]*0.001
    ugr = cal_grad_pressure(typ1.pc[cnt],50,r=distance)
    cont = plt.contourf(x,y,ugr,interval_of_cf, cmap="jet")
    plt.colorbar(cont)
# %%

interact(tess, cnt=(0,66, 1))

# %%
# %%
ugr = cal_grad_pressure(typ1.pc[cnt],50,r=distance)
# %%

r = distance
r0 = 50
X = r/r0
Xp = 0.5
k=2.5
c1_inf = 2/3
c1_XP = 1.2
c1 = c1_inf + (c1_XP - c1_inf)*(X/Xp)**(k-1.) * np.exp((1.-1/k)*(1.-(X/Xp)**(k)))
ugr_x = c1 * ugr * np.cos(np.arctan((y-y0) / (x-x0)) + 2*np.pi/3)
ugr_y = c1 * ugr * np.sin(np.arctan((y-y0) / (x-x0)) + 2*np.pi/3)

# %%
x0 = np.zeros_like(x) + typ1.lon[cnt]
y0 = np.zeros_like(x) + typ1.lat[cnt]
# %%
# %%
cnt = 40
cb_min, cb_max = -10, 10
cb_div = 24
interval_of_cf = np.linspace(cb_min, cb_max, cb_div+1)

plt.contourf(x,y,ugr_x,interval_of_cf, cmap="jet")
plt.contourf(x,y,ugr_y,interval_of_cf, cmap="jet")
# %%
# %%

#寒地土木研究所月報
#台風モデルによる波浪の再現計算と経路変更による感度実験

#%%
grs80 = pyproj.Geod(ellps='GRS80')
grs80.inv(151.5, 8, 151.3, 8.05)[0]
#%%
cyclone.lon
cyclone.lat
# %%
grs80 = pyproj.Geod(ellps='GRS80')
_df = cyclone.hourly_data
_df["lon1"] = cyclone.hourly_data["lon"].shift(-1)
_df["lat1"] = cyclone.hourly_data["lat"].shift(-1)
_df.apply(lambda x: grs80.inv(x["lon"],x["lat"],x["lon1"],x["lat1"])[0],axis=1)
# %%
plt.plot(_df["lon"],_df["lat"])
# %%

import matplotlib.pyplot as plt
# %%
