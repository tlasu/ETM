#%%
import datetime as dt

import numpy as np
import pandas as pd
from pyproj import Transformer
import pyproj


def set_typhoon_track(typhoon_data):
    typhoons = []
    for typoon in typhoon_data:
        s = TyphoonTrack()
        s.read_bst(typoon)
        typhoons += [s]
    return typhoons


def read_bsttxt(filename:str="./bst_all.txt"):  
    with open(filename, "r") as f:
        l_strip = [s.rstrip() for s in f.readlines()]

    typhoon_data = []
    cnt = 0
    while cnt < len(l_strip):
        header = l_strip[cnt].split()
        track_num_data = int(header[2])
        cnt += 1
        data = l_strip[cnt:cnt+track_num_data]
        cnt += track_num_data
        typhoon_data += [{"header":header, "data": data}]
    return set_typhoon_track(typhoon_data)

class TyphoonTrack:
    def __init__(self, data=None) -> None:
        self.typhoon_number = None
        self.num_data = None
        self.tropical_cyclone_number = None
        self.time = []
        self.rank = None
        self.lat = []
        self.lon = []
        self.pc = []
        pass

    def read_bst(self, data=None):
        if type(data) is dict:
            header = data["header"]
            self.typhoon_number = header[1]
            self.num_data = int(header[2])
            self.tropical_cyclone_number = header[3]
            if len(header) > 7:
                self.typhoon_name = header[7]
            typ_data = data['data']

            self.time = [dt.datetime.strptime(_l[0:8], "%y%m%d%H") for _l  in typ_data]
            self.rank = [int(_l[13]) for _l  in typ_data]
            self.lat = [float(_l[15:18])*0.1 for _l  in typ_data]
            self.lon = [float(_l[19:23])*0.1 for _l  in typ_data]
            self.pc = [float(_l[24:28]) for _l  in typ_data]
            self.org_data = pd.DataFrame({'pc':self.pc,'lon':self.lon,'lat':self.lat}, index=self.time)
            self.hourly_data = self.org_data.resample('h').interpolate('time')
        pass
    def set_moving_velocity(self,df):
        #座標変換：wgs84からjgd2011に変換する
        epsg4326_to_epsg6677 = Transformer.from_crs("epsg:4326", "epsg:6677",always_xy=True)
        df["x"], df["y"] = epsg4326_to_epsg6677.transform(df.lon,df.lat)
        df["gap_distance"] = np.sqrt(df["x"].diff()**2 + df["y"].diff()**2)
        df["moving_velocity"] = df.gap_distance / (df.index.to_series().diff() / pd.Timedelta(seconds=1))
        df["moving_velocity"].fillna(0,inplace=True)
        return df

    def set_direction(self,df):
        grs80 = pyproj.Geod(ellps='GRS80')
        df["lon1"] = df["lon"].shift(-1)
        df["lat1"] = df["lat"].shift(-1)
        df["direction"] = df.apply(lambda x: grs80.inv(x["lon"],x["lat"],x["lon1"],x["lat1"])[0],axis=1)
        return df.drop(['lon1', 'lat1'], axis=1)


# %%
