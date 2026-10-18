from dataclasses import dataclass

import numpy as np
import pyproj


@dataclass(frozen=True, eq=False)
class Grid:
    """
    計算格子
    x, y: 1次元の格子座標（経緯度格子では経度・緯度、投影格子では投影座標[m]）
    lons, lats: 各格子点の経度・緯度（形状は (len(y), len(x))）
    crs: x, y の座標系
    """
    x: np.ndarray
    y: np.ndarray
    lons: np.ndarray
    lats: np.ndarray
    crs: str = "EPSG:4326"

    @property
    def shape(self) -> tuple[int, int]:
        return self.lons.shape

    @classmethod
    def from_lonlat(cls, lons, lats) -> "Grid":
        """
        1次元の経度・緯度から経緯度格子を作る
        """
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        array_lons, array_lats = np.meshgrid(lons, lats)
        return cls(lons, lats, array_lons, array_lats)

    @classmethod
    def from_extent(cls, lon_min, lon_max, lat_min, lat_max, resolution) -> "Grid":
        """
        範囲と格子間隔[度]から経緯度格子を作る（両端を含む）
        """
        nlon = int(round((lon_max - lon_min) / resolution)) + 1
        nlat = int(round((lat_max - lat_min) / resolution)) + 1
        return cls.from_lonlat(np.linspace(lon_min, lon_max, nlon), np.linspace(lat_min, lat_max, nlat))

    @classmethod
    def from_projected(cls, x, y, crs) -> "Grid":
        """
        投影座標系の1次元座標 x, y から格子を作る（例: crs="EPSG:6677" 平面直角座標系第IX系）
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        array_lons, array_lats = transformer.transform(*np.meshgrid(x, y))
        return cls(x, y, array_lons, array_lats, crs)
