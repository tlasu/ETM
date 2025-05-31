# %%
import etm.util as util
import geopandas as gpd
import pandas as pd
# %%
typhoons = util.read_bsttxt("../jma-bst/bst_all.txt")
typhoons[0].properties()
# %%
gdf = gpd.GeoDataFrame(pd.concat([_.gdf for _ in typhoons]))
# %%
import keplergl as kp
# %%
m = kp.KeplerGl()
m.add_data(gdf, "typhoons")
m
