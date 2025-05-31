#%%
from etm import util
from etm.TyphoonTrack import interpolate_by_cumulative_distance

# %%
typhoons = util.read_parquet("../jma-bst/typhoon_all.parquet")
typhoons[0].set_direction()
typ_df = typhoons[0].df
# %%
interpolated_df = interpolate_by_cumulative_distance(
    data=typhoons[0],  # TyphoonTrackインスタンス
    constant_velocity=14.0,  # 14 m/s
    time_interval_hours=1.0  # 1時間間隔
)
interpolated_df
# %%
