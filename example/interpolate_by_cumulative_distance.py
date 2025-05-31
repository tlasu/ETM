#%%
from etm import util
from etm.TyphoonTrack import interpolate_by_cumulative_distance, add_cumulative_distance, resample_by_time_interval, calculate_moving_velocity
import matplotlib.pyplot as plt
import matplotlib_fontja
# %%

# %%
typhoons = util.read_parquet("../jma-bst/typhoon_all.parquet")
typhoons[0].set_direction()
typ_df = typhoons[0].df

resampled_1h = resample_by_time_interval(typ_df, time_interval_hours=1.0)
print(f"1時間間隔リサンプル: {len(typ_df)} → {len(resampled_1h)} 点")

# 3時間間隔でリサンプル
resampled_3h = resample_by_time_interval(typ_df, time_interval_hours=3.0)
print(f"3時間間隔リサンプル: {len(typ_df)} → {len(resampled_3h)} 点")

# 6時間間隔でリサンプル
resampled_6h = resample_by_time_interval(typ_df, time_interval_hours=6.0)
print(f"6時間間隔リサンプル: {len(typ_df)} → {len(resampled_6h)} 点")

print("\n1時間間隔リサンプル結果（最初の5点）:")
print(resampled_1h[["ISO_TIME", "LAT", "LON", "PRES", "WIND"]].head())

# 各リサンプル結果をプロット
plt.figure(figsize=(15, 10))
# メインプロット
plt.subplot(2, 2, (1, 2))
plt.plot(typ_df["LON"], typ_df["LAT"], "o-", label="元データ", markersize=6, linewidth=2, alpha=0.8)
plt.plot(resampled_1h["LON"], resampled_1h["LAT"], "x-", label="1時間間隔", markersize=4, alpha=0.7)
plt.plot(resampled_3h["LON"], resampled_3h["LAT"], "s-", label="3時間間隔", markersize=4, alpha=0.7)
plt.plot(resampled_6h["LON"], resampled_6h["LAT"], "^-", label="6時間間隔", markersize=4, alpha=0.7)
plt.xlabel("経度")
plt.ylabel("緯度")
plt.title("時間間隔による台風トラックのリサンプル比較")
plt.legend()
plt.grid(True, alpha=0.3)

# 気圧の時系列
plt.subplot(2, 2, 3)
plt.plot(typ_df["ISO_TIME"], typ_df["PRES"], "o-", label="元データ", markersize=4)
plt.plot(resampled_1h["ISO_TIME"], resampled_1h["PRES"], "x-", label="1時間間隔", markersize=3)
plt.plot(resampled_3h["ISO_TIME"], resampled_3h["PRES"], "s-", label="3時間間隔", markersize=3)
plt.xlabel("時刻")
plt.ylabel("気圧 (hPa)")
plt.title("気圧の時系列変化")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# 風速の時系列
plt.subplot(2, 2, 4)
plt.plot(typ_df["ISO_TIME"], typ_df["WIND"], "o-", label="元データ", markersize=4)
plt.plot(resampled_1h["ISO_TIME"], resampled_1h["WIND"], "x-", label="1時間間隔", markersize=3)
plt.plot(resampled_3h["ISO_TIME"], resampled_3h["WIND"], "s-", label="3時間間隔", markersize=3)
plt.xlabel("時刻")
plt.ylabel("風速 (kt)")
plt.title("風速の時系列変化")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %%
