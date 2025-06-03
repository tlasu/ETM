# %%
"""
台風経路回転の簡潔な使用例
"""

import etm.util as util
import matplotlib.pyplot as plt

# %% 台風データの読み込み
typhoons = util.read_bsttxt("../jma-bst/bst_all.txt")
print(f"読み込んだ台風数: {len(typhoons)}")

# %% 特定の台風を選択
selected_typhoon = typhoons[10]  # 例として11番目の台風を選択
print(f"選択した台風: {selected_typhoon.name}")
print(f"シーズン: {selected_typhoon.season}")
print(f"経路点数: {len(selected_typhoon.df)}")

# %% 基本的な回転
# 日本付近を中心として45度回転
center_lon = 138.0  # 日本の中央付近の経度
center_lat = 36.0   # 日本の中央付近の緯度
angle = 45          # 回転角度（度）

# 簡単な方法で回転
rotated_typhoon = util.rotate_track(
    selected_typhoon, 
    center_lon, 
    center_lat, 
    angle,
    method='simple'
)

print(f"回転完了: {rotated_typhoon.name}")

# %% 結果を可視化
util.plot_original_and_rotated_tracks(
    selected_typhoon,
    rotated_typhoon,
    center_lon,
    center_lat,
    angle
)

# %% 複数角度での回転可視化
angles = [0, 30, 60, 90, 120, 150, 180]
util.plot_multiple_rotations(
    selected_typhoon,
    center_lon,
    center_lat,
    angles,
    method='simple'
)

# %% 投影座標系を使用したより正確な回転
rotated_projected = util.rotate_track(
    selected_typhoon,
    center_lon,
    center_lat,
    angle,
    method='projected'
)

# %% 単純回転と投影座標系回転の比較
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# 単純回転
ax1.plot(selected_typhoon.df['LON'], selected_typhoon.df['LAT'], 'b-o', 
         label='元の経路', markersize=4)
ax1.plot(rotated_typhoon.df['LON'], rotated_typhoon.df['LAT'], 'r-s', 
         label='単純回転', markersize=4)
ax1.plot(center_lon, center_lat, 'ko', markersize=8, label='回転中心')
ax1.set_title(f'単純回転 ({angle}°)')
ax1.set_xlabel('経度')
ax1.set_ylabel('緯度')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 投影座標系回転
ax2.plot(selected_typhoon.df['LON'], selected_typhoon.df['LAT'], 'b-o', 
         label='元の経路', markersize=4)
ax2.plot(rotated_projected.df['LON'], rotated_projected.df['LAT'], 'g-^', 
         label='投影座標系回転', markersize=4)
ax2.plot(center_lon, center_lat, 'ko', markersize=8, label='回転中心')
ax2.set_title(f'投影座標系回転 ({angle}°)')
ax2.set_xlabel('経度')
ax2.set_ylabel('緯度')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.tight_layout()
plt.show()

# %% 複数の台風を一度に回転
# 例として最初の5つの台風を回転
sample_typhoons = typhoons[:5]
rotated_multiple = util.rotate_multiple_tracks(
    sample_typhoons,
    center_lon,
    center_lat,
    60,  # 60度回転
    method='simple'
)

# 結果の確認
print(f"回転前の台風数: {len(sample_typhoons)}")
print(f"回転後の台風数: {len(rotated_multiple)}")

# %% 統計情報の比較
stats = util.calculate_rotation_statistics(selected_typhoon, rotated_typhoon)
print("\n=== 回転前後の統計情報 ===")
for key, value in stats.items():
    print(f"\n{key}:")
    for stat_name, stat_value in value.items():
        if isinstance(stat_value, tuple):
            print(f"  {stat_name}: {stat_value[0]:.2f} - {stat_value[1]:.2f}")
        else:
            print(f"  {stat_name}: {stat_value:.2f}")

print("\n台風経路回転の使用例が完了しました！")
print("主な機能:")
print("1. util.rotate_track() - シンプルな回転関数")
print("2. util.rotate_multiple_tracks() - 複数台風の一括回転")
print("3. util.plot_original_and_rotated_tracks() - 1対1比較の可視化")
print("4. util.plot_multiple_rotations() - 複数角度の可視化")
print("5. util.calculate_rotation_statistics() - 統計情報の比較") 