# %%
import etm.util as util
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.affinity import rotate as shapely_rotate
import pyproj

# %%
typhoons = util.read_bsttxt("../jma-bst/bst_all.txt")
# %%
gdf = gpd.GeoDataFrame(pd.concat([_.gdf for _ in typhoons]))

# %%
def rotate_coordinates_around_center(lon, lat, center_lon, center_lat, angle_degrees):
    """
    指定された中心点（緯度経度）を軸に座標を回転させる
    
    Parameters:
    -----------
    lon : float or array-like
        経度
    lat : float or array-like
        緯度
    center_lon : float
        回転中心の経度
    center_lat : float
        回転中心の緯度
    angle_degrees : float
        回転角度（度）。正の値で反時計回り
        
    Returns:
    --------
    tuple
        (回転後の経度, 回転後の緯度)
    """
    # 角度をラジアンに変換
    angle_rad = np.radians(angle_degrees)
    
    # 中心点を原点に移動
    lon_centered = np.array(lon) - center_lon
    lat_centered = np.array(lat) - center_lat
    
    # 2D回転行列を適用
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    lon_rotated = lon_centered * cos_angle - lat_centered * sin_angle
    lat_rotated = lon_centered * sin_angle + lat_centered * cos_angle
    
    # 中心点を元の位置に戻す
    lon_final = lon_rotated + center_lon
    lat_final = lat_rotated + center_lat
    
    return lon_final, lat_final

def rotate_typhoon_track(typhoon_track, center_lon, center_lat, angle_degrees):
    """
    台風経路を指定された中心点周りに回転させる
    
    Parameters:
    -----------
    typhoon_track : TyphoonTrack
        台風トラックオブジェクト
    center_lon : float
        回転中心の経度
    center_lat : float
        回転中心の緯度
    angle_degrees : float
        回転角度（度）。正の値で反時計回り
        
    Returns:
    --------
    TyphoonTrack
        回転後の台風トラックオブジェクト
    """
    from etm.TyphoonTrack import TyphoonTrack
    
    # 新しいTyphoonTrackオブジェクトを作成
    rotated_track = TyphoonTrack()
    
    # 元のDataFrameをコピー
    original_df = typhoon_track.df.copy()
    
    # 座標を回転
    rotated_lon, rotated_lat = rotate_coordinates_around_center(
        original_df['LON'], 
        original_df['LAT'], 
        center_lon, 
        center_lat, 
        angle_degrees
    )
    
    # 新しいDataFrameを作成
    rotated_df = original_df.copy()
    rotated_df['LON'] = rotated_lon
    rotated_df['LAT'] = rotated_lat
    
    # 新しいトラックオブジェクトにデータを設定
    rotated_track.df = rotated_df
    
    return rotated_track

def plot_original_and_rotated_tracks(original_track, rotated_track, center_lon, center_lat, angle_degrees):
    """
    元の経路と回転後の経路を比較してプロット
    
    Parameters:
    -----------
    original_track : TyphoonTrack
        元の台風トラック
    rotated_track : TyphoonTrack
        回転後の台風トラック
    center_lon : float
        回転中心の経度
    center_lat : float
        回転中心の緯度
    angle_degrees : float
        回転角度（度）
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 元の経路をプロット
    ax.plot(original_track.df['LON'], original_track.df['LAT'], 
            'b-o', label=f'元の経路: {original_track.name}', markersize=4)
    
    # 回転後の経路をプロット
    ax.plot(rotated_track.df['LON'], rotated_track.df['LAT'], 
            'r-s', label=f'回転後の経路 ({angle_degrees}°)', markersize=4)
    
    # 回転中心をプロット
    ax.plot(center_lon, center_lat, 'ko', markersize=8, label='回転中心')
    
    # 開始点と終了点をマーク
    ax.plot(original_track.df['LON'].iloc[0], original_track.df['LAT'].iloc[0], 
            'bo', markersize=8, label='元の開始点')
    ax.plot(original_track.df['LON'].iloc[-1], original_track.df['LAT'].iloc[-1], 
            'bs', markersize=8, label='元の終了点')
    
    ax.plot(rotated_track.df['LON'].iloc[0], rotated_track.df['LAT'].iloc[0], 
            'ro', markersize=8, label='回転後の開始点')
    ax.plot(rotated_track.df['LON'].iloc[-1], rotated_track.df['LAT'].iloc[-1], 
            'rs', markersize=8, label='回転後の終了点')
    
    ax.set_xlabel('経度')
    ax.set_ylabel('緯度')
    ax.set_title(f'台風経路の回転比較\n中心: ({center_lon:.1f}, {center_lat:.1f}), 角度: {angle_degrees}°')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.show()

# %% 実際の使用例
# 台風データを選択（例：最初の台風）
selected_typhoon = typhoons[0]
print(f"選択した台風: {selected_typhoon.name}")
print(f"経路点数: {len(selected_typhoon.df)}")

# 回転中心を設定（例：日本の中心付近）
center_longitude = 138.0  # 経度
center_latitude = 36.0    # 緯度

# 回転角度を設定（例：45度反時計回り）
rotation_angle = 45

# 台風経路を回転
rotated_typhoon = rotate_typhoon_track(
    selected_typhoon, 
    center_longitude, 
    center_latitude, 
    rotation_angle
)

print(f"回転後の台風名: {rotated_typhoon.name}")

# 結果をプロット
plot_original_and_rotated_tracks(
    selected_typhoon, 
    rotated_typhoon, 
    center_longitude, 
    center_latitude, 
    rotation_angle
)

# %% 複数の角度で回転を試してみる
angles = [0, 30, 60, 90, 120, 150, 180]
rotated_typhoons = []

for angle in angles:
    rotated = rotate_typhoon_track(selected_typhoon, center_longitude, center_latitude, angle)
    rotated_typhoons.append((angle, rotated))

# 複数の回転結果をプロット
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# 元の経路
ax.plot(selected_typhoon.df['LON'], selected_typhoon.df['LAT'], 
        'k-', linewidth=3, label='元の経路', alpha=0.8)

# 各回転角度での経路
colors = plt.cm.rainbow(np.linspace(0, 1, len(angles)))
for (angle, rotated), color in zip(rotated_typhoons, colors):
    ax.plot(rotated.df['LON'], rotated.df['LAT'], 
            '-', color=color, label=f'{angle}°', linewidth=2, alpha=0.7)

# 回転中心
ax.plot(center_longitude, center_latitude, 'ko', markersize=10, label='回転中心')

ax.set_xlabel('経度')
ax.set_ylabel('緯度')
ax.set_title(f'台風経路の様々な角度での回転\n台風: {selected_typhoon.name}, 中心: ({center_longitude}, {center_latitude})')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.tight_layout()
plt.show()

# %% GeoPandasを使用したより高度な回転（投影座標系を使用）
def rotate_typhoon_track_projected(typhoon_track, center_lon, center_lat, angle_degrees, crs='EPSG:3857'):
    """
    投影座標系を使用してより正確な回転を行う
    
    Parameters:
    -----------
    typhoon_track : TyphoonTrack
        台風トラックオブジェクト
    center_lon : float
        回転中心の経度
    center_lat : float
        回転中心の緯度
    angle_degrees : float
        回転角度（度）
    crs : str
        使用する投影座標系（デフォルト: Web Mercator）
        
    Returns:
    --------
    TyphoonTrack
        回転後の台風トラックオブジェクト
    """
    from etm.TyphoonTrack import TyphoonTrack
    
    # 元のDataFrameからGeoDataFrameを作成
    original_df = typhoon_track.df.copy()
    geometry = [Point(lon, lat) for lon, lat in zip(original_df['LON'], original_df['LAT'])]
    gdf = gpd.GeoDataFrame(original_df, geometry=geometry, crs='EPSG:4326')
    
    # 投影座標系に変換
    gdf_projected = gdf.to_crs(crs)
    
    # 回転中心も投影座標系に変換
    center_point = gpd.GeoDataFrame(
        [1], geometry=[Point(center_lon, center_lat)], crs='EPSG:4326'
    ).to_crs(crs)
    center_x, center_y = center_point.geometry.iloc[0].x, center_point.geometry.iloc[0].y
    
    # 各点を回転中心を軸に回転
    rotated_geometries = []
    for geom in gdf_projected.geometry:
        # 回転中心を原点にするために平行移動
        translated = Point(geom.x - center_x, geom.y - center_y)
        
        # 回転（Shapely rotateは原点中心の回転）
        rotated = shapely_rotate(translated, angle_degrees)
        
        # 元の位置に戻す
        final_point = Point(rotated.x + center_x, rotated.y + center_y)
        rotated_geometries.append(final_point)
    
    # 回転後のGeoDataFrameを作成
    gdf_rotated = gdf_projected.copy()
    gdf_rotated.geometry = rotated_geometries
    
    # 地理座標系に戻す
    gdf_final = gdf_rotated.to_crs('EPSG:4326')
    
    # 新しいTyphoonTrackオブジェクトを作成
    rotated_track = TyphoonTrack()
    rotated_df = original_df.copy()
    rotated_df['LON'] = [geom.x for geom in gdf_final.geometry]
    rotated_df['LAT'] = [geom.y for geom in gdf_final.geometry]
    
    rotated_track.df = rotated_df
    
    return rotated_track

# %% 投影座標系を使用した回転の例
rotated_typhoon_projected = rotate_typhoon_track_projected(
    selected_typhoon, 
    center_longitude, 
    center_latitude, 
    rotation_angle
)

# 単純回転と投影座標系回転の比較
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# 単純回転
ax1.plot(selected_typhoon.df['LON'], selected_typhoon.df['LAT'], 'b-o', label='元の経路', markersize=4)
ax1.plot(rotated_typhoon.df['LON'], rotated_typhoon.df['LAT'], 'r-s', label='単純回転', markersize=4)
ax1.plot(center_longitude, center_latitude, 'ko', markersize=8, label='回転中心')
ax1.set_title(f'単純回転 ({rotation_angle}°)')
ax1.set_xlabel('経度')
ax1.set_ylabel('緯度')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 投影座標系回転
ax2.plot(selected_typhoon.df['LON'], selected_typhoon.df['LAT'], 'b-o', label='元の経路', markersize=4)
ax2.plot(rotated_typhoon_projected.df['LON'], rotated_typhoon_projected.df['LAT'], 'g-^', label='投影座標系回転', markersize=4)
ax2.plot(center_longitude, center_latitude, 'ko', markersize=8, label='回転中心')
ax2.set_title(f'投影座標系回転 ({rotation_angle}°)')
ax2.set_xlabel('経度')
ax2.set_ylabel('緯度')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.tight_layout()
plt.show()

print("台風経路回転関数の実装が完了しました！")
print("主な機能:")
print("1. rotate_coordinates_around_center: 基本的な座標回転")
print("2. rotate_typhoon_track: 台風トラック全体の回転")
print("3. rotate_typhoon_track_projected: 投影座標系を使用したより正確な回転")
print("4. plot_original_and_rotated_tracks: 結果の可視化")

# %%
rotated_typhoon_projected.df
# %%
