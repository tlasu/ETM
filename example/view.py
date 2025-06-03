# %%
import etm.util as util
import geopandas as gpd
import pandas as pd
import contextily as ctx

# %%
typhoons = util.read_bsttxt("../jma-bst/bst_all.txt")
typhoons[0].properties()
# %%
gdf = gpd.GeoDataFrame(pd.concat([_.gdf for _ in typhoons]))
# %%
import keplergl as kp
from shapely.geometry import Point
import numpy as np



def extract_linestrings_within_distance(gdf, lat, lon, distance_km=100):
    """
    指定した緯度経度から指定距離以内と交差するLinestringを抽出する
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        対象のGeoDataFrame
    lat : float
        緯度
    lon : float
        経度
    distance_km : float
        距離（キロメートル）
    
    Returns:
    --------
    tuple
        (result_gdf, buffer_gdf): 条件に合うLinestringのGeoDataFrameとバッファ円のGeoDataFrame
    """
    
    # 指定した緯度経度からPointを作成
    target_point = Point(lon, lat)
    
    # GeoDataFrameのCRSを確認し、必要に応じて設定
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    
    # 距離計算のため、適切な投影座標系に変換
    # 日本周辺であればJGD2011 / Japan Plane Rectangular CS IX (EPSG:6677)
    # または世界測地系UTM座標系を使用
    gdf_projected = gdf.to_crs('EPSG:3857')  # Web Mercator
    target_point_gdf = gpd.GeoDataFrame([1], geometry=[target_point], crs='EPSG:4326')
    target_point_projected = target_point_gdf.to_crs('EPSG:3857').geometry[0]
    
    # 指定距離（km）をメートルに変換してバッファを作成
    buffer_distance_m = distance_km * 1000
    buffer_geometry = target_point_projected.buffer(buffer_distance_m)
    
    # バッファ円のGeoDataFrameを作成（WGS84に戻す）
    buffer_gdf_projected = gpd.GeoDataFrame(
        {'radius_km': [distance_km], 'center_lat': [lat], 'center_lon': [lon]}, 
        geometry=[buffer_geometry], 
        crs='EPSG:3857'
    )
    buffer_gdf = buffer_gdf_projected.to_crs('EPSG:4326')
    
    # バッファと交差するLinestringを抽出
    intersects_mask = gdf_projected.geometry.intersects(buffer_geometry)
    result_gdf = gdf[intersects_mask].copy()
    
    # 距離も計算して追加
    distances_m = gdf_projected[intersects_mask].geometry.distance(target_point_projected)
    result_gdf['distance_km'] = distances_m / 1000
    
    print(f"指定地点（緯度: {lat}, 経度: {lon}）から{distance_km}km以内と交差するLinestring: {len(result_gdf)}件")
    
    return result_gdf, buffer_gdf

# %%
# 使用例: 東京（35.6762, 139.6503）から100km以内の台風経路を抽出
tokyo_lat = 35.6762
tokyo_lon = 139.6503

nearby_typhoons, buffer_circle = extract_linestrings_within_distance(gdf, tokyo_lat, tokyo_lon, 100)

# 結果を表示
print(f"抽出された台風経路の数: {len(nearby_typhoons)}")
if len(nearby_typhoons) > 0:
    print(f"最も近い台風までの距離: {nearby_typhoons['distance_km'].min():.2f}km")
    print(f"最も遠い台風までの距離: {nearby_typhoons['distance_km'].max():.2f}km")

# %%
# 結果をKeplerGLで可視化
if len(nearby_typhoons) > 0:
    m2 = kp.KeplerGl()
    
    # 抽出された台風経路を追加
    m2.add_data(nearby_typhoons, "nearby_typhoons")
    
    # 基準点（東京）も追加
    tokyo_point = gpd.GeoDataFrame(
        {'name': ['Tokyo']}, 
        geometry=[Point(tokyo_lon, tokyo_lat)], 
        crs='EPSG:4326'
    )
    m2.add_data(tokyo_point, "reference_point")
    
    # 100km円のバッファも追加
    m2.add_data(buffer_circle, "search_area")
    
    m2

# %%
# matplotlibでも可視化
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import contextily as ctx

def plot_typhoons_matplotlib(nearby_typhoons, buffer_circle, tokyo_point, tokyo_lat, tokyo_lon, distance_km=100):
    """
    matplotlibを使って台風経路と検索範囲を可視化
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 抽出された台風経路をプロット
    if len(nearby_typhoons) > 0:
        nearby_typhoons.plot(ax=ax, color='red', alpha=0.7, linewidth=1.5, label=f'台風経路 ({len(nearby_typhoons)}件)')
    
    # バッファ円をプロット
    buffer_circle.plot(ax=ax, facecolor='blue', alpha=0.2, edgecolor='blue', linewidth=2, label=f'{distance_km}km 検索範囲')
    
    # 基準点をプロット
    tokyo_point.plot(ax=ax, color='black', markersize=100, marker='*', label='基準点 (東京)', zorder=5)
    
    # 日本周辺の範囲に制限
    ax.set_xlim(123, 146)  # 経度: 西端から東端
    ax.set_ylim(24, 46)    # 緯度: 沖縄から北海道
    
    # 軸の設定
    ax.set_xlabel('経度 (Longitude)', fontsize=12)
    ax.set_ylabel('緯度 (Latitude)', fontsize=12)
    ax.set_title(f'東京から{distance_km}km以内の台風経路（日本周辺）', fontsize=14, fontweight='bold')
    
    # 凡例を追加
    ax.legend(loc='upper right')
    
    # グリッドを追加
    ax.grid(True, alpha=0.3)
    
    # アスペクト比を調整
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

# matplotlibで可視化実行
if len(nearby_typhoons) > 0:
    plot_typhoons_matplotlib(nearby_typhoons, buffer_circle, tokyo_point, tokyo_lat, tokyo_lon, 100)

# %%
# より詳細なmatplotlib可視化（背景地図付き）
def plot_typhoons_with_basemap(nearby_typhoons, buffer_circle, tokyo_point, tokyo_lat, tokyo_lon, distance_km=100):
    """
    背景地図付きでmatplotlibを使って台風経路と検索範囲を可視化
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # Web Mercatorに変換（背景地図用）
    nearby_typhoons_web = nearby_typhoons.to_crs('EPSG:3857')
    buffer_circle_web = buffer_circle.to_crs('EPSG:3857')
    tokyo_point_web = tokyo_point.to_crs('EPSG:3857')
    
    # 抽出された台風経路をプロット
    if len(nearby_typhoons_web) > 0:
        nearby_typhoons_web.plot(ax=ax, color='red', alpha=0.8, linewidth=2, label=f'台風経路 ({len(nearby_typhoons_web)}件)')
    
    # バッファ円をプロット
    buffer_circle_web.plot(ax=ax, facecolor='blue', alpha=0.2, edgecolor='blue', linewidth=3, label=f'{distance_km}km 検索範囲')
    
    # 基準点をプロット
    tokyo_point_web.plot(ax=ax, color='yellow', markersize=150, marker='*', edgecolor='black', linewidth=2, label='基準点 (東京)', zorder=5)
    
    # 日本周辺の範囲をWeb Mercator座標で設定
    # 緯度経度の範囲をWeb Mercator座標に変換
    import pyproj
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    
    # 日本周辺の範囲 (緯度24-46度、経度123-146度)
    west_lon, south_lat = 123, 24
    east_lon, north_lat = 146, 46
    
    # 座標変換
    west_x, south_y = transformer.transform(west_lon, south_lat)
    east_x, north_y = transformer.transform(east_lon, north_lat)
    
    ax.set_xlim(west_x, east_x)
    ax.set_ylim(south_y, north_y)
    
    # 背景地図を追加
    try:
        ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
        ax.set_title(f'東京から{distance_km}km以内の台風経路（日本周辺・背景地図付き）', fontsize=16, fontweight='bold')
    except:
        ax.set_title(f'東京から{distance_km}km以内の台風経路（日本周辺）', fontsize=16, fontweight='bold')
        print("背景地図の読み込みに失敗しました。インターネット接続を確認してください。")
    
    # 軸ラベルを設定
    ax.set_xlabel('東西方向 (m)', fontsize=12)
    ax.set_ylabel('南北方向 (m)', fontsize=12)
    
    # 凡例を追加
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# 背景地図付きで可視化実行
if len(nearby_typhoons) > 0:
    plot_typhoons_with_basemap(nearby_typhoons, buffer_circle, tokyo_point, tokyo_lat, tokyo_lon, 100)

# %%
