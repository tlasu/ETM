"""
台風経路の回転処理を行うモジュール
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.affinity import rotate as shapely_rotate
import matplotlib.pyplot as plt
from etm.TyphoonTrack import TyphoonTrack


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
        使用する投影座標系（デフォルト: Web Mercator EPSG:3857）
        
    Returns:
    --------
    TyphoonTrack
        回転後の台風トラックオブジェクト
    """
    from shapely.geometry import Point, LineString
    from shapely.affinity import rotate
    
    # 元のDataFrameからGeoDataFrameを作成
    original_df = typhoon_track.df.copy()
    geometry = [Point(lon, lat) for lon, lat in zip(original_df['LON'], original_df['LAT'])]
    gdf = gpd.GeoDataFrame(original_df, geometry=geometry, crs='EPSG:4326')
    
    # 投影座標系に変換
    gdf_projected = gdf.to_crs(crs)
    
    # 回転中心も投影座標系に変換
    center_point = gpd.GeoDataFrame(
        {'id': [1]}, geometry=[Point(center_lon, center_lat)], crs='EPSG:4326'
    ).to_crs(crs)
    center_x, center_y = center_point.geometry.iloc[0].x, center_point.geometry.iloc[0].y
    
    # 各点を回転中心を軸に回転
    rotated_geometries = []
    for geom in gdf_projected.geometry:
        # 回転中心を原点として回転（origin引数を使用）
        rotated = rotate(geom, angle_degrees, origin=(center_x, center_y))
        rotated_geometries.append(rotated)
    
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


def rotate_multiple_tracks(typhoon_tracks, center_lon, center_lat, angle_degrees, method='simple'):
    """
    複数の台風経路を一度に回転させる
    
    Parameters:
    -----------
    typhoon_tracks : list of TyphoonTrack
        台風トラックオブジェクトのリスト
    center_lon : float
        回転中心の経度
    center_lat : float
        回転中心の緯度
    angle_degrees : float
        回転角度（度）
    method : str
        回転方法（'simple' または 'projected'）
        
    Returns:
    --------
    list of TyphoonTrack
        回転後の台風トラックオブジェクトのリスト
    """
    rotated_tracks = []
    
    for track in typhoon_tracks:
        if method == 'projected':
            rotated = rotate_typhoon_track_projected(track, center_lon, center_lat, angle_degrees)
        else:
            rotated = rotate_typhoon_track(track, center_lon, center_lat, angle_degrees)
        rotated_tracks.append(rotated)
    
    return rotated_tracks


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


def plot_multiple_rotations(original_track, center_lon, center_lat, angles, method='simple'):
    """
    複数の角度での回転結果を一度にプロット
    
    Parameters:
    -----------
    original_track : TyphoonTrack
        元の台風トラック
    center_lon : float
        回転中心の経度
    center_lat : float
        回転中心の緯度
    angles : list
        回転角度のリスト
    method : str
        回転方法（'simple' または 'projected'）
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # 元の経路
    ax.plot(original_track.df['LON'], original_track.df['LAT'], 
            'k-', linewidth=3, label='元の経路', alpha=0.8)
    
    # 各回転角度での経路
    colors = plt.cm.rainbow(np.linspace(0, 1, len(angles)))
    for angle, color in zip(angles, colors):
        if method == 'projected':
            rotated = rotate_typhoon_track_projected(original_track, center_lon, center_lat, angle)
        else:
            rotated = rotate_typhoon_track(original_track, center_lon, center_lat, angle)
        
        ax.plot(rotated.df['LON'], rotated.df['LAT'], 
                '-', color=color, label=f'{angle}°', linewidth=2, alpha=0.7)
    
    # 回転中心
    ax.plot(center_lon, center_lat, 'ko', markersize=10, label='回転中心')
    
    ax.set_xlabel('経度')
    ax.set_ylabel('緯度')
    ax.set_title(f'台風経路の様々な角度での回転\n台風: {original_track.name}, 中心: ({center_lon}, {center_lat})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.show()


def calculate_rotation_statistics(original_track, rotated_track):
    """
    回転前後の統計情報を計算
    
    Parameters:
    -----------
    original_track : TyphoonTrack
        元の台風トラック
    rotated_track : TyphoonTrack
        回転後の台風トラック
        
    Returns:
    --------
    dict
        統計情報の辞書
    """
    orig_df = original_track.df
    rot_df = rotated_track.df
    
    stats = {
        '元の経路': {
            '経度範囲': (orig_df['LON'].min(), orig_df['LON'].max()),
            '緯度範囲': (orig_df['LAT'].min(), orig_df['LAT'].max()),
            '経度平均': orig_df['LON'].mean(),
            '緯度平均': orig_df['LAT'].mean(),
            '点数': len(orig_df)
        },
        '回転後の経路': {
            '経度範囲': (rot_df['LON'].min(), rot_df['LON'].max()),
            '緯度範囲': (rot_df['LAT'].min(), rot_df['LAT'].max()),
            '経度平均': rot_df['LON'].mean(),
            '緯度平均': rot_df['LAT'].mean(),
            '点数': len(rot_df)
        }
    }
    
    return stats


if __name__ == "__main__":
    # テスト用のコード
    print("台風経路回転モジュールが正常にロードされました")
    print("主な関数:")
    print("- rotate_coordinates_around_center: 基本的な座標回転")
    print("- rotate_typhoon_track: 台風トラック全体の回転")
    print("- rotate_typhoon_track_projected: 投影座標系を使用した回転")
    print("- rotate_multiple_tracks: 複数の台風トラックの一括回転")
    print("- plot_original_and_rotated_tracks: 結果の可視化")
    print("- plot_multiple_rotations: 複数角度での回転結果の可視化")
    print("- calculate_rotation_statistics: 回転前後の統計情報計算") 