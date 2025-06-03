from etm.TyphoonTrack import TyphoonTrack
import pandas as pd

def set_typhoon_track(typhoon_data:list[dict]):
    typhoons = []
    for typoon in typhoon_data:
        s = TyphoonTrack()
        s.read_bst(typoon)
        typhoons += [s]
    return typhoons

def read_bsttxt(filename:str="./bst_all.txt"):  
    """
    気象庁ベストトラックデータファイルを読み込む
    """
    with open(filename, "r") as f:
        l_strip = [s.replace("\n","") for s in f.readlines()]
    typhoon_data = []
    cnt = 0
    while cnt < len(l_strip):
        header = l_strip[cnt]
        track_num_data = int(header.split()[2])
        cnt += 1
        data = l_strip[cnt:cnt+track_num_data]
        cnt += track_num_data
        typhoon_data += [{"header":header, "data": data}]
    return set_typhoon_track(typhoon_data)

def set_typhoon_track_from_df(df_list:list[dict]):
    typhoon_track = []
    for df in df_list:
        s = TyphoonTrack()
        s.typhoon_number = df["id"]
        s.df = df["df"]
        s.num_data = len(df["df"])
        typhoon_track.append(s)
    return typhoon_track

def read_parquet(path:str):
    df = pd.read_parquet(path)
    df_list = [{"id": id, "df": group} for id, group in df.groupby("international_id")]
    typhoons = set_typhoon_track_from_df(df_list)
    return typhoons

# 回転関数のインポート
from etm.rotation import (
    rotate_coordinates_around_center,
    rotate_typhoon_track,
    rotate_typhoon_track_projected,
    rotate_multiple_tracks,
    plot_original_and_rotated_tracks,
    plot_multiple_rotations,
    calculate_rotation_statistics
)

# 便利な関数のエイリアス
def rotate_track(typhoon_track, center_lon, center_lat, angle_degrees, method='simple'):
    """
    台風経路を回転させる便利関数
    
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
    method : str
        回転方法（'simple' または 'projected'）
        
    Returns:
    --------
    TyphoonTrack
        回転後の台風トラックオブジェクト
    """
    if method == 'projected':
        return rotate_typhoon_track_projected(typhoon_track, center_lon, center_lat, angle_degrees)
    else:
        return rotate_typhoon_track(typhoon_track, center_lon, center_lat, angle_degrees)

