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
        s = TyphoonTrack.TyphoonTrack()
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

