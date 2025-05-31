from shapely.geometry import LineString
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
import pyproj

def parse_header(line: str):
    """
    ヘッダー行をパースして台風情報の辞書を返す
    """
    data = {
        "international_id": line[6:10],
        "ndata": int(line[12:15]),
        "tropical_cyclone_id": line[16:20],
        "replicate_international_id": line[21:25],
        "flag": line[26],
        "difference": line[28],
        "name": line[30:50].strip(),
        "date": line[64:72],
    }
    return data

def parse_dataline(line: str):
    """
    データ行をパースして1レコード分の辞書を返す
    """
    data = {
        "ISO_TIME": parse_time(line[0:8]),
        "SEASON": parse_time(line[0:8]).year,
        "indicator": line[9:12],
        "grade": line[13:14],
        "LAT": float(line[15:18]) * 0.1 if line[15:18].strip() else None,
        "LON": float(line[19:23]) * 0.1 if line[19:23].strip() else None,
        "PRES": int(line[24:28]) if line[24:28].strip() else None,
        "WIND": int(line[33:36]) if line[33:36].strip() else None,
        "r50_dir": line[41],
        "r50_long": int(line[42:46]) if line[42:46].strip() else None,
        "r50_short": int(line[47:51]) if line[47:51].strip() else None,
        "r30_dir": line[53],
        "r30_long": int(line[54:58]) if line[54:58].strip() else None,
        "r30_short": int(line[59:63]) if line[59:63].strip() else None,
        "landfall": line[71:72],
    }
    return data

def parse_datalines(lines:list):
    """
    データ行リストをパースしてDataFrameを返す
    """
    data = []
    for line in lines:
        data.append(parse_dataline(line))
    return pd.DataFrame(data)

def parse_track(lines:list, header:list, header_id:int|None=None):
    """
    台風トラックデータ（ヘッダー＋データ行）をパースしてDataFrameを返す
    """
    if header_id is not None:
        header_data = parse_header(header)
        df = parse_datalines(lines[header_id+1:header_id+header_data["ndata"]+1])
    else:
        header_data = parse_header(header)
        df = parse_datalines(lines)
    df["international_id"] = header_data["international_id"]
    df["tropical_cyclone_id"] = header_data["tropical_cyclone_id"]
    df["NAME"] = header_data["name"]
    # カラム順を指定
    columns = [
        "international_id", "tropical_cyclone_id", "NAME", "SEASON",
        "ISO_TIME", "indicator", "grade", "LAT", "LON", "PRES", "WIND",
        "r50_dir", "r50_long", "r50_short", "r30_dir", "r30_long", "r30_short", "landfall"
    ]
    # 存在するカラムのみ抽出
    columns = [col for col in columns if col in df.columns]
    df = df[columns]
    return df

def parse_time(time_str:str):
    """
    時間データをパースする
    未来（例: 1951年のデータが2051年になる場合）を補正
    """
    import datetime
    dt = datetime.datetime.strptime(time_str, "%y%m%d%H")
    # 未来（例: 1951年のデータが2051年になる場合）を補正
    if dt.year > 2050:
        dt = dt.replace(year=dt.year - 100)
    return dt

class TyphoonTrack:
    """
    台風1個分のトラックデータを管理するクラス
    """
    def __init__(self, data=None) -> None:
        """
        TyphoonTrackインスタンスを初期化
        """
        self.typhoon_number:str|None = None
        self.num_data:int|None = None
        self.tropical_cyclone_number:str|None = None
        self.name:str|None = None
        self.season:int|None = None
        self._df:pd.DataFrame|None = None
        self.gdf: gpd.GeoDataFrame | None = None
        pass
    
    @property
    def df(self):
        """
        DataFrameを取得する
        """
        return self._df
    
    @df.setter
    def df(self, value):
        """
        DataFrameを設定し、設定後に自動的にLinestringを更新する
        """
        self._df = value
        if self._df is not None and not self._df.empty:
            try:
                self._set_linestring()
                self._set_properties()
            except (AssertionError, ValueError) as e:
                print(f"警告: LineString設定中にエラーが発生しました: {e}")
                self.gdf = None
    
    def _set_properties(self):
        _df = self.df
        self.typhoon_number = _df["international_id"].iloc[0]
        self.season = int(_df["SEASON"].iloc[0])
        self.tropical_cyclone_number = _df["tropical_cyclone_id"].iloc[0]
        self.name = _df["NAME"].iloc[0]
        self.num_data = len(_df)

    def _set_linestring(self):
        _df = self.df
        assert not _df["LON"].isna().any(), "LONにNaNが含まれています"
        assert not _df["LAT"].isna().any(), "LATにNaNが含まれています"
        self.gdf = gpd.GeoDataFrame(geometry = [LineString(_df[["LON","LAT"]])])
        pass

    def properties(self):
        """
        台風情報のプロパティを返す
        """
        return {
            "international_id": self.typhoon_number,
            "tropical_cyclone_id": self.tropical_cyclone_number,
            "name": self.name,
            "num_data": self.num_data,
            "season": self.season,
        }

    def read_bst(self, data=None):
        """
        辞書形式のデータから台風情報を読み込む
        """
        if type(data) is dict:
            header = data["header"]
            header_data = parse_header(header)
            self.typhoon_number = header_data["international_id"]
            self.num_data = header_data["ndata"]
            self.tropical_cyclone_number = header_data["tropical_cyclone_id"]
            self.typhoon_name = header_data["name"]
            typ_data = data["data"]
            self.df = parse_track(typ_data, header=header)  # セッターが呼ばれてset_linestring()が自動実行される
        pass

    def set_moving_velocity(self):
        """
        実際のトラックから移動速度を計算してDataFrameに移動速度（m/s）を計算して追加する
        """
        result_df = calculate_moving_velocity(self)
        self.df = result_df

    def set_direction(self):
        """
        DataFrameに進行方向（度）を計算して追加する
        """
        grs80 = pyproj.Geod(ellps='GRS80')
        df = self.df
        df["LON1"] = df["LON"].shift(-1)
        df["LAT1"] = df["LAT"].shift(-1)
        df["direction"] = df.apply(lambda x: grs80.inv(x["LON"],x["LAT"],x["LON1"],x["LAT1"])[0],axis=1)
        result_df = df.drop(['LON1', 'LAT1'], axis=1)
        # DataFrameを更新したので、set_linestring()が自動実行される
        self.df = result_df

    def set_Rmax(self, model = "kokuso"):
        """
        Rmax（最大風速半径）を各種モデルで計算し、DataFrameに追加して返す
        model: "toyoda", "kokuso", "pari", "kisho" から選択
        """
        df = self.df
        Pc = df["PRES"]
        if model == "toyoda":
            R25 = (df["r50_long"] + df["r50_short"]) * 0.5
            R15 = (df["r30_long"] + df["r30_short"]) * 0.5
            Vmax = df["WIND"] * 0.514 # knot -> m/s

            # NaNを0に置換  
            R25 = R25.fillna(0)
            R15 = R15.fillna(0)
            Vmax = Vmax.fillna(0)
            LAT = df["LAT"].fillna(0)

            df["Rmax"] = np.where(
                LAT <= 20,
                -1.17 * Vmax - 0.004 * R25 + 0.03 * R15 + 3.54 * LAT + 38.1,
                -1.50 * Vmax - 0.090 * R25 + 0.09 * R15 + 1.54 * LAT + 68.8
            )
        elif model == "kokuso":
            df["Rmax"] = np.where(
                Pc <= 950,
                80 - 0.769 * (950-Pc),
                80 + 1.633 * (Pc-950)
            )
        elif model == "pari":
            df["Rmax"] = 94.89 * np.exp((Pc - 967)/61.5)

        elif model == "kisho":
            df["Rmax"] = 52.125 * np.exp((Pc - 952.7)/44.09)
        
        # DataFrameを更新したので、set_linestring()が自動実行される    
        self.df = df

    def interpolate_by_cumulative_distance(self, constant_velocity: float, time_interval_hours: float = 1.0):
        """
        累積移動距離をもとに線形補間で特定の時間ごとの座標を算出する
        
        Parameters:
        -----------
        constant_velocity : float
            一定の移動速度（m/s）
        time_interval_hours : float
            時間間隔（時間）。デフォルトは1時間。
            
        Returns:
        --------
        pd.DataFrame
            補間された座標データを含むDataFrame
        """
        return interpolate_by_cumulative_distance(self, constant_velocity, time_interval_hours)

    def resample_by_time_interval(self, time_interval_hours: float = 1.0):
        """
        指定した時間間隔でDataFrameをリサンプルする
        
        Parameters:
        -----------
        time_interval_hours : float
            時間間隔（時間）。デフォルトは1時間。
            
        Returns:
        --------
        pd.DataFrame
            リサンプルされたDataFrame
        """
        return resample_by_time_interval(self, time_interval_hours)

def interpolate_by_cumulative_distance(data, constant_velocity: float, time_interval_hours: float = 1.0):
    """
    累積移動距離をもとに線形補間で特定の時間ごとの座標を算出する独立関数
    
    Parameters:
    -----------
    data : TyphoonTrack or pd.DataFrame
        台風トラックデータ（TyphoonTrackインスタンスまたはDataFrame）
    constant_velocity : float
        一定の移動速度（m/s）
    time_interval_hours : float
        時間間隔（時間）。デフォルトは1時間。
        
    Returns:
    --------
    pd.DataFrame
        補間された座標データを含むDataFrame
    """
    # データの取得
    if hasattr(data, 'df') and hasattr(data, '__class__') and data.__class__.__name__ == 'TyphoonTrack':
        df = data.df.copy()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("dataはTyphoonTrackインスタンスまたはpd.DataFrameである必要があります")
    
    if df is None or df.empty or len(df) < 2:
        raise ValueError("データが空または1点のみです")
    
    # 座標変換：wgs84からjgd2011に変換する
    epsg4326_to_epsg6677 = Transformer.from_crs("epsg:4326", "epsg:6677", always_xy=True)
    df["x"], df["y"] = epsg4326_to_epsg6677.transform(df["LON"], df["LAT"])
    
    # 各点間の距離を計算
    df["segment_distance"] = 0.0
    for i in range(1, len(df)):
        dx = df["x"].iloc[i] - df["x"].iloc[i-1]
        dy = df["y"].iloc[i] - df["y"].iloc[i-1]
        df.loc[i, "segment_distance"] = np.sqrt(dx**2 + dy**2)
    
    # 累積距離を計算
    df["cumulative_distance"] = df["segment_distance"].cumsum()
    
    # 総距離
    total_distance = df["cumulative_distance"].iloc[-1]
    
    # 指定速度での移動時間を計算（秒単位）
    time_interval_seconds = time_interval_hours * 3600
    distance_per_step = constant_velocity * time_interval_seconds
    
    # 新しい時刻とそれに対応する累積距離を生成
    start_time = df["ISO_TIME"].iloc[0]
    new_times = []
    new_cumulative_distances = []
    
    current_distance = 0.0
    step = 0
    
    while current_distance <= total_distance:
        new_times.append(start_time + pd.Timedelta(hours=step * time_interval_hours))
        new_cumulative_distances.append(current_distance)
        current_distance += distance_per_step
        step += 1
    
    # 最後に終点を追加（もし到達していない場合）
    if new_cumulative_distances[-1] < total_distance:
        new_times.append(start_time + pd.Timedelta(hours=step * time_interval_hours))
        new_cumulative_distances.append(total_distance)
    
    # 線形補間で座標を計算
    new_lons = np.interp(new_cumulative_distances, df["cumulative_distance"], df["LON"])
    new_lats = np.interp(new_cumulative_distances, df["cumulative_distance"], df["LAT"])
    
    # その他の属性も線形補間
    new_pres = np.interp(new_cumulative_distances, df["cumulative_distance"], 
                        df["PRES"].ffill().bfill())
    new_wind = np.interp(new_cumulative_distances, df["cumulative_distance"], 
                        df["WIND"].ffill().bfill())
    
    # 強風半径の線形補間（存在する場合のみ）
    interpolated_data = {
        "ISO_TIME": new_times,
        "LAT": new_lats,
        "LON": new_lons,
        "PRES": new_pres,
        "WIND": new_wind,
        "cumulative_distance": new_cumulative_distances,
        "moving_velocity": [constant_velocity] * len(new_times),
        "time_interval_hours": [time_interval_hours] * len(new_times)
    }
    
    # 強風半径カラムが存在する場合のみ追加
    for col in ["r50_long", "r50_short", "r30_long", "r30_short"]:
        if col in df.columns:
            interpolated_data[col] = np.interp(new_cumulative_distances, df["cumulative_distance"], 
                                             df[col].ffill().bfill())
    
    # 新しいDataFrameを作成
    new_df = pd.DataFrame(interpolated_data)
    
    # 元のDataFrameから固定属性をコピー
    for col in ["international_id", "tropical_cyclone_id", "NAME", "SEASON"]:
        if col in df.columns:
            new_df[col] = df[col].iloc[0]
    
    return new_df

def calculate_moving_velocity(data):
    """
    実際のトラックから移動速度を計算してDataFrameに追加する独立関数
    
    Parameters:
    -----------
    data : TyphoonTrack or pd.DataFrame
        台風トラックデータ（TyphoonTrackインスタンスまたはDataFrame）
        
    Returns:
    --------
    pd.DataFrame
        移動速度が追加されたDataFrame
    """
    # データの取得
    if hasattr(data, 'df') and hasattr(data, '__class__') and data.__class__.__name__ == 'TyphoonTrack':
        df = data.df.copy()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("dataはTyphoonTrackインスタンスまたはpd.DataFrameである必要があります")
    
    if df is None or df.empty or len(df) < 2:
        raise ValueError("データが空または2点未満です")
    
    # 実際のトラックから移動速度を計算
    # 座標変換：wgs84からjgd2011に変換する
    epsg4326_to_epsg6677 = Transformer.from_crs("epsg:4326", "epsg:6677", always_xy=True)
    df["x"], df["y"] = epsg4326_to_epsg6677.transform(df["LON"], df["LAT"])
    df["gap_distance"] = np.sqrt(df["x"].diff()**2 + df["y"].diff()**2)
    df["moving_velocity"] = df["gap_distance"] / (df["ISO_TIME"].diff() / pd.Timedelta(seconds=1))
    df["moving_velocity"] = df["moving_velocity"].fillna(0)
    
    return df

def add_cumulative_distance(df):
    """
    累積距離を計算してDataFrameに追加する
    """
    # 座標変換：wgs84からjgd2011に変換する
    epsg4326_to_epsg6677 = Transformer.from_crs("epsg:4326", "epsg:6677", always_xy=True)
    df["x"], df["y"] = epsg4326_to_epsg6677.transform(df["LON"], df["LAT"])
    
    # 各点間の距離を計算
    df["segment_distance"] = 0.0
    for i in range(1, len(df)):
        dx = df["x"].iloc[i] - df["x"].iloc[i-1]
        dy = df["y"].iloc[i] - df["y"].iloc[i-1]
        df.loc[i, "segment_distance"] = np.sqrt(dx**2 + dy**2)
    
    # 累積距離を計算
    df["cumulative_distance"] = df["segment_distance"].cumsum()
    return df

def resample_by_time_interval(data, time_interval_hours: float = 1.0):
    """
    指定した時間間隔でDataFrameをリサンプルする独立関数
    
    Parameters:
    -----------
    data : TyphoonTrack or pd.DataFrame
        台風トラックデータ（TyphoonTrackインスタンスまたはDataFrame）
    time_interval_hours : float
        時間間隔（時間）。デフォルトは1時間。
        
    Returns:
    --------
    pd.DataFrame
        リサンプルされたDataFrame
    """
    # データの取得
    if hasattr(data, 'df') and hasattr(data, '__class__') and data.__class__.__name__ == 'TyphoonTrack':
        df = data.df.copy()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("dataはTyphoonTrackインスタンスまたはpd.DataFrameである必要があります")
    
    if df is None or df.empty or len(df) < 2:
        raise ValueError("データが空または1点のみです")
    
    # 時刻順にソート
    df = df.sort_values('ISO_TIME').reset_index(drop=True)
    
    # 開始時刻と終了時刻を取得
    start_time = df['ISO_TIME'].iloc[0]
    end_time = df['ISO_TIME'].iloc[-1]
    
    # 新しい時刻を生成（指定間隔で）
    new_times = []
    current_time = start_time
    while current_time <= end_time:
        new_times.append(current_time)
        current_time += pd.Timedelta(hours=time_interval_hours)
    
    # 最後に終了時刻を追加（まだ含まれていない場合）
    if new_times[-1] < end_time:
        new_times.append(end_time)
    
    # 時刻を数値に変換（補間用）
    df['time_numeric'] = (df['ISO_TIME'] - start_time).dt.total_seconds()
    new_times_numeric = [(t - start_time).total_seconds() for t in new_times]
    
    # 線形補間で座標を計算
    new_lons = np.interp(new_times_numeric, df['time_numeric'], df['LON'])
    new_lats = np.interp(new_times_numeric, df['time_numeric'], df['LAT'])
    
    # その他の数値属性も線形補間
    interpolated_data = {
        'ISO_TIME': new_times,
        'LAT': new_lats,
        'LON': new_lons,
        'time_interval_hours': [time_interval_hours] * len(new_times)
    }
    
    # 数値カラムの線形補間
    numeric_columns = ['PRES', 'WIND', 'r50_long', 'r50_short', 'r30_long', 'r30_short']
    for col in numeric_columns:
        if col in df.columns:
            # NaNでない値のみで補間
            valid_mask = df[col].notna()
            if valid_mask.sum() >= 2:  # 最低2点のデータが必要
                valid_df = df[valid_mask]
                interpolated_data[col] = np.interp(new_times_numeric, 
                                                 valid_df['time_numeric'], 
                                                 valid_df[col])
            else:
                # データが不足している場合は前方填充
                interpolated_data[col] = [df[col].ffill().bfill().iloc[0]] * len(new_times)
    
    # 新しいDataFrameを作成
    new_df = pd.DataFrame(interpolated_data)
    
    # 文字列カラムや分類データは最初の値を使用
    categorical_columns = ['international_id', 'tropical_cyclone_id', 'NAME', 'SEASON', 
                          'indicator', 'grade', 'r50_dir', 'r30_dir', 'landfall']
    for col in categorical_columns:
        if col in df.columns:
            new_df[col] = df[col].iloc[0]
    
    return new_df

