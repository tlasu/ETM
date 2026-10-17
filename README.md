# ETM
Empirical Typhoon Model (ETM) 

## セットアップ

```bash
uv sync          # パッケージ（etm）と開発用依存をインストール
uv run pytest    # テスト実行
```

## 使い方

```python
from etm import util
from etm.domain import Grid
from etm.field import compute_wind_field

typhoons = util.read_bsttxt("jma-bst/bst_all.txt")  # または util.read_parquet(...)
track = next(t for t in typhoons if t.name == "HAGIBIS" and t.season == 2019)

grid = Grid.from_extent(130, 150, 25, 45, resolution=0.1)  # 投影格子は Grid.from_projected(x, y, "EPSG:6677")
ds = compute_wind_field(track, grid, dt_hours=1.0, rmax="kokuso", start="2019-10-11", end="2019-10-13")
ds.to_netcdf("hagibis.nc")  # pressure, u10, v10, wind_speed（次元: time, y, x）
```

- `rmax`: 最大風速半径。数値[km]で一定値、または `"kokuso"`, `"toyoda"`, `"pari"`, `"kisho"` のモデル名
- 詳しい例は `example/cal.py`

## 単位・座標の約束

- 距離 `r`, `r0`: km（内部でコリオリ項は m に換算）
- 気圧: hPa
- 進行方向 `direction`: 方位角（北=0°、時計回り）。`TyphoonTrack.set_direction` と同じ定義
- コリオリパラメータ: `lat` 引数の緯度で計算（`cal_gradient_velocity` は省略時に台風中心の緯度 `y0` を使用）
- 距離・方位: GRS80 楕円体上の測地線（`pyproj.Geod`）で計算
- 経路の回転: 回転中心からの測地線距離を保って方位角を回転（正の角度で反時計回り）。`rotate_typhoon_track_projected` は既定で回転中心を原点とする正距方位図法を使用
