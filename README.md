# ETM
Empirical Typhoon Model (ETM) 

## セットアップ

```bash
uv sync          # パッケージ（etm）と開発用依存をインストール
uv run pytest    # テスト実行
```

## 単位・座標の約束

- 距離 `r`, `r0`: km（内部でコリオリ項は m に換算）
- 気圧: hPa
- 進行方向 `direction`: 方位角（北=0°、時計回り）。`TyphoonTrack.set_direction` と同じ定義
- コリオリパラメータ: `lat` 引数の緯度で計算（`cal_gradient_velocity` は省略時に台風中心の緯度 `y0` を使用）
