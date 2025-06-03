# %%
"""
台風経路回転のテストとデバッグ
"""
import sys
import os
sys.path.append('src')

import etm.util as util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォントの設定
plt.rcParams['font.family'] = 'DejaVu Sans'
# 日本語テキストを英語に変更する

# %% データの読み込みテスト
print("=== データ読み込みテスト ===")
try:
    typhoons = util.read_bsttxt("jma-bst/bst_all.txt")
    print(f"✓ 台風データ読み込み成功: {len(typhoons)}個")
    
    # 最初の台風を選択
    test_typhoon = typhoons[0]
    print(f"✓ テスト台風: {test_typhoon.name}")
    print(f"✓ 台風ID: {test_typhoon.typhoon_number}")
    print(f"✓ 経路点数: {len(test_typhoon.df)}")
    
    # 元の座標を確認
    original_df = test_typhoon.df
    print(f"✓ 元の経度範囲: {original_df['LON'].min():.2f} - {original_df['LON'].max():.2f}")
    print(f"✓ 元の緯度範囲: {original_df['LAT'].min():.2f} - {original_df['LAT'].max():.2f}")
    
except Exception as e:
    print(f"✗ データ読み込みエラー: {e}")
    sys.exit(1)

# %% 回転関数の直接テスト
print("\n=== 座標回転テスト ===")
from etm.rotation import rotate_coordinates_around_center

# テスト座標
test_lon = [130.0, 135.0, 140.0]
test_lat = [30.0, 35.0, 40.0]
center_lon = 135.0
center_lat = 35.0
angle = 90  # 90度回転

print(f"元の座標: LON={test_lon}, LAT={test_lat}")
print(f"回転中心: ({center_lon}, {center_lat})")
print(f"回転角度: {angle}度")

rotated_lon, rotated_lat = rotate_coordinates_around_center(
    test_lon, test_lat, center_lon, center_lat, angle
)

print(f"回転後座標: LON={[f'{x:.2f}' for x in rotated_lon]}, LAT={[f'{x:.2f}' for x in rotated_lat]}")

# 変化を確認
lon_diff = np.array(rotated_lon) - np.array(test_lon)
lat_diff = np.array(rotated_lat) - np.array(test_lat)
print(f"経度の変化: {[f'{x:.2f}' for x in lon_diff]}")
print(f"緯度の変化: {[f'{x:.2f}' for x in lat_diff]}")

# %% 台風トラック回転のテスト（単純回転）
print("\n=== 台風トラック回転テスト（単純回転） ===")
from etm.rotation import rotate_typhoon_track

# 回転実行
try:
    rotated_typhoon_simple = rotate_typhoon_track(
        test_typhoon, 
        center_lon, 
        center_lat, 
        angle
    )
    
    print(f"✓ 単純回転処理成功")
    print(f"✓ 回転後の台風名: {rotated_typhoon_simple.name}")
    print(f"✓ 回転後の点数: {len(rotated_typhoon_simple.df)}")
    
    # 座標変化を確認
    rotated_df_simple = rotated_typhoon_simple.df
    print(f"✓ 回転後の経度範囲: {rotated_df_simple['LON'].min():.2f} - {rotated_df_simple['LON'].max():.2f}")
    print(f"✓ 回転後の緯度範囲: {rotated_df_simple['LAT'].min():.2f} - {rotated_df_simple['LAT'].max():.2f}")
    
    # 変化量の統計
    lon_change_simple = rotated_df_simple['LON'] - original_df['LON']
    lat_change_simple = rotated_df_simple['LAT'] - original_df['LAT']
    print(f"✓ 経度変化統計: 平均={lon_change_simple.mean():.4f}, 標準偏差={lon_change_simple.std():.4f}")
    print(f"✓ 緯度変化統計: 平均={lat_change_simple.mean():.4f}, 標準偏差={lat_change_simple.std():.4f}")
    
    # 最大変化量
    max_lon_change_simple = max(abs(lon_change_simple.min()), abs(lon_change_simple.max()))
    max_lat_change_simple = max(abs(lat_change_simple.min()), abs(lat_change_simple.max()))
    print(f"✓ 最大経度変化: {max_lon_change_simple:.4f}")
    print(f"✓ 最大緯度変化: {max_lat_change_simple:.4f}")
    
    if max_lon_change_simple < 0.001 and max_lat_change_simple < 0.001:
        print("⚠️  警告: 座標の変化が非常に小さいです。回転が正しく動作していない可能性があります。")
    else:
        print("✓ 座標が正常に変化しています。")
        
except Exception as e:
    print(f"✗ 単純回転処理エラー: {e}")
    import traceback
    traceback.print_exc()

# %% 台風トラック回転のテスト（投影座標系回転）
print("\n=== 台風トラック回転テスト（投影座標系回転） ===")
from etm.rotation import rotate_typhoon_track_projected

# 回転実行
try:
    rotated_typhoon_projected = rotate_typhoon_track_projected(
        test_typhoon, 
        center_lon, 
        center_lat, 
        angle,
        crs='EPSG:3857'  # Web Mercator
    )
    
    print(f"✓ 投影座標系回転処理成功")
    print(f"✓ 回転後の台風名: {rotated_typhoon_projected.name}")
    print(f"✓ 回転後の点数: {len(rotated_typhoon_projected.df)}")
    
    # 座標変化を確認
    rotated_df_projected = rotated_typhoon_projected.df
    print(f"✓ 回転後の経度範囲: {rotated_df_projected['LON'].min():.2f} - {rotated_df_projected['LON'].max():.2f}")
    print(f"✓ 回転後の緯度範囲: {rotated_df_projected['LAT'].min():.2f} - {rotated_df_projected['LAT'].max():.2f}")
    
    # 変化量の統計
    lon_change_projected = rotated_df_projected['LON'] - original_df['LON']
    lat_change_projected = rotated_df_projected['LAT'] - original_df['LAT']
    print(f"✓ 経度変化統計: 平均={lon_change_projected.mean():.4f}, 標準偏差={lon_change_projected.std():.4f}")
    print(f"✓ 緯度変化統計: 平均={lat_change_projected.mean():.4f}, 標準偏差={lat_change_projected.std():.4f}")
    
    # 最大変化量
    max_lon_change_projected = max(abs(lon_change_projected.min()), abs(lon_change_projected.max()))
    max_lat_change_projected = max(abs(lat_change_projected.min()), abs(lat_change_projected.max()))
    print(f"✓ 最大経度変化: {max_lon_change_projected:.4f}")
    print(f"✓ 最大緯度変化: {max_lat_change_projected:.4f}")
    
    if max_lon_change_projected < 0.001 and max_lat_change_projected < 0.001:
        print("⚠️  警告: 座標の変化が非常に小さいです。回転が正しく動作していない可能性があります。")
    else:
        print("✓ 座標が正常に変化しています。")
        
except Exception as e:
    print(f"✗ 投影座標系回転処理エラー: {e}")
    import traceback
    traceback.print_exc()

# %% 単純回転と投影座標系回転の比較
print("\n=== 単純回転 vs 投影座標系回転 比較 ===")
try:
    # 差異の計算
    lon_diff_methods = rotated_df_projected['LON'] - rotated_df_simple['LON']
    lat_diff_methods = rotated_df_projected['LAT'] - rotated_df_simple['LAT']
    
    print(f"回転方法間の経度差: 平均={lon_diff_methods.mean():.6f}, 最大={abs(lon_diff_methods).max():.6f}")
    print(f"回転方法間の緯度差: 平均={lat_diff_methods.mean():.6f}, 最大={abs(lat_diff_methods).max():.6f}")
    
    # 距離での差異
    distance_diff = np.sqrt(lon_diff_methods**2 + lat_diff_methods**2)
    print(f"回転方法間の距離差: 平均={distance_diff.mean():.6f}, 最大={distance_diff.max():.6f}")
    
    if distance_diff.max() > 0.1:
        print("⚠️  単純回転と投影座標系回転で大きな差があります")
    else:
        print("✓ 単純回転と投影座標系回転の差は小さいです")
        
except Exception as e:
    print(f"✗ 比較エラー: {e}")

# %% 可視化テスト（3つの経路を比較）
print("\n=== 可視化テスト（比較） ===")
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 左側: 元の経路と単純回転
    ax1.plot(original_df['LON'], original_df['LAT'], 'b-o', 
            label='元の経路', markersize=4, linewidth=2)
    ax1.plot(rotated_df_simple['LON'], rotated_df_simple['LAT'], 'r-s', 
            label=f'単純回転 ({angle}°)', markersize=4, linewidth=2)
    ax1.plot(center_lon, center_lat, 'ko', markersize=10, label='回転中心')
    
    # 開始点と終了点
    ax1.plot(original_df['LON'].iloc[0], original_df['LAT'].iloc[0], 
            'bo', markersize=8, label='元の開始点')
    ax1.plot(rotated_df_simple['LON'].iloc[0], rotated_df_simple['LAT'].iloc[0], 
            'ro', markersize=8, label='回転後の開始点')
    
    ax1.set_xlabel('経度')
    ax1.set_ylabel('緯度')
    ax1.set_title(f'単純回転\n中心: ({center_lon}, {center_lat}), 角度: {angle}°')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 右側: 元の経路と投影座標系回転
    ax2.plot(original_df['LON'], original_df['LAT'], 'b-o', 
            label='元の経路', markersize=4, linewidth=2)
    ax2.plot(rotated_df_projected['LON'], rotated_df_projected['LAT'], 'g-^', 
            label=f'投影座標系回転 ({angle}°)', markersize=4, linewidth=2)
    ax2.plot(center_lon, center_lat, 'ko', markersize=10, label='回転中心')
    
    # 開始点と終了点
    ax2.plot(original_df['LON'].iloc[0], original_df['LAT'].iloc[0], 
            'bo', markersize=8, label='元の開始点')
    ax2.plot(rotated_df_projected['LON'].iloc[0], rotated_df_projected['LAT'].iloc[0], 
            'go', markersize=8, label='回転後の開始点')
    
    ax2.set_xlabel('経度')
    ax2.set_ylabel('緯度')
    ax2.set_title(f'投影座標系回転\n中心: ({center_lon}, {center_lat}), 角度: {angle}°')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('rotation_comparison_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3つすべてを重ねたプロット
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.plot(original_df['LON'], original_df['LAT'], 'b-o', 
            label='元の経路', markersize=5, linewidth=3, alpha=0.8)
    ax.plot(rotated_df_simple['LON'], rotated_df_simple['LAT'], 'r-s', 
            label=f'単純回転 ({angle}°)', markersize=4, linewidth=2)
    ax.plot(rotated_df_projected['LON'], rotated_df_projected['LAT'], 'g-^', 
            label=f'投影座標系回転 ({angle}°)', markersize=4, linewidth=2)
    ax.plot(center_lon, center_lat, 'ko', markersize=12, label='回転中心')
    
    ax.set_xlabel('経度')
    ax.set_ylabel('緯度')
    ax.set_title(f'回転方法の比較\n中心: ({center_lon}, {center_lat}), 角度: {angle}°')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('rotation_all_comparison_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ 可視化完了: rotation_comparison_test.png と rotation_all_comparison_test.png に保存されました")
    
except Exception as e:
    print(f"✗ 可視化エラー: {e}")

# %% 詳細なデータ比較（3つの方法）
print("\n=== 詳細データ比較（3つの方法） ===")
comparison_df = pd.DataFrame({
    '点番号': range(len(original_df)),
    '元_経度': original_df['LON'].values,
    '元_緯度': original_df['LAT'].values,
    '単純回転_経度': rotated_df_simple['LON'].values,
    '単純回転_緯度': rotated_df_simple['LAT'].values,
    '投影回転_経度': rotated_df_projected['LON'].values,
    '投影回転_緯度': rotated_df_projected['LAT'].values,
})

# 各方法での変化量
comparison_df['単純_経度差'] = comparison_df['単純回転_経度'] - comparison_df['元_経度']
comparison_df['単純_緯度差'] = comparison_df['単純回転_緯度'] - comparison_df['元_緯度']
comparison_df['単純_距離変化'] = np.sqrt(comparison_df['単純_経度差']**2 + comparison_df['単純_緯度差']**2)

comparison_df['投影_経度差'] = comparison_df['投影回転_経度'] - comparison_df['元_経度']
comparison_df['投影_緯度差'] = comparison_df['投影回転_緯度'] - comparison_df['元_緯度']
comparison_df['投影_距離変化'] = np.sqrt(comparison_df['投影_経度差']**2 + comparison_df['投影_緯度差']**2)

# 方法間の差異
comparison_df['方法間_経度差'] = comparison_df['投影回転_経度'] - comparison_df['単純回転_経度']
comparison_df['方法間_緯度差'] = comparison_df['投影回転_緯度'] - comparison_df['単純回転_緯度']
comparison_df['方法間_距離差'] = np.sqrt(comparison_df['方法間_経度差']**2 + comparison_df['方法間_緯度差']**2)

print("最初の5点の比較:")
print(comparison_df[['点番号', '元_経度', '元_緯度', '単純回転_経度', '単純回転_緯度', 
                   '投影回転_経度', '投影回転_緯度', '方法間_距離差']].head().round(4))

print(f"\n統計サマリー:")
print(f"【単純回転】")
print(f"  経度差の範囲: {comparison_df['単純_経度差'].min():.4f} ~ {comparison_df['単純_経度差'].max():.4f}")
print(f"  緯度差の範囲: {comparison_df['単純_緯度差'].min():.4f} ~ {comparison_df['単純_緯度差'].max():.4f}")
print(f"  距離変化の平均: {comparison_df['単純_距離変化'].mean():.4f}")
print(f"  距離変化の最大: {comparison_df['単純_距離変化'].max():.4f}")

print(f"【投影座標系回転】")
print(f"  経度差の範囲: {comparison_df['投影_経度差'].min():.4f} ~ {comparison_df['投影_経度差'].max():.4f}")
print(f"  緯度差の範囲: {comparison_df['投影_緯度差'].min():.4f} ~ {comparison_df['投影_緯度差'].max():.4f}")
print(f"  距離変化の平均: {comparison_df['投影_距離変化'].mean():.4f}")
print(f"  距離変化の最大: {comparison_df['投影_距離変化'].max():.4f}")

print(f"【方法間の差異】")
print(f"  経度差の範囲: {comparison_df['方法間_経度差'].min():.6f} ~ {comparison_df['方法間_経度差'].max():.6f}")
print(f"  緯度差の範囲: {comparison_df['方法間_緯度差'].min():.6f} ~ {comparison_df['方法間_緯度差'].max():.6f}")
print(f"  距離差の平均: {comparison_df['方法間_距離差'].mean():.6f}")
print(f"  距離差の最大: {comparison_df['方法間_距離差'].max():.6f}")

if comparison_df['単純_距離変化'].max() > 1.0 and comparison_df['投影_距離変化'].max() > 1.0:
    print("✓ 両方の回転方法が正常に動作しています")
    if comparison_df['方法間_距離差'].max() > 1.0:
        print("⚠️  回転方法間で大きな差があります。用途に応じて適切な方法を選択してください。")
    else:
        print("✓ 回転方法間の差は小さく、どちらを使用しても問題ありません。")
else:
    print("⚠️  回転の効果が小さすぎます。パラメータを確認してください。")

print("\n=== テスト完了 ===") 