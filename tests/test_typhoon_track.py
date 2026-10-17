from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etm.TyphoonTrack import (
    add_cumulative_distance,
    calculate_moving_velocity,
    interpolate_by_cumulative_distance,
    segment_distances,
)

BST2023 = Path(__file__).parents[1] / "jma-bst" / "bst2023.txt"


def test_segment_distances_geodesic():
    """赤道上の経度1°は約111.32km、北緯60°では約半分"""
    df = pd.DataFrame({"LON": [150.0, 151.0, 150.0, 151.0], "LAT": [0.0, 0.0, 60.0, 60.0]})
    d = segment_distances(df)
    assert d[0] == 0
    assert d[1] == pytest.approx(111_320, rel=1e-3)
    assert d[3] == pytest.approx(55_800, rel=1e-2)


def test_moving_velocity(make_track):
    """6時間で緯度1°（約110.7km）北上 -> 約5.1m/s"""
    track = make_track([140.0, 140.0, 140.0], [20.0, 21.0, 22.0])
    df = calculate_moving_velocity(track)
    assert df["moving_velocity"].iloc[0] == 0
    assert df["moving_velocity"].iloc[1] == pytest.approx(110_700 / 21_600, rel=1e-2)
    assert "x" not in df.columns


def test_cumulative_distance_with_non_default_index(make_track):
    """0始まりの連番でないインデックスでも行が増えない"""
    df = make_track([130.0, 131.0, 132.0], [20.0, 21.0, 22.0]).df.copy()
    df.index = [10, 11, 12]
    result = add_cumulative_distance(df)
    assert len(result) == 3
    np.testing.assert_allclose(result["cumulative_distance"], np.cumsum(segment_distances(df)))


def test_interpolate_by_cumulative_distance_end_point(make_track):
    track = make_track([130.0, 131.0, 132.0], [20.0, 21.0, 22.0])
    result = interpolate_by_cumulative_distance(track, constant_velocity=5.0)
    assert result["LON"].iloc[-1] == pytest.approx(132.0)
    assert result["LAT"].iloc[-1] == pytest.approx(22.0)
    np.testing.assert_allclose(np.diff(result["cumulative_distance"].iloc[:-1]), 5.0 * 3600)


def test_set_direction(make_track):
    track = make_track([130.0, 131.0, 131.0], [20.0, 20.0, 21.0])
    track.set_direction()
    direction = track.df["direction"]
    assert direction.iloc[0] == pytest.approx(90, abs=0.5)  # 東進
    assert direction.iloc[1] == pytest.approx(0, abs=1e-6)  # 北進
    assert np.isnan(direction.iloc[2])


@pytest.mark.parametrize("shape", [(1,), (1, 1), (3,)])
def test_geod_inv_keeps_array_shape(shape):
    """要素数1の配列でも警告を出さず、入力と同じ形状の配列を返す"""
    from etm.TyphoonTrack import geod_inv

    lons = np.full(shape, 131.0)
    lats = np.full(shape, 20.0)
    az, _, dist = geod_inv(130.0, 20.0, lons, lats)
    assert isinstance(dist, np.ndarray) and dist.shape == shape
    np.testing.assert_allclose(az, 89.8, atol=0.1)


def test_read_bsttxt_matches_line_parser():
    """一括パーサーの結果が1行ずつのパーサーと一致する"""
    from etm import util
    from etm.TyphoonTrack import TyphoonTrack

    path = BST2023
    new = util.read_bsttxt(path)
    lines = open(path, encoding="ascii").read().splitlines()
    old, c = [], 0
    while c < len(lines):
        n = int(lines[c].split()[2])
        t = TyphoonTrack()
        t.read_bst({"header": lines[c], "data": lines[c + 1:c + 1 + n]})
        old.append(t)
        c += n + 1
    assert len(new) == len(old) > 0
    for a, b in zip(old, new):
        assert a.name == b.name and a.typhoon_number == b.typhoon_number
        # 旧パーサーは欠損を None、新パーサーは NaN で表す
        pd.testing.assert_frame_equal(a.df.fillna(np.nan).infer_objects(), b.df, check_dtype=False)


def test_parse_bst_lines_rejects_wrong_count():
    from etm.TyphoonTrack import parse_bst_lines

    lines = open(BST2023, encoding="ascii").read().splitlines()
    with pytest.raises(ValueError, match="一致しません"):
        parse_bst_lines(lines[:-1])


def test_gdf_is_created_lazily(make_track):
    track = make_track([130.0, 131.0], [20.0, 21.0])
    assert track._gdf is None
    assert track.gdf.geometry.iloc[0].coords[:] == [(130.0, 20.0), (131.0, 21.0)]


def test_gdf_warns_on_nan(make_track):
    track = make_track([130.0, np.nan], [20.0, 21.0])
    with pytest.warns(UserWarning, match="NaN"):
        assert track.gdf is None


def test_set_rmax_unknown_model(make_track):
    with pytest.raises(ValueError):
        make_track([130.0, 131.0], [20.0, 21.0]).set_Rmax("unknown")
