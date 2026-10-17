import pandas as pd
import pytest

from etm.TyphoonTrack import TyphoonTrack


@pytest.fixture
def make_track():
    def _make(lons, lats, hours=6):
        n = len(lons)
        df = pd.DataFrame({
            "international_id": ["0001"] * n,
            "tropical_cyclone_id": ["0001"] * n,
            "NAME": ["TEST"] * n,
            "SEASON": [2000] * n,
            "ISO_TIME": pd.date_range("2000-08-01", periods=n, freq=f"{hours}h"),
            "LAT": lats,
            "LON": lons,
            "PRES": [960] * n,
        })
        track = TyphoonTrack()
        track.df = df
        return track
    return _make
