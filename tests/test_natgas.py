import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from natgas import model, storage  # noqa: E402


def fake_storage(first=1990, last=2005):
    idx = pd.date_range(f"{first}-01-01", f"{last}-12-01", freq="MS")
    return pd.Series([d.year * 100 + d.month for d in idx], index=idx, dtype=float)


def test_five_year_average_uses_previous_years_only():
    feats = storage.five_year_stats(fake_storage())
    row = feats.loc["2000-03-01"]
    assert row["avg5"] == np.mean([1995, 1996, 1997, 1998, 1999]) * 100 + 3
    assert row["min5"] == 1995 * 100 + 3
    assert row["max5"] == 1999 * 100 + 3


def test_first_years_have_no_average():
    feats = storage.five_year_stats(fake_storage())
    assert np.isnan(feats.loc["1991-03-01", "avg5"])


def test_surplus_matches_definition():
    feats = storage.five_year_stats(fake_storage()).dropna(subset=["avg5"])
    assert np.allclose(feats["surplus_bcf"], feats["storage"] - feats["avg5"])
    assert np.allclose(feats["surplus_pct"], 100 * feats["surplus_bcf"] / feats["avg5"])


def test_missing_month_is_rejected():
    s = fake_storage().drop(pd.Timestamp("1995-06-01"))
    try:
        storage.five_year_stats(s)
    except ValueError:
        return
    raise AssertionError("gap in the series should raise")


def test_target_is_next_months_price_change():
    s = fake_storage()
    price = pd.Series(np.arange(1.0, len(s) + 1.0), index=s.index)
    feats = storage.five_year_stats(s)
    ds = model.monthly_dataset(price, feats)
    t = pd.Timestamp("2001-05-01")
    expected = np.log(price[pd.Timestamp("2001-06-01")] / price[t])
    assert np.isclose(ds.loc[t, "target"], expected)


def test_last_storage_month_keeps_its_target_when_price_runs_one_month_longer():
    s = fake_storage()
    price = pd.Series(np.arange(1.0, len(s) + 2.0), index=pd.date_range(s.index[0], periods=len(s) + 1, freq="MS"))
    ds = model.monthly_dataset(price, storage.five_year_stats(s))
    assert ds.index.max() == s.index.max()


def _toy(n, seed, slope=-0.02):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    df = pd.DataFrame({"change_dev": rng.normal(size=n), "surplus_pct": rng.normal(size=n)}, index=idx)
    df["target"] = slope * df["change_dev"] + rng.normal(scale=0.01, size=n)
    df["dlogp"] = df["target"]
    return df


def test_ols_recovers_known_coefficient():
    fit = model.fit_ols(_toy(600, 7))
    assert abs(fit.params["change_dev"] + 0.02) < 0.003


def test_same_month_fit_recovers_known_coefficient():
    fit = model.fit_same_month(_toy(600, 11))
    assert abs(fit.params["change_dev"] + 0.02) < 0.003


def test_walk_forward_uses_no_future_rows():
    df = _toy(400, 3)
    base = model.walk_forward(df, min_train=120)
    altered = df.copy()
    altered.iloc[-1, altered.columns.get_loc("target")] += 5.0
    again = model.walk_forward(altered, min_train=120)
    # changing the very last target must not change any earlier prediction
    assert np.allclose(base["pred"].iloc[:-1], again["pred"].iloc[:-1])
