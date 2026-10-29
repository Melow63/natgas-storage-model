"""Next-month price reaction to storage, and a walk-forward check of the relationship."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

FEATURES = ["change_dev", "surplus_pct"]


def monthly_dataset(price, storage_feats):
    """Line up end-of-month storage with the next month's change in the average price.

    Storage at the end of month t is known from the weekly reports in the first days
    of month t+1, before most of that month's prices are set. The target is therefore
    log(avg price[t+1]) - log(avg price[t]), and the features are dated t.
    """
    price = price[price > 0]
    moves = pd.DataFrame({"price": price, "dlogp": np.log(price).diff()})
    moves["target"] = moves["dlogp"].shift(-1)  # shift on the price calendar, before joining
    df = storage_feats.join(moves, how="inner")
    df["change_dev"] = df["change_dev"] / 100.0  # per 100 Bcf, keeps coefficients readable
    return df.dropna(subset=FEATURES + ["target"])


def fit_ols(df, features=FEATURES, hac_lags=3):
    X = sm.add_constant(df[features])
    return sm.OLS(df["target"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def fit_same_month(df, features=FEATURES, hac_lags=3):
    """Descriptive fit: price move in month t against the storage news of month t.

    Price and storage react to the same weather, so this is an association, not a cause.
    """
    d = df.dropna(subset=["dlogp"])
    X = sm.add_constant(d[features])
    return sm.OLS(d["dlogp"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def walk_forward(df, features=FEATURES, min_train=120):
    """Expanding window: refit every month, predict the next one, never look ahead."""
    preds, dates = [], []
    for i in range(min_train, len(df)):
        train = df.iloc[:i]
        model = sm.OLS(train["target"], sm.add_constant(train[features])).fit()
        row = sm.add_constant(df[features].iloc[[i]], has_constant="add")
        preds.append(float(model.predict(row).iloc[0]))
        dates.append(df.index[i])
    out = pd.DataFrame({"pred": preds, "actual": df["target"].iloc[min_train:].to_numpy()}, index=dates)
    out["benchmark"] = df["target"].expanding().mean().shift(1).iloc[min_train:].to_numpy()
    return out


def oos_metrics(wf):
    sse = ((wf["actual"] - wf["pred"]) ** 2).sum()
    sse_bench = ((wf["actual"] - wf["benchmark"]) ** 2).sum()
    return {
        "months": int(len(wf)),
        "oos_r2_vs_mean": float(1 - sse / sse_bench),
        "hit_rate": float((np.sign(wf["pred"]) == np.sign(wf["actual"])).mean()),
    }
