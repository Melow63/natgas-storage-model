"""Storage features: how the inventory and its monthly change compare with the last five years."""
import pandas as pd


def five_year_stats(storage, years=5, min_years=3):
    """Compare each month with the same calendar month in the previous `years` years.

    `storage` is the end-of-month inventory (Bcf). The comparison window only holds
    earlier years, so the current year never leaks into its own average.
    Returns the inventory, its five-year average / min / max for that month, the
    monthly change, the five-year average change, and the surplus measures.
    """
    df = pd.DataFrame({"storage": storage.astype(float)}).sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    if len(full) != len(df):
        raise ValueError("storage series has missing months")
    df["change"] = df["storage"].diff()
    month = df.index.month

    def trailing(col, how):
        shifted = df.groupby(month)[col].shift(1)
        roll = shifted.groupby(month).rolling(years, min_periods=min_years)
        out = getattr(roll, how)()
        out.index = out.index.droplevel(0)
        return out.sort_index()

    df["avg5"] = trailing("storage", "mean")
    df["min5"] = trailing("storage", "min")
    df["max5"] = trailing("storage", "max")
    df["avg5_change"] = trailing("change", "mean")
    df["surplus_bcf"] = df["storage"] - df["avg5"]
    df["surplus_pct"] = 100 * df["surplus_bcf"] / df["avg5"]
    df["change_dev"] = df["change"] - df["avg5_change"]
    return df
