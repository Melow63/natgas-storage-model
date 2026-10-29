"""Load the EIA tables, build the features, fit the models, write charts and a results file."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natgas import data, model, plots, storage  # noqa: E402


def _metrics_block(f, title, m):
    f.write(title + "\n")
    for k, v in m.items():
        f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")
    f.write("\n")


def main():
    stor = data.storage_monthly()
    price = data.henry_hub_monthly()

    feats = storage.five_year_stats(stor)
    ds = model.monthly_dataset(price, feats)

    same = model.fit_same_month(ds)
    nxt = model.fit_ols(ds)
    wf = model.walk_forward(ds)
    post10 = ds.loc["2010":]
    wf10 = model.walk_forward(post10, min_train=84)
    # Winter Storm Uri: Henry Hub spiked ~70% in February 2021 and fell back in March
    wf_ex = wf.drop(wf.loc["2021-01-01":"2021-02-01"].index)

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    plots.storage_vs_range(feats, out / "storage_vs_range.png")
    plots.price_and_surplus(ds, out / "price_and_surplus.png")
    plots.reaction_scatter(ds, same, out / "reaction_scatter.png")
    plots.walk_forward_chart(wf, out / "walk_forward.png")

    with open(out / "results.txt", "w") as f:
        f.write(f"Storage data: {stor.index.min():%Y-%m} to {stor.index.max():%Y-%m}\n")
        f.write(f"Price data:   {price.index.min():%Y-%m} to {price.index.max():%Y-%m}\n")
        f.write(f"Sample: {ds.index.min():%Y-%m} to {ds.index.max():%Y-%m} ({len(ds)} months)\n\n")
        f.write("1) Same month: price move in month t on the storage news of month t\n")
        f.write(str(same.summary()) + f"\nR-squared: {same.rsquared:.3f}\n\n")
        f.write("2) Next month: price move in t+1 on the storage news of month t\n")
        f.write(str(nxt.summary()) + "\n\n")
        _metrics_block(f, "3) Walk-forward, next month (expanding window, min 120 months)", model.oos_metrics(wf))
        _metrics_block(f, "4) Walk-forward, 2010 onward only (min 84 months)", model.oos_metrics(wf10))
        _metrics_block(f, "5) Walk-forward, next month, without the Winter Storm Uri months (Jan-Feb 2021 rows)", model.oos_metrics(wf_ex))
    print((out / "results.txt").read_text())


if __name__ == "__main__":
    main()
