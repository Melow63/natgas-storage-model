import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREY = "#1f5fa8", "#e07b00", "#8a8a8a"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def storage_vs_range(feats, path, since="2018-01-01"):
    d = feats.loc[since:]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(d.index, d["min5"], d["max5"], color=GREY, alpha=0.25, label="5-year range")
    ax.plot(d.index, d["avg5"], color=GREY, lw=1.5, label="5-year average")
    ax.plot(d.index, d["storage"], color=BLUE, lw=2, label="Working gas in storage")
    ax.set_ylabel("Bcf, end of month")
    ax.set_title("U.S. working gas in storage vs. the previous five years")
    ax.legend(frameon=False)
    _save(fig, path)


def price_and_surplus(df, path, since="2010-01-01"):
    d = df.loc[since:]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(d.index, d["price"], color=BLUE, lw=1.6)
    ax.set_ylabel("Henry Hub monthly average, USD/MMBtu", color=BLUE)
    ax2 = ax.twinx()
    ax2.plot(d.index, d["surplus_pct"], color=ORANGE, lw=1.2)
    ax2.axhline(0, color=GREY, lw=0.8)
    ax2.set_ylabel("Storage vs. 5-year average, %", color=ORANGE)
    ax.set_title("Henry Hub price and the storage surplus or deficit")
    _save(fig, path)


def reaction_scatter(df, fit, path):
    d = df.dropna(subset=["dlogp"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(d["change_dev"], 100 * d["dlogp"], s=10, color=BLUE, alpha=0.45)
    xs = np.linspace(d["change_dev"].min(), d["change_dev"].max(), 50)
    p = fit.params
    ys = p["const"] + p["change_dev"] * xs + p["surplus_pct"] * d["surplus_pct"].mean()
    ax.plot(xs, 100 * ys, color=ORANGE, lw=2)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("Monthly storage change minus 5-year average change (100 Bcf)")
    ax.set_ylabel("Same-month Henry Hub move, %")
    ax.set_title("Price move vs. the storage number, same month")
    _save(fig, path)


def walk_forward_chart(wf, path):
    """Cumulative squared-error gain of the storage model over the naive mean."""
    gain = ((wf["actual"] - wf["benchmark"]) ** 2 - (wf["actual"] - wf["pred"]) ** 2).cumsum()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(gain.index, gain, color=BLUE, lw=1.8)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_ylabel("Cumulative squared-error gain")
    ax.set_title("Next-month forecast: storage model minus naive mean (above 0 means the model helps)")
    _save(fig, path)
