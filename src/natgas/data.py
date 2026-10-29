"""Load the EIA Natural Gas Monthly tables (Excel downloads) kept in data/raw/.

Files, all from https://www.eia.gov/naturalgas/monthly/ (Table of contents, XLS links):
    ngm01_supply_disposition.xlsx   Table 1, supply and disposition
    ngm03_prices.xlsx               Table 3, national average prices (includes Henry Hub)
    ngm08_storage.xlsx              Table 8, underground storage, all operators
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

HENRY_HUB = "RNGWHHD"
WORKING_GAS = "N5020US1"
DRY_PRODUCTION = "N9070US1"
CONSUMPTION = "N9140US1"


def _table(filename, sheet="Data 1"):
    """Read an EIA workbook sheet; the second row holds the stable series keys."""
    df = pd.read_excel(RAW / filename, sheet_name=sheet, header=1)
    df = df.rename(columns={df.columns[0]: "date"})
    df = df[df["date"].map(lambda x: isinstance(x, (datetime, pd.Timestamp)))].copy()
    df["date"] = pd.to_datetime(df["date"])
    # EIA stamps monthly points on the 15th; use month start so tables line up
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.set_index("date").apply(pd.to_numeric, errors="coerce")


def storage_monthly(since="1976-01-01"):
    """End-of-month U.S. working gas in underground storage, Bcf.

    The table has gaps in 1974-75, so the series starts in 1976 where it is continuous.
    """
    return _table("ngm08_storage.xlsx")[WORKING_GAS].dropna().loc[since:].rename("storage")


def henry_hub_monthly():
    """Monthly average Henry Hub spot price, USD/MMBtu."""
    return _table("ngm03_prices.xlsx")[HENRY_HUB].dropna().rename("price")


def balance_monthly():
    """Dry production and total consumption, Bcf per month."""
    t = _table("ngm01_supply_disposition.xlsx")
    return t[[DRY_PRODUCTION, CONSUMPTION]].rename(
        columns={DRY_PRODUCTION: "production", CONSUMPTION: "consumption"}
    ).dropna()
