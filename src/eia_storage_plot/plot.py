from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

def select_apr_oct_last5(df: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    """
    Keep a continuous dataset for Apr–Oct for the last 5 *completed* years.
    Example (today=2025-09-17): keep 2020–2024, months 4..10, inclusive.
    """
    if today is None:
        today = date.today()
    end_year = today.year - 1
    start_year = end_year - 4

    d = df.copy()
    d["year"] = d["period"].dt.year
    d["month"] = d["period"].dt.month
    mask = (d["year"].between(start_year, end_year)) & (d["month"].between(4, 10))
    d = d.loc[mask].drop(columns=["year", "month"])
    # Sort by period just to be neat
    d = d.sort_values("period").reset_index(drop=True)
    return d

def make_scatter(df: pd.DataFrame, out_png: str) -> None:
    """
    Scatter: x = salt_bcf (South Central Salt), y = henryhub ($/MMBtu),
    size/color = optional helpers, but keep it simple for CI determinism.
    """
    if df.empty:
        raise RuntimeError("No data to plot after filtering.")

    x = df["salt_bcf"]
    y = df["henryhub"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, alpha=0.7, edgecolors="none")
    ax.set_title("South Central Salt vs Henry Hub (Apr–Oct, last 5 completed years)")
    ax.set_xlabel("South Central Salt Storage (Bcf)")
    ax.set_ylabel("Henry Hub ($/MMBtu)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
