from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

def filter_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Aug–Oct 2024 and Aug–Sep 2025 (to date)."""
    d = df.copy()
    d["year"] = d["period"].dt.year
    d["month"] = d["period"].dt.month
    mask_2024 = d["year"].eq(2024) & d["month"].isin([8, 9, 10])
    mask_2025 = d["year"].eq(2025) & d["month"].isin([8, 9])
    return d[mask_2024 | mask_2025].copy()

def make_scatter(df: pd.DataFrame, out_png: str, title: str | None = None) -> None:
    if df.empty:
        raise ValueError("No data to plot")
    plt.figure(figsize=(9,7))
    # X: weekly avg Henry Hub, Y: Salt; color = U.S. total (no explicit colors set)
    sc = plt.scatter(df["henryhub"], df["salt_bcf"], c=df["us_bcf"], alpha=0.85)
    plt.colorbar(sc, label="Total U.S. Working Gas (Bcf)")
    plt.xlabel("Henry Hub price (weekly avg, $/MMBtu)")
    plt.ylabel("South Central SALT working gas (Bcf)")
    plt.title(title or "Salt Storage vs Henry Hub (Aug–Oct 2024 vs Aug–Sep 2025)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
