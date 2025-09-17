from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from typing import Tuple


# ---------- Window selection: Jun–Nov from 2015 through current year (incl. 2025) ----------

def select_jun_nov_since_2015_including_current(df: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    """
    Keep Jun–Nov data from 2015 through the current year (inclusive).
    Example (today=2025-09-17): keep 2015–2025, months 6..11 inclusive.
    """
    if today is None:
        today = date.today()
    start_year = 2015
    end_year = today.year

    d = df.copy()
    d["year"] = d["period"].dt.year
    d["month"] = d["period"].dt.month
    mask = (d["year"].between(start_year, end_year)) & (d["month"].between(6, 11))
    d = d.loc[mask].drop(columns=["year", "month"])
    return d.sort_values("period").reset_index(drop=True)


# ---------- Quadratic fit helper ----------

def _quad_fit_sorted(x_sorted: np.ndarray, y_sorted: np.ndarray):
    """
    Fit y = a x^2 + b x + c on sorted x for a smooth curve.
    Returns (coeffs a,b,c), yhat_sorted, R^2.
    """
    if len(np.unique(x_sorted)) < 3:
        yhat = np.full_like(y_sorted, np.nan, dtype=float)
        return (np.nan, np.nan, np.nan), yhat, np.nan

    coeffs = np.polyfit(x_sorted, y_sorted, deg=2)  # a, b, c
    yhat = np.polyval(coeffs, x_sorted)
    ss_res = np.nansum((y_sorted - yhat) ** 2)
    ss_tot = np.nansum((y_sorted - np.nanmean(y_sorted)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return tuple(coeffs), yhat, r2


# ---------- Plotting with 2025 highlight + 2025-only annotations (no outlier trimming) ----------

def _short_date(dt: pd.Timestamp) -> str:
    # Short date like 9/17/25
    return f"{dt.month}/{dt.day}/{str(dt.year)[-2:]}"

def _scatter_with_quadratic(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_png: str,
) -> None:
    """
    Build scatter with:
      - NO outlier trimming
      - quadratic trend line + R^2
      - 2025 points colored yellow
      - annotate the 5 most-recent 2025 points with short dates
    """
    d = df.dropna(subset=[x_col, y_col]).copy()
    if d.empty:
        raise RuntimeError("No data to plot after filtering.")

    # Identify 2025 points
    d["year"] = d["period"].dt.year
    is_2025 = d["year"] == 2025
    not_2025 = ~is_2025

    # Prepare arrays for fit (use all available points)
    x = d[x_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    coeffs, yhat_sorted, r2 = _quad_fit_sorted(x_sorted, y_sorted)

    # Plot
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    # Non-2025 points
    if not_2025.any():
        ax.scatter(
            d.loc[not_2025, x_col],
            d.loc[not_2025, y_col],
            alpha=0.75,
            edgecolors="none",
            label="Weeks (≤ 2024)",
        )
    # 2025 in yellow on top
    if is_2025.any():
        ax.scatter(
            d.loc[is_2025, x_col],
            d.loc[is_2025, y_col],
            alpha=0.95,
            edgecolors="black",
            linewidths=0.4,
            label="Weeks (2025)",
            color="yellow",
            zorder=5,
        )

    # Fit curve
    if not np.isnan(yhat_sorted).all():
        ax.plot(x_sorted, yhat_sorted, linewidth=2.2, label="Quadratic fit", zorder=4)
        a, b, c = coeffs
        eq = f"y = {a:.4g}x² + {b:.4g}x + {c:.4g}"
        r2txt = "R² = " + (f"{r2:.3f}" if not np.isnan(r2) else "n/a")
        ax.text(
            0.02, 0.98,
            f"{eq}\n{r2txt}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        )

    # Annotate the 5 most-recent 2025 points
    recent_2025 = d.loc[is_2025].sort_values("period").tail(5)
    for _, row in recent_2025.iterrows():
        ax.annotate(
            _short_date(row["period"]),
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def make_scatter_salt_vs_price(df: pd.DataFrame, out_png: str) -> None:
    _scatter_with_quadratic(
        df=df,
        x_col="salt_bcf",
        y_col="henryhub",
        title="South Central Salt vs Henry Hub (Jun–Nov, 2015–present; no outlier trim)",
        xlabel="South Central Salt Storage (Bcf)",
        ylabel="Henry Hub ($/MMBtu)",
        out_png=out_png,
    )


def make_scatter_us_total_vs_price(df: pd.DataFrame, out_png: str) -> None:
    _scatter_with_quadratic(
        df=df,
        x_col="us_bcf",
        y_col="henryhub",
        title="U.S. Total Storage vs Henry Hub (Jun–Nov, 2015–present; no outlier trim)",
        xlabel="U.S. Total Working Gas (Bcf)",
        ylabel="Henry Hub ($/MMBtu)",
        out_png=out_png,
    )
