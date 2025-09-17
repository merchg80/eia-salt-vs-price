from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from typing import Tuple


# ---------- Window selection: last 5 years incl. current (e.g., includes 2025) ----------

def select_apr_oct_last5_including_current(df: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    """
    Keep a continuous dataset for Apr–Oct for the last 5 years, INCLUDING the current year.
    Example (today=2025-09-17): keep 2021–2025, months 4..10 inclusive.
    """
    if today is None:
        today = date.today()
    end_year = today.year            # include current year
    start_year = end_year - 4        # last 5 years inclusive

    d = df.copy()
    d["year"] = d["period"].dt.year
    d["month"] = d["period"].dt.month
    mask = (d["year"].between(start_year, end_year)) & (d["month"].between(4, 10))
    d = d.loc[mask].drop(columns=["year", "month"])
    return d.sort_values("period").reset_index(drop=True)


# ---------- Robust outlier trimming (bivariate) ----------

def _mad(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med)) or 1.0

def trim_outliers_bivariate(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    frac: float = 0.20,
) -> pd.DataFrame:
    """
    Remove the top `frac` of points farthest from the robust center in standardized space.
    Steps:
      - Robust center = median(x), median(y)
      - Scale: MAD for x and y
      - Distance = sqrt(zx^2 + zy^2)
      - Drop the largest `frac` quantile by distance
    """
    if df.empty or frac <= 0:
        return df

    d = df[[x_col, y_col]].to_numpy(dtype=float)
    x = d[:, 0]
    y = d[:, 1]

    xm, ym = np.nanmedian(x), np.nanmedian(y)
    xmad, ymad = _mad(x), _mad(y)
    zx = (x - xm) / (1.4826 * xmad)
    zy = (y - ym) / (1.4826 * ymad)
    dist = np.sqrt(zx ** 2 + zy ** 2)

    q = np.nanquantile(dist, 1.0 - frac)
    keep = dist <= q
    return df.loc[keep].copy()


# ---------- Quadratic fit helper ----------

def _quad_fit_sorted(x_sorted: np.ndarray, y_sorted: np.ndarray) -> Tuple[Tuple[float, float, float], np.ndarray, float]:
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


# ---------- Plotting with 2025 highlight + annotations ----------

def _short_date(dt: pd.Timestamp) -> str:
    # Cross-platform short date like 9/17/25
    return f"{dt.month}/{dt.day}/{str(dt.year)[-2:]}"

def _scatter_with_quadratic_and_options(
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
      - 20% robust outlier removal (bivariate)
      - quadratic trend line + R^2
      - 2025 points colored yellow
      - annotate the 5 most-recent (by period) NON-2025 points with short dates
    """
    if df.empty:
        raise RuntimeError("No data to plot after filtering.")

    # Trim 20% outliers (bivariate)
    trimmed = trim_outliers_bivariate(df.dropna(subset=[x_col, y_col]), x_col, y_col, frac=0.20)
    if trimmed.empty:
        raise RuntimeError("All points were removed by outlier trimming; cannot plot.")

    # Split 2025 vs others (after trimming)
    trimmed = trimmed.copy()
    trimmed["year"] = trimmed["period"].dt.year
    is_2025 = trimmed["year"] == 2025
    not_2025 = ~is_2025

    # Prepare arrays for fit (use all remaining points)
    x = trimmed[x_col].to_numpy(dtype=float)
    y = trimmed[y_col].to_numpy(dtype=float)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    coeffs, yhat_sorted, r2 = _quad_fit_sorted(x_sorted, y_sorted)

    # Plot
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    # Non-2025 points
    ax.scatter(
        trimmed.loc[not_2025, x_col],
        trimmed.loc[not_2025, y_col],
        alpha=0.75,
        edgecolors="none",
        label="Weeks (non-2025)",
    )
    # 2025 in yellow on top
    if is_2025.any():
        ax.scatter(
            trimmed.loc[is_2025, x_col],
            trimmed.loc[is_2025, y_col],
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

    # Annotate the 5 most-recent NON-2025 points AFTER trimming
    recent_non_2025 = trimmed.loc[not_2025].sort_values("period").tail(5)
    for _, row in recent_non_2025.iterrows():
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
    _scatter_with_quadratic_and_options(
        df=df,
        x_col="salt_bcf",
        y_col="henryhub",
        title="South Central Salt vs Henry Hub (Apr–Oct, last 5 years incl. 2025; 20% outliers trimmed)",
        xlabel="South Central Salt Storage (Bcf)",
        ylabel="Henry Hub ($/MMBtu)",
        out_png=out_png,
    )


def make_scatter_us_total_vs_price(df: pd.DataFrame, out_png: str) -> None:
    _scatter_with_quadratic_and_options(
        df=df,
        x_col="us_bcf",
        y_col="henryhub",
        title="U.S. Total Storage vs Henry Hub (Apr–Oct, last 5 years incl. 2025; 20% outliers trimmed)",
        xlabel="U.S. Total Working Gas (Bcf)",
        ylabel="Henry Hub ($/MMBtu)",
        out_png=out_png,
    )
