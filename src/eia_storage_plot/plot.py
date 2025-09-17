from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date


def select_apr_oct_last5(df: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    """
    Keep a continuous dataset for Apr–Oct for the last 5 *completed* years.
    Example (today=2025-09-17): keep 2020–2024, months 4..10 inclusive.
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
    return d.sort_values("period").reset_index(drop=True)


def _quad_fit(x: np.ndarray, y: np.ndarray):
    """
    Fit y ~ a x^2 + b x + c. Return coefficients, yhat, and R^2.
    """
    # Guard: need at least 3 unique x for a quadratic fit
    if len(np.unique(x)) < 3:
        # Fallback to a flat line through mean
        yhat = np.full_like(y, fill_value=np.nan, dtype=float)
        return (np.nan, np.nan, np.nan), yhat, np.nan

    coeffs = np.polyfit(x, y, deg=2)  # a, b, c
    yhat = np.polyval(coeffs, x)
    ss_res = np.nansum((y - yhat) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return coeffs, yhat, r2


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
    Make a scatter with a quadratic fit line and R^2 annotation.
    """
    if df.empty:
        raise RuntimeError("No data to plot after filtering.")

    # Drop NA rows for the two columns
    d = df[[x_col, y_col]].dropna().copy()
    if d.empty:
        raise RuntimeError("No overlapping non-null data points for plotting.")

    x = d[x_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)

    # Sort by x for a smooth fit curve
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    coeffs, yhat_sorted, r2 = _quad_fit(x_sorted, y_sorted)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, alpha=0.7, edgecolors="none", label="Weekly points")

    # Draw fitted curve if valid
    if not np.isnan(yhat_sorted).all():
        ax.plot(x_sorted, yhat_sorted, linewidth=2, label="Quadratic fit")

        a, b, c = coeffs
        eq = f"y = {a:.4g}x² + {b:.4g}x + {c:.4g}"
        r2txt = "R² = " + (f"{r2:.3f}" if not np.isnan(r2) else "n/a")
        ax.text(
            0.02, 0.98,
            f"{eq}\n{r2txt}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
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
        title="South Central Salt vs Henry Hub (Apr–Oct, last 5 completed years)",
        xlabel="South Central Salt Storage (Bcf)",
        ylabel="Henry Hub ($/MMBtu)",
        out_png=out_png,
    )


def make_scatter_us_total_vs_price(df: pd.DataFrame, out_png: str) -> None:
    _scatter_with_quadratic(
        df=df,
        x_col="us_bcf",
        y_col="henryhub",
        title="U.S. Total Storage vs Henry Hub (Apr–Oct, last 5 completed years)",
        xlabel="U.S. Total Working Gas (Bcf)",
        ylabel="Henry Hub ($/MMBtu)",
        out_png=out_png,
    )
