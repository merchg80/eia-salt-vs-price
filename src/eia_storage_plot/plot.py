from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

HIGHLIGHT_YEAR = 2026


def select_jun_nov_since_2015_including_current(df: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    if today is None:
        today = date.today()

    d = df.copy()
    d["year"] = d["period"].dt.year
    d["month"] = d["period"].dt.month

    mask = (d["year"].between(2015, today.year)) & (d["month"].between(6, 11))
    d = d.loc[mask].drop(columns=["year", "month"])

    return d.sort_values("period").reset_index(drop=True)


def _quad_fit_sorted(x_sorted: np.ndarray, y_sorted: np.ndarray):
    if len(np.unique(x_sorted)) < 3:
        yhat = np.full_like(y_sorted, np.nan, dtype=float)
        return (np.nan, np.nan, np.nan), yhat, np.nan

    coeffs = np.polyfit(x_sorted, y_sorted, deg=2)
    yhat = np.polyval(coeffs, x_sorted)

    ss_res = np.nansum((y_sorted - yhat) ** 2)
    ss_tot = np.nansum((y_sorted - np.nanmean(y_sorted)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    return tuple(coeffs), yhat, r2


def _short_date(dt: pd.Timestamp) -> str:
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
    d = df.dropna(subset=[x_col, y_col]).copy()
    if d.empty:
        raise RuntimeError("No data to plot after filtering.")

    d["year"] = d["period"].dt.year
    is_highlight = d["year"] == HIGHLIGHT_YEAR
    not_highlight = ~is_highlight

    x = d[x_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    coeffs, yhat_sorted, r2 = _quad_fit_sorted(x_sorted, y_sorted)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    if not_highlight.any():
        ax.scatter(
            d.loc[not_highlight, x_col],
            d.loc[not_highlight, y_col],
            alpha=0.75,
            edgecolors="none",
            label=f"Weeks (2015–{HIGHLIGHT_YEAR - 1})",
        )

    if is_highlight.any():
        ax.scatter(
            d.loc[is_highlight, x_col],
            d.loc[is_highlight, y_col],
            alpha=0.95,
            edgecolors="black",
            linewidths=0.4,
            label=f"Weeks ({HIGHLIGHT_YEAR})",
            color="yellow",
            zorder=5,
        )

    if not np.isnan(yhat_sorted).all():
        ax.plot(x_sorted, yhat_sorted, linewidth=2.2, label="Quadratic fit", zorder=4)

        a, b, c = coeffs
        eq = f"y = {a:.4g}x² + {b:.4g}x + {c:.4g}"
        r2txt = "R² = " + (f"{r2:.3f}" if not np.isnan(r2) else "n/a")

        ax.text(
            0.02,
            0.98,
            f"{eq}\n{r2txt}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        )

    recent_highlight = d.loc[is_highlight].sort_values("period").tail(5)
    for _, row in recent_highlight.iterrows():
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
        title="South Central Salt vs Henry Hub (Jun–Nov, 2015–present; 2026 highlighted)",
        xlabel="South Central Salt Storage (Bcf)",
        ylabel="Henry Hub ($/MMBtu)",
        out_png=out_png,
    )


def make_scatter_us_total_vs_price(df: pd.DataFrame, out_png: str) -> None:
    _scatter_with_quadratic(
        df=df,
        x_col="us_bcf",
        y_col="henryhub",
        title="U.S. Total Storage vs Henry Hub (Jun–Nov, 2015–present; 2026 highlighted)",
        xlabel="U.S. Total Working Gas (Bcf)",
        ylabel="Henry Hub ($/MMBtu)",
        out_png=out_png,
    )
