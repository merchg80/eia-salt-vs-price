from __future__ import annotations
import os
import time
import requests
import pandas as pd

EIA_API_KEY = os.getenv("EIA_API_KEY", "")

# Primary: EIA API v1 series endpoint (more stable when v2 hiccups)
SERIES_URL_V1 = "https://api.eia.gov/series/?api_key={key}&series_id={sid}"

# Backup (kept only as a last resort; we won't rely on it unless needed)
STORAGES_URL_V2 = "https://api.eia.gov/v2/natural-gas/storages/data/"
HENRY_HUB_URL_V2 = "https://api.eia.gov/v2/natural-gas/pri/dpr/data/"

# Series IDs (v1)
SID_SALT_WEEKLY   = "NG.W_EPG0_SSO_NUS_DW"   # South Central Salt, weekly, Bcf
SID_US_TOTAL_WEEK = "NG.W_EPG0_SWO_NUS_DW"   # U.S. Total working gas, weekly, Bcf
SID_HENRY_DAILY   = "NG.RNGWHHD.D"           # Henry Hub spot, daily, $/MMBtu


def _http_get(url: str, retries: int = 6, backoff_base: float = 1.7) -> requests.Response:
    """
    Robust GET with exponential backoff on 5xx/429 and network exceptions.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code >= 500 or r.status_code == 429:
                wait = backoff_base ** attempt
                print(f"[EIA] HTTP {r.status_code}; retrying in {wait:.1f}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            wait = backoff_base ** attempt
            print(f"[EIA] RequestException: {e}; retrying in {wait:.1f}s (attempt {attempt+1}/{retries})")
            time.sleep(wait)
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown HTTP error contacting EIA")


def _df_from_v1_series(resp_json: dict) -> pd.DataFrame:
    """
    Parse EIA v1 /series response into DataFrame with columns: period (datetime), value (float).
    Schema: {"series": [{"data": [["2025-09-12", 2.34], ...]}]}
    Dates may be weekly (YYYY-MM-DD) or daily (YYYY-MM-DD).
    """
    series = resp_json.get("series", [])
    if not series:
        return pd.DataFrame(columns=["period", "value"])
    # Take the first series object
    data = series[0].get("data", [])
    if not data:
        return pd.DataFrame(columns=["period", "value"])
    df = pd.DataFrame(data, columns=["period", "value"])
    # EIA returns most-recent first; sort ascending
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values("period").reset_index(drop=True)
    return df[["period", "value"]]


def _clip(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    return df[(df["period"] >= s) & (df["period"] <= e)].copy()


def _fetch_series_v1(series_id: str) -> pd.DataFrame:
    url = SERIES_URL_V1.format(key=EIA_API_KEY, sid=series_id)
    r = _http_get(url)
    return _df_from_v1_series(r.json())


# ---------- Public fetchers ----------

def fetch_salt_weekly(start: str, end: str) -> pd.DataFrame:
    """South Central 'Salt' weekly storage (Bcf) via v1 /series."""
    df = _fetch_series_v1(SID_SALT_WEEKLY).rename(columns={"value": "salt_bcf"})
    df = _clip(df, start, end)
    return df[["period", "salt_bcf"]]


def fetch_us_total_weekly(start: str, end: str) -> pd.DataFrame:
    """U.S. Total working gas weekly (Bcf) via v1 /series."""
    df = _fetch_series_v1(SID_US_TOTAL_WEEK).rename(columns={"value": "us_bcf"})
    df = _clip(df, start, end)
    return df[["period", "us_bcf"]]


def fetch_henry_hub_daily(start: str, end: str) -> pd.DataFrame:
    """Henry Hub daily spot price ($/MMBtu) via v1 /series."""
    df = _fetch_series_v1(SID_HENRY_DAILY).rename(columns={"value": "henryhub"})
    df = _clip(df, start, end)
    return df[["period", "henryhub"]]


def build_weekly_join(start: str, end: str) -> pd.DataFrame:
    """
    Merge weekly SALT + U.S. Total with Henry Hub daily (resampled to W-FRI).
    """
    if not EIA_API_KEY:
        raise RuntimeError("EIA_API_KEY is not set.")

    salt = fetch_salt_weekly(start, end)
    us   = fetch_us_total_weekly(start, end)
    hh   = fetch_henry_hub_daily(start, end)

    if salt.empty or us.empty or hh.empty:
        raise RuntimeError(
            "One or more datasets are empty after fetching from EIA v1 /series. "
            "Verify EIA_API_KEY and date range; try again if EIA is experiencing delays."
        )

    # Weekly avg Henry Hub aligned to Friday week-ending (to match storage 'period')
    hh_w = (
        hh.set_index("period")
          .resample("W-FRI")
          .mean()
          .reset_index()
    )

    df = salt.merge(us, on="period", how="inner").merge(hh_w, on="period", how="inner")
    return df
