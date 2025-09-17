from __future__ import annotations
import os
import time
import requests
import pandas as pd

EIA_API_KEY = os.getenv("EIA_API_KEY", "")

# EIA API v2 endpoints
STORAGES_URL = "https://api.eia.gov/v2/natural-gas/storages/data/"
HENRY_HUB_URL = "https://api.eia.gov/v2/natural-gas/pri/dpr/data/"

# EIA “SeriesID bridge” (v2 wrapper around v1 series ids)
SERIESID_URL = "https://api.eia.gov/v2/seriesid/{sid}/?api_key={key}"

# Known series IDs (v1-style) for robust fallback
SERIES_SALT = "NG.W_EPG0_SSO_NUS_DW"  # South Central Salt, weekly, Bcf
SERIES_US_TOT = "NG.W_EPG0_SWO_NUS_DW"  # U.S. Total working gas, weekly, Bcf
SERIES_HENRYHUB_DAILY = "NG.RNGWHHD.D"  # Henry Hub spot, daily, $/MMBtu


def _fetch_json(url: str, retries: int = 5, backoff_base: float = 1.5) -> dict:
    """
    GET with retry on transient server issues (>=500) and rate limits (429).
    Exponential backoff: backoff_base ** attempt seconds.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            # Retry on 5xx and 429
            if r.status_code >= 500 or r.status_code == 429:
                wait = backoff_base ** attempt
                print(f"[EIA] HTTP {r.status_code}; retrying in {wait:.1f}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_exc = e
            wait = backoff_base ** attempt
            print(f"[EIA] RequestException: {e}; retrying in {wait:.1f}s (attempt {attempt+1}/{retries})")
            time.sleep(wait)
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown error fetching from EIA API")


def _df_from_v2(resp: dict) -> pd.DataFrame:
    data = resp.get("response", {}).get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df
    if "period" in df.columns:
        df["period"] = pd.to_datetime(df["period"])
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _df_from_seriesid(resp: dict) -> pd.DataFrame:
    """
    Normalize v2 /seriesid/ response into columns: period (datetime), value (float).
    """
    series = resp.get("response", {}).get("data", [])
    if not series:
        return pd.DataFrame(columns=["period", "value"])
    # /seriesid/ returns an array of observations with "period" and "value"
    df = pd.DataFrame(series)
    if "period" in df.columns:
        df["period"] = pd.to_datetime(df["period"])
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    # Ensure ascending date order
    if "period" in df.columns:
        df = df.sort_values("period").reset_index(drop=True)
    return df[["period", "value"]]


def _attempt_primary_then_fallback(primary_fn, fallback_fn, label: str) -> pd.DataFrame:
    """
    Try primary fetch; if empty or raises, try fallback.
    """
    try:
        df = primary_fn()
        if df is not None and not df.empty:
            print(f"[OK] {label}: primary endpoint returned {len(df)} rows")
            return df
        print(f"[WARN] {label}: primary endpoint returned no data; trying fallback")
    except Exception as e:
        print(f"[WARN] {label}: primary endpoint failed: {e}; trying fallback")
    # Fallback
    df_fb = fallback_fn()
    if df_fb is None or df_fb.empty:
        raise RuntimeError(f"{label}: both primary and fallback endpoints returned no data.")
    print(f"[OK] {label}: fallback endpoint returned {len(df_fb)} rows")
    return df_fb


# -------------------------
# SALT weekly (primary v2 storages + facets; fallback seriesid)
# -------------------------
def fetch_salt_weekly(start: str, end: str) -> pd.DataFrame:
    def primary():
        url = (
            f"{STORAGES_URL}?api_key={EIA_API_KEY}"
            "&frequency=weekly&sort[0][column]=period&sort[0][direction]=asc"
            "&data[0]=value"
            "&facets[region][]=South%20Central&facets[storageType][]=Salt"
            f"&start={start}&end={end}"
        )
        resp = _fetch_json(url)
        df = _df_from_v2(resp).rename(columns={"value": "salt_bcf"})
        return df[["period", "salt_bcf"]]

    def fallback():
        url = SERIESID_URL.format(sid=SERIES_SALT, key=EIA_API_KEY)
        resp = _fetch_json(url)
        df = _df_from_seriesid(resp).rename(columns={"value": "salt_bcf"})
        # clip to date range
        df = df[(df["period"] >= pd.to_datetime(start)) & (df["period"] <= pd.to_datetime(end))]
        return df[["period", "salt_bcf"]]

    return _attempt_primary_then_fallback(primary, fallback, "South Central SALT weekly")


# -------------------------
# U.S. Total weekly (primary v2 storages; fallback seriesid)
# -------------------------
def fetch_us_total_weekly(start: str, end: str) -> pd.DataFrame:
    def primary():
        url = (
            f"{STORAGES_URL}?api_key={EIA_API_KEY}"
            "&frequency=weekly&sort[0][column]=period&sort[0][direction]=asc"
            "&data[0]=value"
            "&facets[region][]=U.S.&facets[storageType][]=Total"
            f"&start={start}&end={end}"
        )
        resp = _fetch_json(url)
        df = _df_from_v2(resp).rename(columns={"value": "us_bcf"})
        return df[["period", "us_bcf"]]

    def fallback():
        url = SERIESID_URL.format(sid=SERIES_US_TOT, key=EIA_API_KEY)
        resp = _fetch_json(url)
        df = _df_from_seriesid(resp).rename(columns={"value": "us_bcf"})
        df = df[(df["period"] >= pd.to_datetime(start)) & (df["period"] <= pd.to_datetime(end))]
        return df[["period", "us_bcf"]]

    return _attempt_primary_then_fallback(primary, fallback, "U.S. TOTAL weekly")


# -------------------------
# Henry Hub daily (primary v2 daily price; fallback seriesid)
# -------------------------
def fetch_henry_hub_daily(start: str, end: str) -> pd.DataFrame:
    def primary():
        url = (
            f"{HENRY_HUB_URL}?api_key={EIA_API_KEY}"
            "&frequency=daily&sort[0][column]=period&sort[0][direction]=asc"
            "&data[0]=value"
            "&facets[series][]=Henry%20Hub%20Natural%20Gas%20Spot%20Price"
            f"&start={start}&end={end}"
        )
        resp = _fetch_json(url)
        df = _df_from_v2(resp).rename(columns={"value": "henryhub"})
        return df[["period", "henryhub"]]

    def fallback():
        url = SERIESID_URL.format(sid=SERIES_HENRYHUB_DAILY, key=EIA_API_KEY)
        resp = _fetch_json(url)
        df = _df_from_seriesid(resp).rename(columns={"value": "henryhub"})
        df = df[(df["period"] >= pd.to_datetime(start)) & (df["period"] <= pd.to_datetime(end))]
        return df[["period", "henryhub"]]

    return _attempt_primary_then_fallback(primary, fallback, "Henry Hub daily")


def build_weekly_join(start: str, end: str) -> pd.DataFrame:
    salt = fetch_salt_weekly(start, end)
    us = fetch_us_total_weekly(start, end)
    hh = fetch_henry_hub_daily(start, end)

    if salt.empty or us.empty or hh.empty:
        raise RuntimeError(
            "One or more EIA endpoints returned no data after retries/fallbacks. "
            "Check EIA_API_KEY, date range, and try again shortly."
        )

    # Weekly avg price aligned to Friday (EIA storage week ending)
    hh_w = (
        hh.set_index("period")
          .resample("W-FRI")
          .mean()
          .reset_index()
    )
    return salt.merge(us, on="period", how="inner").merge(hh_w, on="period", how="inner")
