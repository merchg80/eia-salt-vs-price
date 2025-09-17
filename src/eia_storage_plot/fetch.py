from __future__ import annotations
import os
import time
import requests
import pandas as pd

EIA_API_KEY = os.getenv("EIA_API_KEY", "")

# EIA API v2 endpoints
STORAGES_URL = "https://api.eia.gov/v2/natural-gas/storages/data/"
HENRY_HUB_URL = "https://api.eia.gov/v2/natural-gas/pri/dpr/data/"

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
            # For network hiccups, also backoff
            wait = backoff_base ** attempt
            print(f"[EIA] RequestException: {e}; retrying in {wait:.1f}s (attempt {attempt+1}/{retries})")
            time.sleep(wait)
    # Exhausted retries
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

def fetch_salt_weekly(start: str, end: str) -> pd.DataFrame:
    """South Central 'Salt' weekly storage (Bcf)."""
    url = (
        f"{STORAGES_URL}?api_key={EIA_API_KEY}"
        "&frequency=weekly&sort[0][column]=period&sort[0][direction]=asc"
        "&data[0]=value"
        "&facets[region][]=South%20Central&facets[storageType][]=Salt"
        f"&start={start}&end={end}"
    )
    df = _df_from_v2(_fetch_json(url)).rename(columns={"value": "salt_bcf"})
    return df[["period", "salt_bcf"]]

def fetch_us_total_weekly(start: str, end: str) -> pd.DataFrame:
    """U.S. Total working gas weekly (Bcf)."""
    url = (
        f"{STORAGES_URL}?api_key={EIA_API_KEY}"
        "&frequency=weekly&sort[0][column]=period&sort[0][direction]=asc"
        "&data[0]=value"
        "&facets[region][]=U.S.&facets[storageType][]=Total"
        f"&start={start}&end={end}"
    )
    df = _df_from_v2(_fetch_json(url)).rename(columns={"value": "us_bcf"})
    return df[["period", "us_bcf"]]

def fetch_henry_hub_daily(start: str, end: str) -> pd.DataFrame:
    """Henry Hub daily price (USD/MMBtu), to be resampled weekly."""
    url = (
        f"{HENRY_HUB_URL}?api_key={EIA_API_KEY}"
        "&frequency=daily&sort[0][column]=period&sort[0][direction]=asc"
        "&data[0]=value"
        "&facets[series][]=Henry%20Hub%20Natural%20Gas%20Spot%20Price"
        f"&start={start}&end={end}"
    )
    df = _df_from_v2(_fetch_json(url)).rename(columns={"value": "henryhub"})
    return df[["period", "henryhub"]]

def build_weekly_join(start: str, end: str) -> pd.DataFrame:
    salt = fetch_salt_weekly(start, end)
    us = fetch_us_total_weekly(start, end)
    hh = fetch_henry_hub_daily(start, end)
    if salt.empty or us.empty or hh.empty:
        raise RuntimeError(
            "One or more EIA endpoints returned no data after retries. "
            "Check EIA_API_KEY, date range, and try again shortly (EIA may be temporarily unavailable)."
        )
    # Weekly avg price aligned to Friday (EIA storage week ending)
    hh_w = (
        hh.set_index("period")
          .resample("W-FRI")
          .mean()
          .reset_index()
    )
    return salt.merge(us, on="period", how="inner").merge(hh_w, on="period", how="inner")
