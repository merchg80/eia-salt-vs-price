from __future__ import annotations
import os, requests, pandas as pd

EIA_API_KEY = os.getenv("EIA_API_KEY", "")
STORAGES_URL = "https://api.eia.gov/v2/natural-gas/storages/data/"
HENRY_HUB_URL = "https://api.eia.gov/v2/natural-gas/pri/dpr/data/"

def _fetch_json(url: str) -> dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

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
        raise RuntimeError("EIA endpoints returned no data for the given range.")

    hh_w = (
        hh.set_index("period")
          .resample("W-FRI")
          .mean()
          .reset_index()
    )

    return salt.merge(us, on="period", how="inner").merge(hh_w, on="period", how="inner")
