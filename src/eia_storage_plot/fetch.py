from __future__ import annotations
import os
import time
import io
import requests
import pandas as pd

EIA_API_KEY = os.getenv("EIA_API_KEY", "")

# ---- API endpoints ----
SERIES_URL_V1 = "https://api.eia.gov/series/?api_key={key}&series_id={sid}"

# v1 Series IDs
SID_SALT_WEEKLY   = "NG.W_EPG0_SSO_NUS_DW"   # (sometimes 404s / temporarily unavailable)
SID_US_TOTAL_WEEK = "NG.W_EPG0_SWO_NUS_DW"
SID_HENRY_DAILY   = "NG.RNGWHHD.D"

# ---- Direct XLS fallbacks (same data, from EIA "View History → Download Data (XLS)") ----
# Salt (South Central): page https://www.eia.gov/dnav/ng/hist/nw2_epg0_sso_r33_bcfw.htm
XLS_SALT_WEEKLY   = "https://www.eia.gov/dnav/ng/hist_xls/NW2_EPG0_SSO_R33_BCFw.xls"
# Lower 48 Total: page https://www.eia.gov/dnav/ng/hist/nw2_epg0_swo_r48_bcfw.htm
XLS_US_TOTAL_WEEK = "https://www.eia.gov/dnav/ng/hist_xls/NW2_EPG0_SWO_R48_BCFw.xls"
# Henry Hub daily: page https://www.eia.gov/dnav/ng/hist/rngwhhdd.htm
XLS_HENRY_DAILY   = "https://www.eia.gov/dnav/ng/hist_xls/rngwhhdd.xls"


# =======================================================================================
# HTTP helpers
# =======================================================================================

def _http_get(url: str, retries: int = 6, backoff_base: float = 1.7, stream: bool = False) -> requests.Response:
    """
    Robust GET with exponential backoff on 5xx/429 and network exceptions.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60, stream=stream)
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


# =======================================================================================
# Normalizers
# =======================================================================================

def _df_from_v1_series(resp_json: dict) -> pd.DataFrame:
    """
    Parse EIA v1 /series response into DataFrame with columns: period (datetime), value (float).
    Schema: {"series": [{"data": [["2025-09-12", 2.34], ...]}]}
    """
    series = resp_json.get("series", [])
    if not series:
        return pd.DataFrame(columns=["period", "value"])
    data = series[0].get("data", [])
    if not data:
        return pd.DataFrame(columns=["period", "value"])
    df = pd.DataFrame(data, columns=["period", "value"])
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["period"]).sort_values("period").reset_index(drop=True)
    return df[["period", "value"]]


def _df_from_hist_xls(binary: bytes) -> pd.DataFrame:
    """
    Many EIA 'hist_xls' files are simple two-column sheets (Date, Value) or have header rows before data.
    This parser:
      1) loads the first sheet without trusting headers,
      2) keeps the first two columns,
      3) coerces the first to datetime and the second to numeric,
      4) drops non-date rows, sorts ascending.
    """
    with io.BytesIO(binary) as buf:
        raw = pd.read_excel(buf, sheet_name=0, header=None)
    if raw.shape[1] < 2:
        # sometimes data starts later; try all columns then pick first 2 that parse cleanly
        raw = pd.read_excel(io.BytesIO(binary), sheet_name=0, header=None)
    # Take first two columns
    df = raw.iloc[:, :2].copy()
    df.columns = ["period_raw", "value_raw"]
    # Coerce
    df["period"] = pd.to_datetime(df["period_raw"], errors="coerce")
    df["value"] = pd.to_numeric(df["value_raw"], errors="coerce")
    df = df.dropna(subset=["period"]).dropna(subset=["value"])
    df = df[["period", "value"]].sort_values("period").reset_index(drop=True)
    return df


def _clip(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    return df[(df["period"] >= s) & (df["period"] <= e)].copy()


# =======================================================================================
# Fetchers with layered fallbacks: v1 /series  → hist_xls
# =======================================================================================

def _fetch_series_v1(series_id: str) -> pd.DataFrame:
    url = SERIES_URL_V1.format(key=EIA_API_KEY, sid=series_id)
    r = _http_get(url)
    return _df_from_v1_series(r.json())

def _fetch_hist_xls(url: str) -> pd.DataFrame:
    r = _http_get(url, stream=True)
    # Some servers require full read after streaming
    content = r.content if not r.raw.closed else r.raw.read()
    if not content:
        content = r.content
    return _df_from_hist_xls(content)

def _try_v1_then_xls(series_id: str, xls_url: str, label: str) -> pd.DataFrame:
    # Try v1
    try:
        df = _fetch_series_v1(series_id)
        if not df.empty:
            print(f"[OK] {label}: v1 /series returned {len(df)} rows")
            return df
        print(f"[WARN] {label}: v1 /series returned no rows; falling back to XLS")
    except Exception as e:
        print(f"[WARN] {label}: v1 /series failed: {e}; falling back to XLS")

    # Fallback to XLS
    df_x = _fetch_hist_xls(xls_url)
    if df_x.empty:
        raise RuntimeError(f"{label}: XLS fallback returned no rows")
    print(f"[OK] {label}: XLS fallback returned {len(df_x)} rows")
    return df_x


def fetch_salt_weekly(start: str, end: str) -> pd.DataFrame:
    """
    South Central 'Salt' weekly storage (Bcf).
    """
    df = _try_v1_then_xls(SID_SALT_WEEKLY, XLS_SALT_WEEKLY, "South Central SALT weekly")
    df = _clip(df, start, end).rename(columns={"value": "salt_bcf"})
    return df[["period", "salt_bcf"]]

def fetch_us_total_weekly(start: str, end: str) -> pd.DataFrame:
    """
    Lower 48 / U.S. Total weekly storage (Bcf). (The dnav page labels it "Lower 48 States", same time series used in WNGSR totals.)
    """
    df = _try_v1_then_xls(SID_US_TOTAL_WEEK, XLS_US_TOTAL_WEEK, "U.S. TOTAL weekly")
    df = _clip(df, start, end).rename(columns={"value": "us_bcf"})
    return df[["period", "us_bcf"]]

def fetch_henry_hub_daily(start: str, end: str) -> pd.DataFrame:
    """
    Henry Hub daily spot price ($/MMBtu).
    """
    df = _try_v1_then_xls(SID_HENRY_DAILY, XLS_HENRY_DAILY, "Henry Hub daily")
    df = _clip(df, start, end).rename(columns={"value": "henryhub"})
    return df[["period", "henryhub"]]


# =======================================================================================
# Public join
# =======================================================================================

def build_weekly_join(start: str, end: str) -> pd.DataFrame:
    """
    Merge weekly SALT + U.S. Total with Henry Hub daily (resampled to W-FRI).
    """
    # API key is only needed for v1 /series; XLS fallbacks work without it.
    if not EIA_API_KEY:
        print("[INFO] EIA_API_KEY not set; will use XLS fallbacks as needed.")

    salt = fetch_salt_weekly(start, end)
    us   = fetch_us_total_weekly(start, end)
    hh   = fetch_henry_hub_daily(start, end)

    if salt.empty or us.empty or hh.empty:
        raise RuntimeError(
            "One or more datasets are empty after all fallbacks. "
            "Please retry; EIA may be temporarily unavailable."
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
