from __future__ import annotations
import os
import time
import io
import itertools
import warnings
import requests
import pandas as pd

# Quiet noisy date-parsing warnings from HTML/XLS fallbacks
warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually",
    category=UserWarning,
)

EIA_API_KEY = os.getenv("EIA_API_KEY", "")

# ---- EIA endpoints ----
HENRY_HUB_URL_V2 = "https://api.eia.gov/v2/natural-gas/pri/dpr/data/"  # v2 daily price
SERIES_URL_V1 = "https://api.eia.gov/series/?api_key={key}&series_id={sid}"

# v1 Series IDs
SID_SALT_WEEKLY     = "NG.W_EPG0_SSO_NUS_DW"   # South Central Salt weekly (Bcf)
SID_US_TOTAL_WEEK   = "NG.W_EPG0_SWO_NUS_DW"   # Lower 48 / U.S. Total weekly (Bcf)
SID_HENRY_DAILY     = "NG.RNGWHHD.D"           # Henry Hub daily spot ($/MMBtu)
SID_HENRY_WEEKLY    = "NG.RNGWHHD.W"           # Henry Hub weekly spot ($/MMBtu)

# ---- XLS fallbacks ----
XLS_SALT_WEEKLY     = "https://www.eia.gov/dnav/ng/hist_xls/NW2_EPG0_SSO_R33_BCFw.xls"
XLS_US_TOTAL_WEEK   = "https://www.eia.gov/dnav/ng/hist_xls/NW2_EPG0_SWO_R48_BCFw.xls"
XLS_HENRY_DAILY     = "https://www.eia.gov/dnav/ng/hist_xls/rngwhhdd.xls"
XLS_HENRY_WEEKLY    = "https://www.eia.gov/dnav/ng/hist_xls/rngwhhdw.xls"

# ---- HTML fallbacks ----
HTML_SALT_WEEKLY    = "https://www.eia.gov/dnav/ng/hist/nw2_epg0_sso_r33_bcfw.htm"
HTML_US_TOTAL_WEEK  = "https://www.eia.gov/dnav/ng/hist/nw2_epg0_swo_r48_bcfw.htm"
HTML_HENRY_DAILY    = "https://www.eia.gov/dnav/ng/hist/rngwhhdd.htm"
HTML_HENRY_WEEKLY   = "https://www.eia.gov/dnav/ng/hist/rngwhhdw.htm"


# =======================================================================
# HTTP helper
# =======================================================================

def _http_get(url: str, retries: int = 6, backoff_base: float = 1.7, stream: bool = False) -> requests.Response:
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


# =======================================================================
# Normalizers
# =======================================================================

def _df_from_v1_series(resp_json: dict) -> pd.DataFrame:
    series = resp_json.get("series", [])
    if not series:
        return pd.DataFrame(columns=["period", "value"])
    data = series[0].get("data", [])
    if not data:
        return pd.DataFrame(columns=["period", "value"])
    df = pd.DataFrame(data, columns=["period", "value"])
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["value"]  = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["period"]).sort_values("period").reset_index(drop=True)
    return df[["period", "value"]]


def _df_from_v2_price(resp_json: dict) -> pd.DataFrame:
    data = resp_json.get("response", {}).get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["period", "value"])
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["value"]  = pd.to_numeric(df.get("value"), errors="coerce")
    df = df.dropna(subset=["period", "value"]).sort_values("period").reset_index(drop=True)
    return df[["period", "value"]]


def _df_from_hist_xls(binary: bytes) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(binary), sheet_name=0, header=None, engine="xlrd")
    best = None
    ncols = raw.shape[1]
    for date_idx, val_idx in itertools.product(range(ncols), repeat=2):
        if date_idx == val_idx:
            continue
        d = pd.DataFrame({
            "period": pd.to_datetime(raw.iloc[:, date_idx], errors="coerce"),
            "value":  pd.to_numeric(raw.iloc[:, val_idx], errors="coerce")
        }).dropna(subset=["period", "value"])
        score = len(d)
        if score >= 10:
            d = d.sort_values("period").reset_index(drop=True)
            best = (score, d) if (best is None or score > best[0]) else best
    if best is None:
        d = pd.DataFrame({
            "period": pd.to_datetime(raw.iloc[:, 0], errors="coerce"),
            "value":  pd.to_numeric(raw.iloc[:, 1], errors="coerce")
        }).dropna(subset=["period", "value"]).sort_values("period").reset_index(drop=True)
        return d[["period", "value"]]
    return best[1][["period", "value"]]


def _df_from_hist_html(url: str) -> pd.DataFrame:
    tables = pd.read_html(url)
    best = None
    for t in tables:
        t = t.copy()
        t.columns = [str(c) for c in range(t.shape[1])]
        ncols = t.shape[1]
        for date_idx, val_idx in itertools.product(range(ncols), repeat=2):
            if date_idx == val_idx:
                continue
            d = pd.DataFrame({
                "period": pd.to_datetime(t.iloc[:, date_idx], errors="coerce"),
                "value":  pd.to_numeric(t.iloc[:, val_idx], errors="coerce")
            }).dropna(subset=["period", "value"])
            score = len(d)
            if score >= 10:
                d = d.sort_values("period").reset_index(drop=True)
                best = (score, d) if (best is None or score > best[0]) else best
    return pd.DataFrame(columns=["period", "value"]) if best is None else best[1][["period", "value"]]


def _clip(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    return df[(df["period"] >= s) & (df["period"] <= e)].copy()


# =======================================================================
# Fetchers + fallbacks
# =======================================================================

def _fetch_series_v1(series_id: str) -> pd.DataFrame:
    url = SERIES_URL_V1.format(key=EIA_API_KEY, sid=series_id)
    r = _http_get(url)
    return _df_from_v1_series(r.json())

def _fetch_hist_xls(url: str) -> pd.DataFrame:
    r = _http_get(url, stream=True)
    content = r.content if not r.raw.closed else r.raw.read()
    if not content:
        content = r.content
    return _df_from_hist_xls(content)

def _fetch_price_v2_daily(start: str, end: str) -> pd.DataFrame:
    params = (
        f"?api_key={EIA_API_KEY}"
        "&frequency=daily"
        "&sort[0][column]=period&sort[0][direction]=asc"
        "&data[0]=value"
        "&facets[series][]=Henry%20Hub%20Natural%20Gas%20Spot%20Price"
        f"&start={start}&end={end}"
    )
    url = HENRY_HUB_URL_V2 + params
    r = _http_get(url)
    return _df_from_v2_price(r.json())


def _try_chain(*funcs):
    last_err = None
    for fn in funcs:
        try:
            df = fn()
            if df is not None and not df.empty:
                return df
        except Exception as e:
            last_err = e
            print(f"[WARN] fallback step failed: {e}")
    if last_err:
        print(f"[WARN] all fallbacks failed; last error: {last_err}")
    return pd.DataFrame(columns=["period", "value"])


def fetch_salt_weekly(start: str, end: str) -> pd.DataFrame:
    df = _try_chain(
        lambda: _clip(_fetch_series_v1(SID_SALT_WEEKLY), start, end),
        lambda: _clip(_fetch_hist_xls(XLS_SALT_WEEKLY), start, end),
        lambda: _clip(_df_from_hist_html(HTML_SALT_WEEKLY), start, end),
    ).rename(columns={"value": "salt_bcf"})
    return df[["period", "salt_bcf"]]


def fetch_us_total_weekly(start: str, end: str) -> pd.DataFrame:
    df = _try_chain(
        lambda: _clip(_fetch_series_v1(SID_US_TOTAL_WEEK), start, end),
        lambda: _clip(_fetch_hist_xls(XLS_US_TOTAL_WEEK), start, end),
        lambda: _clip(_df_from_hist_html(HTML_US_TOTAL_WEEK), start, end),
    ).rename(columns={"value": "us_bcf"})
    return df[["period", "us_bcf"]]


def fetch_henry_hub_daily(start: str, end: str) -> pd.DataFrame:
    """
    Henry Hub daily ($/MMBtu): v2 → v1 → XLS → HTML.
    """
    df = _try_chain(
        lambda: _clip(_fetch_price_v2_daily(start, end), start, end),
        lambda: _clip(_fetch_series_v1(SID_HENRY_DAILY), start, end),
        lambda: _clip(_fetch_hist_xls(XLS_HENRY_DAILY), start, end),
        lambda: _clip(_df_from_hist_html(HTML_HENRY_DAILY), start, end),
    ).rename(columns={"value": "henryhub"})
    return df[["period", "henryhub"]]


def fetch_henry_hub_weekly(start: str, end: str) -> pd.DataFrame:
    """
    Henry Hub weekly ($/MMBtu): v1 → XLS → HTML.
    Used only if daily data is missing.
    """
    df = _try_chain(
        lambda: _clip(_fetch_series_v1(SID_HENRY_WEEKLY), start, end),
        lambda: _clip(_fetch_hist_xls(XLS_HENRY_WEEKLY), start, end),
        lambda: _clip(_df_from_hist_html(HTML_HENRY_WEEKLY), start, end),
    ).rename(columns={"value": "henryhub_w"})
    return df[["period", "henryhub_w"]]


# =======================================================================
# Public join (diagnostic + resilient)
# =======================================================================

def _daterange_summary(df: pd.DataFrame, col: str) -> str:
    if df.empty:
        return "empty"
    return f"{df[col].min().date()} → {df[col].max().date()} ({len(df)} rows)"

def build_weekly_join(start: str, end: str) -> pd.DataFrame:
    """
    Merge weekly SALT + U.S. Total with Henry Hub price.
    Prefer daily→weekly (W-FRI mean). If daily is unavailable, fall back to EIA weekly series.
    """
    if not EIA_API_KEY:
        print("[INFO] EIA_API_KEY not set or v2/v1 may be flaky; using fallbacks as needed.")

    salt = fetch_salt_weekly(start, end)
    us   = fetch_us_total_weekly(start, end)
    hh_d = fetch_henry_hub_daily(start, end)

    print(f"[INFO] SALT window: {_daterange_summary(salt, 'period')}")
    print(f"[INFO] US   window: {_daterange_summary(us, 'period')}")
    print(f"[INFO] HH-D window: {_daterange_summary(hh_d, 'period')}")

    # Build price weekly series:
    if not hh_d.empty:
        hh_w = (
            hh_d.set_index("period")
                .resample("W-FRI")
                .mean()
                .reset_index()
                .rename(columns={"henryhub": "price"})
        )
        price_source = "daily→weekly (v2/v1/XLS/HTML)"
    else:
        hh_w_series = fetch_henry_hub_weekly(start, end)
        print(f"[INFO] HH-W window: {_daterange_summary(hh_w_series, 'period')}")
        if hh_w_series.empty:
            raise RuntimeError(
                "Henry Hub price unavailable from all sources (daily and weekly). "
                "Re-run later or slightly widen the window."
            )
        hh_w = hh_w_series.rename(columns={"henryhub_w": "price"})
        price_source = "weekly (v1/XLS/HTML)"

    print(f"[INFO] Using Henry Hub price source: {price_source}")

    # Align by week-ending Friday
    merged = salt.merge(us, on="period", how="inner").merge(hh_w, on="period", how="inner")

    if merged.empty:
        msg = [
            "Merged dataset is empty after alignment.",
            f"SALT: {_daterange_summary(salt, 'period')}",
            f"US  : {_daterange_summary(us, 'period')}",
            f"HH  : {_daterange_summary(hh_w, 'period')} (source: {price_source})",
            f"Requested clip: {start} → {end}",
            "Tip: extend the window by ~1 week on either side; some sources lag a few days.",
        ]
        raise RuntimeError("\n".join(msg))

    # For downstream plotting, keep the expected column names
    merged = merged.rename(columns={"price": "henryhub"})
    return merged
