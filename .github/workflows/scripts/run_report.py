from __future__ import annotations

# --- make sure src/ is on the path when running in Actions or locally ---
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
from datetime import date
import pandas as pd  # noqa: F401 (handy if you add debugging/printing)

from eia_storage_plot.fetch import build_weekly_join
from eia_storage_plot.plot import filter_windows, make_scatter

DEFAULT_START = "2024-08-01"
DEFAULT_END = date.today().strftime("%Y-%m-%d")

def parse_args():
    p = argparse.ArgumentParser(description="Build EIA salt vs Henry Hub weekly scatterplot.")
    p.add_argument("--start", default=DEFAULT_START, help="YYYY-MM-DD inclusive (default: 2024-08-01)")
    p.add_argument("--end", default=DEFAULT_END, help="YYYY-MM-DD inclusive (default: today)")
    p.add_argument("--all-data", action="store_true", help="Skip month filtering; plot all returned weeks.")
    return p.parse_args()

def main():
    args = parse_args()
    start, end = args.start, args.end
    df = build_weekly_join(start, end)

    os.makedirs("out/data", exist_ok=True)
    os.makedirs("out/plots", exist_ok=True)
    df.to_csv("out/data/merged.csv", index=False)

    plot_df = df if args.all_data else filter_windows(df)

    out_png = "out/plots/salt_vs_henryhub.png"
    make_scatter(plot_df, out_png)
    os.makedirs("docs/plots", exist_ok=True)

    import shutil
    shutil.copyfile(out_png, "docs/plots/salt_vs_henryhub.png")

    print("Rows merged:", len(df))
    print("Rows plotted:", len(plot_df))

if __name__ == "__main__":
    main()
