from __future__ import annotations

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
from datetime import date

from eia_storage_plot.fetch import build_weekly_join
from eia_storage_plot.plot import select_apr_oct_last5, make_scatter

def parse_args():
    p = argparse.ArgumentParser(description="Build Apr–Oct last-5-year EIA salt vs Henry Hub scatterplot.")
    p.add_argument("--season-last5", action="store_true", help="Use last 5 completed Apr–Oct seasons (default).")
    p.add_argument("--start", help="Override start YYYY-MM-DD (ignored when --season-last5).")
    p.add_argument("--end", help="Override end YYYY-MM-DD (ignored when --season-last5).")
    return p.parse_args()

def main():
    args = parse_args()
    today = date.today()

    if args.season_last5 or (not args.start and not args.end):
        end_year = today.year - 1
        start_year = end_year - 4
        start, end = f"{start_year}-04-01", f"{end_year}-10-31"
        print(f"[runner] Using last 5 completed Apr–Oct seasons: {start} → {end}")
        df = build_weekly_join(start, end)
        df = select_apr_oct_last5(df, today=today)
    else:
        start = args.start
        end = args.end or today.strftime("%Y-%m-%d")
        print(f"[runner] Using manual window: {start} → {end}")
        df = build_weekly_join(start, end)

    os.makedirs("out/data", exist_ok=True)
    os.makedirs("out/plots", exist_ok=True)
    df.to_csv("out/data/merged.csv", index=False)

    out_png = "out/plots/salt_vs_henryhub.png"
    make_scatter(df, out_png)

    os.makedirs("docs/plots", exist_ok=True)
    import shutil
    shutil.copyfile(out_png, "docs/plots/salt_vs_henryhub.png")

    print("Rows plotted:", len(df))

if __name__ == "__main__":
    main()
