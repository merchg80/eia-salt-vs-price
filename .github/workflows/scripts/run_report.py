from __future__ import annotations

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from datetime import date

from eia_storage_plot.fetch import build_weekly_join
from eia_storage_plot.plot import (
    select_apr_oct_last10_including_current,
    make_scatter_salt_vs_price,
    make_scatter_us_total_vs_price,
)

def main():
    today = date.today()
    # Last 10 years window INCLUDING current year
    start_year = today.year - 9
    end_year = today.year
    start, end = f"{start_year}-04-01", f"{end_year}-10-31"
    print(f"[runner] Using Apr–Oct last 10 years incl. current: {start} → {end}")

    df = build_weekly_join(start, end)
    df = select_apr_oct_last10_including_current(df, today=today)

    os.makedirs("out/data", exist_ok=True)
    os.makedirs("out/plots", exist_ok=True)
    df.to_csv("out/data/merged.csv", index=False)

    # Plots
    salt_png = "out/plots/salt_vs_henryhub.png"
    us_png   = "out/plots/us_total_vs_henryhub.png"

    make_scatter_salt_vs_price(df, salt_png, today=today)
    make_scatter_us_total_vs_price(df, us_png, today=today)

    os.makedirs("docs/plots", exist_ok=True)
    import shutil
    shutil.copyfile(salt_png, "docs/plots/salt_vs_henryhub.png")
    shutil.copyfile(us_png, "docs/plots/us_total_vs_henryhub.png")

    print("Rows plotted:", len(df))

if __name__ == "__main__":
    main()
