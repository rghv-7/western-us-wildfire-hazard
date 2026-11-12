#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01b_download_cems_fwi.py

Downloads the Copernicus Emergency Management Service (CEMS) historical
fire danger indices (ERA5-driven, Canadian FWI system) used ONLY as an
independent benchmark for the FWI computed in this project.

IMPORTANT: this dataset (cems-fire-historical-v1) is hosted on the Early
Warning Data Store (EWDS), not the CDS. It needs:
  1. an EWDS account/token (profile page on https://ewds.climate.copernicus.eu)
  2. the dataset licence accepted on its EWDS dataset page
  3. the token in an environment variable (never commit it):
        Windows (PowerShell) :  $env:EWDS_API_KEY = "<your token>"
        WSL / Linux / macOS  :  export EWDS_API_KEY="<your token>"

If the request is rejected, open the dataset page on EWDS, tick the same
options (reanalysis, consolidated dataset, the variables, years, months,
area) and use "Show API request" to copy the exact field names into
build_request() below. Field names have changed between versions before.

Usage:
    python 01b_download_cems_fwi.py --start-year 2020 --end-year 2020
    python 01b_download_cems_fwi.py            # 2001-2022
    python 01b_download_cems_fwi.py --dry-run
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOG_FILE = RAW_DIR / "download_log.jsonl"


# =====================================================
# SETTINGS
# =====================================================

EWDS_URL = "https://ewds.climate.copernicus.eu/api"
DATASET = "cems-fire-historical-v1"

# [North, West, South, East] - same box as the ERA5 download
AREA = [50, -125, 30, -108]

MONTHS = ["04", "05", "06", "07", "08", "09"]
DAYS = [f"{d:02d}" for d in range(1, 32)]

# FWI plus the moisture codes, so the benchmark can check components too
VARIABLES = [
    "fire_weather_index",
    "fine_fuel_moisture_code",
    "duff_moisture_code",
    "drought_code",
    "initial_fire_spread_index",
    "build_up_index",
]

DEFAULT_START_YEAR = 2001
DEFAULT_END_YEAR = 2022


# =====================================================
# HELPERS
# =====================================================

def build_request(year):
    return {
        "product_type": "reanalysis",
        "variable": VARIABLES,
        "dataset_type": "consolidated_dataset",
        "system_version": ["4_1"],
        "year": [str(year)],
        "month": MONTHS,
        "day": DAYS,
        "grid": "0.25/0.25",
        "area": AREA,
        "data_format": "netcdf",
    }


def out_path(year):
    return RAW_DIR / f"cems_fwi_{year}_AprSep.nc"


def looks_like_netcdf(path):
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with open(path, "rb") as f:
        head = f.read(8)
    return head[:3] == b"CDF" or head[:4] == b"\x89HDF"


def write_log(year, request, outfile):
    entry = {
        "dataset": DATASET,
        "service": "EWDS",
        "year": year,
        "file": outfile.name,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request": request,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    years = list(range(args.start_year, args.end_year + 1))

    print("=" * 60)
    print("CEMS FIRE DANGER (EWDS) - benchmark download")
    print("=" * 60)
    print(f"[INFO] Output dir : {RAW_DIR}")
    print(f"[INFO] Years      : {args.start_year}-{args.end_year}")

    if args.dry_run:
        for y in years:
            state = "exists" if looks_like_netcdf(out_path(y)) else "to do"
            print(f"[INFO] {out_path(y).name}  [{state}]")
        print("[INFO] Dry run only, nothing requested")
        return 0

    key = os.environ.get("EWDS_API_KEY")
    if not key:
        print("[FAILURE] EWDS_API_KEY is not set (see the script header)")
        return 1

    import cdsapi  # imported here so --dry-run works without cdsapi installed

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(url=EWDS_URL, key=key)

    failed = []
    for year in years:
        outfile = out_path(year)
        if looks_like_netcdf(outfile):
            print(f"[INFO] Exists, skipping: {outfile.name}")
            continue

        request = build_request(year)
        partial = outfile.with_suffix(".nc.part")
        try:
            print(f"[INFO] Requesting {outfile.name}")
            client.retrieve(DATASET, request, str(partial))
            if not looks_like_netcdf(partial):
                raise RuntimeError(
                    "Downloaded file is not a plain NetCDF (ZIP?) - inspect it"
                )
            partial.replace(outfile)
            write_log(year, request, outfile)
            print(f"[SUCCESS] {outfile.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAILURE] {outfile.name}: {exc}")
            if partial.exists():
                partial.unlink()
            failed.append(year)

    print("=" * 60)
    if failed:
        print(f"[FAILURE] Failed years: {failed}")
        return 1
    print("[SUCCESS] CEMS benchmark files in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
