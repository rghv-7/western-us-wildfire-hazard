#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_download_era5.py

Downloads ERA5 single-level fields from the Copernicus Climate Data Store
(CDS) for the Western United States fire-weather workflow.

Two requests per year (April-September), kept separate so each response is
a single NetCDF file instead of a ZIP with instant/accum parts:

  instant : t2m, d2m, u10, v10, swvl1 at 20:00 UTC only
            (about 13:00-14:00 local time in the Western US, close to the
            early-afternoon observation time the Fire Weather Index assumes)
  precip  : tp, all 24 hours
            (ERA5 tp is the accumulation over the hour ending at each
            valid time, so a 24 h total needs all 24 hourly values)

Output goes to data/raw/ and a request log (data/raw/download_log.jsonl)
records dataset, request body and retrieval date for the README.

Usage (run from anywhere):
    python 01_download_era5.py --start-year 2020 --end-year 2020   # test year
    python 01_download_era5.py                                     # 2001-2022
    python 01_download_era5.py --dry-run                           # list only

Existing, valid files are skipped, so the script can be re-run safely.
"""

import argparse
import json
import shutil
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOG_FILE = RAW_DIR / "download_log.jsonl"

LOG_LOCK = threading.Lock()


# =====================================================
# SETTINGS
# =====================================================

DATASET = "reanalysis-era5-single-levels"

# ERA5 format: [North, West, South, East]
# Covers CA, OR, WA, NV, ID, AZ (easternmost point is Idaho, about -111 deg).
AREA = [50, -125, 30, -108]

# April-September: April-May is spin-up for the FWI moisture codes,
# the analysis itself uses June-September (JJAS).
MONTHS = ["04", "05", "06", "07", "08", "09"]
DAYS = [f"{d:02d}" for d in range(1, 32)]

INSTANT_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "volumetric_soil_water_layer_1",
]
INSTANT_TIMES = ["20:00"]

PRECIP_VARIABLES = ["total_precipitation"]
PRECIP_TIMES = [f"{h:02d}:00" for h in range(24)]

DEFAULT_START_YEAR = 2001
DEFAULT_END_YEAR = 2022
DEFAULT_WORKERS = 3
MAX_RETRIES = 3


# =====================================================
# HELPERS
# =====================================================

def build_request(kind, year):
    """Return the CDS request body for one year and one kind."""
    if kind == "instant":
        variables, times = INSTANT_VARIABLES, INSTANT_TIMES
    elif kind == "precip":
        variables, times = PRECIP_VARIABLES, PRECIP_TIMES
    else:
        raise ValueError(f"Unknown kind: {kind}")

    return {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [str(year)],
        "month": MONTHS,
        "day": DAYS,
        "time": times,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def out_path(kind, year):
    return RAW_DIR / f"era5_{kind}_{year}_AprSep.nc"


def looks_like_netcdf(path):
    """True if the file starts with a NetCDF3 or NetCDF4/HDF5 signature."""
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with open(path, "rb") as f:
        head = f.read(8)
    return head[:3] == b"CDF" or head[:4] == b"\x89HDF"


def unwrap_zip_if_needed(path):
    """
    CDS sometimes returns a ZIP even when the name ends in .nc.
    If that happens and the ZIP holds exactly one NetCDF, unwrap it.
    """
    with open(path, "rb") as f:
        if f.read(2) != b"PK":
            return

    print(f"[WARNING] {path.name} is a ZIP archive, unwrapping")
    tmp_dir = path.with_suffix(".unzip_tmp")
    tmp_dir.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(path, "r") as z:
            members = [m for m in z.namelist() if m.endswith(".nc")]
            if len(members) != 1:
                raise RuntimeError(
                    f"{path.name}: expected 1 NetCDF in ZIP, found {members}"
                )
            z.extract(members[0], tmp_dir)
        shutil.move(str(tmp_dir / members[0]), str(path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def write_log(kind, year, request, outfile):
    entry = {
        "dataset": DATASET,
        "kind": kind,
        "year": year,
        "file": outfile.name,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request": request,
    }
    with LOG_LOCK:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def download_one(kind, year):
    """Download one file, with retries. Returns (kind, year, status)."""
    import cdsapi  # imported here so --dry-run works without cdsapi installed

    outfile = out_path(kind, year)

    if looks_like_netcdf(outfile):
        print(f"[INFO] Exists, skipping: {outfile.name}")
        return kind, year, "skipped"

    request = build_request(kind, year)
    partial = outfile.with_suffix(".nc.part")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[INFO] Requesting {outfile.name} (attempt {attempt})")
            client = cdsapi.Client()
            client.retrieve(DATASET, request, str(partial))

            unwrap_zip_if_needed(partial)
            if not looks_like_netcdf(partial):
                raise RuntimeError("Downloaded file is not a valid NetCDF")

            partial.replace(outfile)
            write_log(kind, year, request, outfile)
            print(f"[SUCCESS] {outfile.name}")
            return kind, year, "downloaded"

        except Exception as exc:  # noqa: BLE001 - report and retry
            print(f"[WARNING] {outfile.name}: {exc}")
            if partial.exists():
                partial.unlink()

    print(f"[FAILURE] {outfile.name} failed after {MAX_RETRIES} attempts")
    return kind, year, "failed"


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument(
        "--kind", choices=["instant", "precip", "all"], default="all"
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    kinds = ["instant", "precip"] if args.kind == "all" else [args.kind]
    years = range(args.start_year, args.end_year + 1)
    jobs = [(k, y) for y in years for k in kinds]

    print("=" * 60)
    print("ERA5 DOWNLOAD - April-September, Western United States")
    print("=" * 60)
    print(f"[INFO] Project root : {PROJECT_ROOT}")
    print(f"[INFO] Output dir   : {RAW_DIR}")
    print(f"[INFO] Area (N,W,S,E): {AREA}")
    print(f"[INFO] Years        : {args.start_year}-{args.end_year}")
    print(f"[INFO] Jobs         : {len(jobs)}")

    if args.dry_run:
        for kind, year in jobs:
            state = "exists" if looks_like_netcdf(out_path(kind, year)) else "to do"
            print(f"[INFO] {out_path(kind, year).name}  [{state}]")
        print("[INFO] Dry run only, nothing requested")
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, k, y) for k, y in jobs]
        for fut in as_completed(futures):
            results.append(fut.result())

    failed = [(k, y) for k, y, s in results if s == "failed"]
    done = sum(1 for _, _, s in results if s in ("downloaded", "skipped"))

    print("=" * 60)
    print(f"[INFO] Finished: {done}/{len(jobs)} files available")
    if failed:
        print(f"[FAILURE] Re-run to retry: {failed}")
        return 1
    print("[SUCCESS] All requested files are in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
