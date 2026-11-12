#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_extract_and_validate.py

Validates the ERA5 files written by 01_download_era5.py before any science
is done with them. No extraction step is needed: 01 already unwraps ZIP
responses, so every file in data/raw is a plain NetCDF.

Checks per year:
  instant file (t2m, d2m, u10, v10, swvl1)
    - opens, expected variables present
    - one timestep per day, all at 20:00 UTC, Apr 1 - Sep 30 (183 days)
    - plausible value ranges and missing-value fraction
  precip file (tp)
    - opens, tp present
    - hourly time axis with no gaps (183 days x 24 h = 4392 steps)
    - tp >= 0 and plausible magnitude
    - accumulation semantics test: if tp is the accumulation over the
      hour ending at each valid time, the domain-mean series rises and
      falls freely; if it were a running total from forecast start, it
      would climb steadily and drop only at the reset hours.

Prints plain [INFO]/[SUCCESS]/[WARNING]/[FAILURE] lines and exits with 1
if any hard check fails, so the later scripts can rely on clean inputs.

Usage:
    python 02_extract_and_validate.py                  # all years found
    python 02_extract_and_validate.py --year 2020      # one year
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"


# =====================================================
# EXPECTATIONS
# =====================================================

N_DAYS = 30 + 31 + 30 + 31 + 31 + 30  # Apr-Sep = 183

INSTANT_VARS = {
    # name: (min, max, unit text)
    "t2m": (230.0, 330.0, "K"),
    "d2m": (200.0, 310.0, "K"),
    "u10": (-60.0, 60.0, "m/s"),
    "v10": (-60.0, 60.0, "m/s"),
    # ERA5 soil water can be very slightly negative (numerical artefact);
    # tolerate it here and clip to 0 in preprocessing.
    "swvl1": (-0.05, 1.0, "m3/m3"),
}

# ERA5 writes units in its own notation; treat these spellings as equal.
UNIT_ALIASES = {
    "K": {"K"},
    "m/s": {"m/s", "m s**-1", "m s-1", "m.s-1"},
    "m3/m3": {"m3/m3", "m**3 m**-3", "m3 m-3"},
}

# 0.15 m of rain in one hour is far beyond anything physical for ERA5
TP_MAX_HOURLY_M = 0.15


# =====================================================
# HELPERS
# =====================================================

class Report:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def info(self, msg):
        print(f"[INFO] {msg}")

    def ok(self, msg):
        print(f"[SUCCESS] {msg}")

    def warn(self, msg):
        self.warnings += 1
        print(f"[WARNING] {msg}")

    def fail(self, msg):
        self.failures += 1
        print(f"[FAILURE] {msg}")


def time_name(ds):
    for name in ("valid_time", "time"):
        if name in ds.coords or name in ds.dims:
            return name
    return None


def check_grid(ds, rep, label):
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    rep.info(
        f"{label}: grid {len(lat)} x {len(lon)}, "
        f"lat {lat.min():.2f}..{lat.max():.2f}, "
        f"lon {lon.min():.2f}..{lon.max():.2f}"
    )
    if "expver" in ds.dims:
        rep.warn(f"{label}: 'expver' is a dimension (ERA5T mixed in) - inspect")


def check_time_index(times, expected_hours, expected_steps, rep, label):
    """Common time checks. times is a pandas DatetimeIndex."""
    if len(times) != expected_steps:
        rep.fail(f"{label}: {len(times)} time steps, expected {expected_steps}")
    else:
        rep.ok(f"{label}: {len(times)} time steps")

    if times.min().month != 4 or times.min().day != 1:
        rep.fail(f"{label}: starts {times.min()}, expected 1 April")
    if times.max().month != 9 or times.max().day != 30:
        rep.fail(f"{label}: ends {times.max()}, expected 30 September")

    if not times.is_monotonic_increasing:
        rep.fail(f"{label}: time axis is not sorted")
    if times.has_duplicates:
        rep.fail(f"{label}: duplicate timestamps")

    bad_hours = sorted(set(times.hour) - set(expected_hours))
    if bad_hours:
        rep.fail(f"{label}: unexpected hours {bad_hours}")


def check_values(da, lo, hi, rep, label):
    values = da.values
    nan_frac = float(np.isnan(values).mean())
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    msg = f"{label}: min {vmin:.4g}, max {vmax:.4g}, missing {nan_frac:.2%}"
    if nan_frac > 0.0:
        rep.warn(msg + " (missing values present)")
    elif vmin < lo or vmax > hi:
        rep.fail(msg + f" - outside plausible range [{lo}, {hi}]")
    else:
        rep.ok(msg)
    if nan_frac == 0.0 and (vmin < lo or vmax > hi):
        return


def validate_instant(path, rep):
    label = path.name
    rep.info(f"--- {label}")
    ds = xr.open_dataset(path, engine="netcdf4")
    try:
        tname = time_name(ds)
        if tname is None:
            rep.fail(f"{label}: no time coordinate found")
            return

        check_grid(ds, rep, label)

        missing = [v for v in INSTANT_VARS if v not in ds.data_vars]
        if missing:
            rep.fail(f"{label}: missing variables {missing}; has {list(ds.data_vars)}")
        else:
            rep.ok(f"{label}: variables {list(ds.data_vars)}")

        times = pd.DatetimeIndex(ds[tname].values)
        check_time_index(times, [20], N_DAYS, rep, label)

        for var, (lo, hi, unit) in INSTANT_VARS.items():
            if var in ds.data_vars:
                units = ds[var].attrs.get("units", "?")
                if units not in UNIT_ALIASES[unit]:
                    rep.warn(f"{label}: {var} units '{units}', expected '{unit}'")
                check_values(ds[var], lo, hi, rep, f"{label} {var}")

        if "swvl1" in ds.data_vars:
            neg = float((ds["swvl1"] < 0).mean())
            if neg > 0:
                rep.info(
                    f"{label}: swvl1 < 0 in {neg:.3%} of values "
                    "(ERA5 artefact, will be clipped to 0 in preprocessing)"
                )

        if "t2m" in ds and "d2m" in ds:
            bad = float((ds["d2m"] > ds["t2m"] + 0.5).mean())
            if bad > 0.001:
                rep.warn(f"{label}: dewpoint > temperature in {bad:.2%} of cells")
            else:
                rep.ok(f"{label}: dewpoint <= temperature (consistent)")
    finally:
        ds.close()


def validate_precip(path, rep):
    label = path.name
    rep.info(f"--- {label}")
    ds = xr.open_dataset(path, engine="netcdf4")
    try:
        tname = time_name(ds)
        if tname is None:
            rep.fail(f"{label}: no time coordinate found")
            return

        check_grid(ds, rep, label)

        if "tp" not in ds.data_vars:
            rep.fail(f"{label}: 'tp' missing; has {list(ds.data_vars)}")
            return

        tp = ds["tp"]
        rep.info(f"{label}: tp attrs: {dict(tp.attrs)}")

        times = pd.DatetimeIndex(ds[tname].values)
        check_time_index(times, list(range(24)), N_DAYS * 24, rep, label)

        # hourly spacing, no gaps
        steps = np.unique(np.diff(times.values))
        if len(steps) != 1 or steps[0] != np.timedelta64(1, "h"):
            rep.fail(f"{label}: time steps are not uniformly 1 hour: {steps}")
        else:
            rep.ok(f"{label}: uniform 1-hour spacing, no gaps")

        values = tp.values
        nan_frac = float(np.isnan(values).mean())
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        msg = (
            f"{label} tp: min {vmin:.3g} m, max {vmax:.3g} m "
            f"({vmax * 1000:.1f} mm/h), missing {nan_frac:.2%}"
        )
        if vmin < -1e-6 or vmax > TP_MAX_HOURLY_M:
            rep.fail(msg + " - implausible")
        elif nan_frac > 0:
            rep.warn(msg)
        else:
            rep.ok(msg)

        # accumulation semantics: look at the domain-mean hourly series
        spatial_dims = [d for d in tp.dims if d != tname]
        mean_series = tp.mean(dim=spatial_dims).values
        diffs = np.diff(mean_series)
        frac_down = float((diffs < 0).mean())
        rep.info(
            f"{label}: domain-mean tp decreases in {frac_down:.0%} of hourly steps"
        )
        if frac_down >= 0.30:
            rep.ok(
                f"{label}: tp behaves like per-hour accumulation "
                "(rises and falls freely)"
            )
        else:
            rep.warn(
                f"{label}: tp rarely decreases - it may be a running total; "
                "do NOT sum it until this is checked"
            )

        # preview of the 24 h total, window 21 UTC (D-1) .. 20 UTC (D)
        shifted = pd.DatetimeIndex(times) + pd.Timedelta(hours=3)
        daily = (
            pd.Series(mean_series, index=shifted)
            .groupby(shifted.normalize())
            .agg(["sum", "count"])
        )
        full = daily[daily["count"] == 24]
        rep.info(
            f"{label}: {len(full)} complete 24 h windows (21..20 UTC), "
            f"domain-mean max {full['sum'].max() * 1000:.1f} mm/day; "
            f"first window has {int(daily['count'].iloc[0])} h (expected 21)"
        )
    finally:
        ds.close()


# =====================================================
# MAIN
# =====================================================

def find_years():
    years = set()
    for p in RAW_DIR.glob("era5_*_*_AprSep.nc"):
        m = re.match(r"era5_(instant|precip)_(\d{4})_AprSep\.nc", p.name)
        if m:
            years.add(int(m.group(2)))
    return sorted(years)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()

    rep = Report()

    print("=" * 60)
    print("ERA5 VALIDATION")
    print("=" * 60)
    print(f"[INFO] Raw dir: {RAW_DIR}")

    years = [args.year] if args.year else find_years()
    if not years:
        print("[FAILURE] No era5_*_AprSep.nc files found. Run 01 first.")
        return 1
    print(f"[INFO] Years to check: {years}")

    for year in years:
        print("=" * 60)
        print(f"[INFO] YEAR {year}")
        for kind, func in (("instant", validate_instant), ("precip", validate_precip)):
            path = RAW_DIR / f"era5_{kind}_{year}_AprSep.nc"
            if not path.exists():
                rep.fail(f"{path.name} not found")
                continue
            try:
                func(path, rep)
            except Exception as exc:  # noqa: BLE001
                rep.fail(f"{path.name}: could not be checked: {exc}")

    print("=" * 60)
    print(f"[INFO] Failures: {rep.failures}   Warnings: {rep.warnings}")
    if rep.failures:
        print("[FAILURE] Validation failed - fix before processing")
        return 1
    print("[SUCCESS] Validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
