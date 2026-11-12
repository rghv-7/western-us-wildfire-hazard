#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_compute_fwi.py

Computes the Canadian Fire Weather Index (FWI) System from ERA5 for each
year, April-September, over the Western United States.

Steps per year
  1. Read the ERA5 files from data/raw (written by 01, checked by 02).
  2. Derive the daily FWI inputs at 20:00 UTC (early afternoon local time):
       temperature (deg C)       t2m - 273.15
       relative humidity (%)     Magnus formula from t2m and d2m
       wind speed (km/h)         sqrt(u10^2 + v10^2) * 3.6
       24 h rain (mm)            sum of hourly tp over 21:00 UTC (day D-1)
                                 to 20:00 UTC (day D), converted m -> mm
  3. Run the FWI moisture codes day by day from 1 April
       FFMC (fine fuels), DMC (duff), DC (deep organic layer),
       then ISI, BUI and FWI.
  4. Save everything to data/processed/fwi_<year>.nc

Method references
  Van Wagner, C.E. and Pickett, T.L. (1985) Equations and FORTRAN program
  for the Canadian Forest Fire Weather Index System. Forestry Technical
  Report 33, Canadian Forestry Service.
  Standard start-up values are used on 1 April: FFMC 85, DMC 6, DC 15.
  April and May act as spin-up; analysis later uses June-September.

Limitations
  - ERA5 0.25 degree grid (about 28 km), one afternoon observation per day.
  - 1 April rain window has only 21 h (no data before 00 UTC on 1 April).
  - The FWI was developed for Canadian forests. Here it is a weather-based
    hazard indicator, not a validated US fire-danger rating, and it says
    nothing about ignition, fuels or fire spread.
  - Dewpoint-based RH uses the Magnus approximation.

Usage:
    python 03_compute_fwi.py --selftest        # check the FWI maths only
    python 03_compute_fwi.py --year 2020       # one year
    python 03_compute_fwi.py                   # all years found
    python 03_compute_fwi.py --force           # recompute existing outputs
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# =====================================================
# FWI CONSTANTS
# =====================================================

# Standard start-up values (Van Wagner & Pickett 1985)
FFMC_START = 85.0
DMC_START = 6.0
DC_START = 15.0

# Day-length adjustment factors, January..December (Northern Hemisphere)
DMC_DAYLENGTH = np.array(
    [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.2, 8.0, 7.0, 6.0]
)
DC_DAYLENGTH = np.array(
    [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
)


# =====================================================
# FWI SYSTEM (vectorised: works on scalars or arrays)
# =====================================================

def ffmc_step(ffmc0, T, H, W, rain):
    """Fine Fuel Moisture Code. T in C, H in %, W in km/h, rain in mm."""
    H = np.clip(H, 0.0, 100.0)

    mo = 147.2 * (101.0 - ffmc0) / (59.5 + ffmc0)

    # rain phase
    wet = rain > 0.5
    rf = np.where(wet, rain - 0.5, 1.0)
    mr = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (
        1.0 - np.exp(-6.93 / rf)
    )
    mr = np.where(mo > 150.0, mr + 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf), mr)
    mr = np.minimum(mr, 250.0)
    mo = np.where(wet, mr, mo)

    # equilibrium moisture contents (drying and wetting)
    ed = (
        0.942 * H ** 0.679
        + 11.0 * np.exp((H - 100.0) / 10.0)
        + 0.18 * (21.1 - T) * (1.0 - np.exp(-0.115 * H))
    )
    ew = (
        0.618 * H ** 0.753
        + 10.0 * np.exp((H - 100.0) / 10.0)
        + 0.18 * (21.1 - T) * (1.0 - np.exp(-0.115 * H))
    )

    # drying
    ko = 0.424 * (1.0 - (H / 100.0) ** 1.7) + 0.0694 * np.sqrt(W) * (
        1.0 - (H / 100.0) ** 8
    )
    kd = ko * 0.581 * np.exp(0.0365 * T)
    m_dry = ed + (mo - ed) * 10.0 ** (-kd)

    # wetting
    k1 = 0.424 * (1.0 - ((100.0 - H) / 100.0) ** 1.7) + 0.0694 * np.sqrt(W) * (
        1.0 - ((100.0 - H) / 100.0) ** 8
    )
    kw = k1 * 0.581 * np.exp(0.0365 * T)
    m_wet = ew - (ew - mo) * 10.0 ** (-kw)

    m = np.where(mo > ed, m_dry, np.where(mo < ew, m_wet, mo))

    ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    return np.clip(ffmc, 0.0, 101.0)


def dmc_step(dmc0, T, H, rain, month):
    """Duff Moisture Code."""
    H = np.clip(H, 0.0, 100.0)
    T = np.maximum(T, -1.1)
    le = DMC_DAYLENGTH[np.asarray(month) - 1]

    rk = 1.894 * (T + 1.1) * (100.0 - H) * le * 1.0e-4

    wet = rain > 1.5
    rw = np.where(wet, 0.92 * rain - 1.27, 1.0)
    log_d = np.log(np.maximum(dmc0, 1.0))
    b = np.where(
        dmc0 <= 33.0,
        100.0 / (0.5 + 0.3 * dmc0),
        np.where(dmc0 <= 65.0, 14.0 - 1.3 * log_d, 6.2 * log_d - 17.2),
    )
    wmi = 20.0 + 280.0 / np.exp(0.023 * dmc0)
    wmr = wmi + 1000.0 * rw / (48.77 + b * rw)
    pr = 43.43 * (5.6348 - np.log(np.maximum(wmr - 20.0, 1.0e-6)))
    pr = np.where(wet, pr, dmc0)
    pr = np.maximum(pr, 0.0)

    return np.maximum(pr + rk, 0.0)


def dc_step(dc0, T, rain, month):
    """Drought Code."""
    T = np.maximum(T, -2.8)
    lf = DC_DAYLENGTH[np.asarray(month) - 1]

    pe = np.maximum((0.36 * (T + 2.8) + lf) / 2.0, 0.0)

    wet = rain > 2.8
    rw = np.where(wet, 0.83 * rain - 1.27, 1.0)
    smi = 800.0 * np.exp(-dc0 / 400.0)
    dr = dc0 - 400.0 * np.log(1.0 + 3.937 * rw / smi)
    dr = np.maximum(dr, 0.0)
    dr = np.where(wet, dr, dc0)

    return dr + pe


def isi_calc(ffmc, W):
    """Initial Spread Index."""
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    f_w = np.exp(0.05039 * W)
    f_f = 91.9 * np.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
    return 0.208 * f_w * f_f


def bui_calc(dmc, dc):
    """Build-Up Index."""
    denom = np.maximum(dmc + 0.4 * dc, 1.0e-9)
    bui_low = 0.8 * dmc * dc / denom
    bui_high = dmc - (1.0 - 0.8 * dc / denom) * (
        0.92 + (0.0114 * dmc) ** 1.7
    )
    bui = np.where(dmc <= 0.4 * dc, bui_low, bui_high)
    return np.maximum(bui, 0.0)


def fwi_calc(isi, bui):
    """Fire Weather Index."""
    f_d = np.where(
        bui <= 80.0,
        0.626 * bui ** 0.809 + 2.0,
        1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui)),
    )
    b = 0.1 * isi * f_d
    return np.where(
        b > 1.0,
        np.exp(2.72 * (0.434 * np.log(np.maximum(b, 1.0))) ** 0.647),
        b,
    )


def run_fwi(T, H, W, rain, months):
    """
    Run the FWI System over a time series.
    Inputs have shape (time, ...) and months has shape (time,).
    Returns a dict of arrays with the same shape as T.
    """
    nt = T.shape[0]
    shape = T.shape[1:]

    ffmc = np.full(shape, FFMC_START)
    dmc = np.full(shape, DMC_START)
    dc = np.full(shape, DC_START)

    out = {k: np.empty(T.shape, dtype=np.float32)
           for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi")}

    for t in range(nt):
        ffmc = ffmc_step(ffmc, T[t], H[t], W[t], rain[t])
        dmc = dmc_step(dmc, T[t], H[t], rain[t], months[t])
        dc = dc_step(dc, T[t], rain[t], months[t])
        isi = isi_calc(ffmc, W[t])
        bui = bui_calc(dmc, dc)
        fwi = fwi_calc(isi, bui)

        out["ffmc"][t] = ffmc
        out["dmc"][t] = dmc
        out["dc"][t] = dc
        out["isi"][t] = isi
        out["bui"][t] = bui
        out["fwi"][t] = fwi

    return out


# =====================================================
# SELF-TEST
# =====================================================

def selftest():
    """
    Checks the maths against the published worked example
    (Van Wagner & Pickett 1985 test data, first line):
      April, T=17 C, RH=42 %, wind 25 km/h, rain 0 mm,
      yesterday FFMC 85, DMC 6, DC 15
      -> FFMC 87.692, DMC 8.545, DC 19.013, ISI 10.853, BUI 8.490, FWI 10.096
    plus basic behaviour checks and array/scalar consistency.
    """
    print("[INFO] Running FWI self-test")
    fails = 0

    def check(name, got, want, tol):
        nonlocal fails
        got = float(got)
        ok = abs(got - want) <= tol
        tag = "SUCCESS" if ok else "FAILURE"
        print(f"[{tag}] {name}: got {got:.3f}, expected {want:.3f} (tol {tol})")
        if not ok:
            fails += 1

    T = np.array(17.0)
    H = np.array(42.0)
    W = np.array(25.0)
    R = np.array(0.0)

    ffmc = ffmc_step(np.array(85.0), T, H, W, R)
    dmc = dmc_step(np.array(6.0), T, H, R, 4)
    dc = dc_step(np.array(15.0), T, R, 4)
    isi = isi_calc(ffmc, W)
    bui = bui_calc(dmc, dc)
    fwi = fwi_calc(isi, bui)

    check("FFMC", ffmc, 87.692, 0.01)
    check("DMC", dmc, 8.545, 0.01)
    check("DC", dc, 19.013, 0.01)
    check("ISI", isi, 10.853, 0.05)
    check("BUI", bui, 8.490, 0.05)
    check("FWI", fwi, 10.096, 0.05)

    # behaviour: heavy rain lowers every moisture code
    r = np.array(20.0)
    check_less = [
        ("FFMC rain", ffmc_step(np.array(90.0), T, H, W, r),
         ffmc_step(np.array(90.0), T, H, W, R)),
        ("DMC rain", dmc_step(np.array(40.0), T, H, r, 7),
         dmc_step(np.array(40.0), T, H, R, 7)),
        ("DC rain", dc_step(np.array(300.0), T, r, 7),
         dc_step(np.array(300.0), T, R, 7)),
    ]
    for name, wet_v, dry_v in check_less:
        ok = float(wet_v) < float(dry_v)
        print(f"[{'SUCCESS' if ok else 'FAILURE'}] {name}: "
              f"{float(wet_v):.2f} (rain) < {float(dry_v):.2f} (dry)")
        if not ok:
            fails += 1

    # behaviour: hotter, drier, windier means higher FWI
    def one_day(t, h, w):
        f = ffmc_step(np.array(85.0), np.array(t), np.array(h), np.array(w), R)
        d = dmc_step(np.array(30.0), np.array(t), np.array(h), R, 7)
        c = dc_step(np.array(200.0), np.array(t), R, 7)
        return float(fwi_calc(isi_calc(f, np.array(w)), bui_calc(d, c)))

    mild, extreme = one_day(15, 60, 8), one_day(36, 12, 35)
    ok = extreme > mild
    print(f"[{'SUCCESS' if ok else 'FAILURE'}] FWI mild {mild:.1f} < "
          f"extreme {extreme:.1f}")
    if not ok:
        fails += 1

    # array run gives the same answer as the scalar single-day run
    nt = 3
    Ta = np.full((nt, 2, 2), 17.0)
    Ha = np.full((nt, 2, 2), 42.0)
    Wa = np.full((nt, 2, 2), 25.0)
    Ra = np.zeros((nt, 2, 2))
    res = run_fwi(Ta, Ha, Wa, Ra, np.array([4, 4, 4]))
    day1_ok = abs(float(res["ffmc"][0, 0, 0]) - 87.692) < 0.01
    finite = all(np.isfinite(v).all() for v in res.values())
    print(f"[{'SUCCESS' if day1_ok and finite else 'FAILURE'}] "
          f"array run: day-1 FFMC matches reference, all values finite")
    if not (day1_ok and finite):
        fails += 1

    # extreme inputs must not produce NaN or infinity
    Te = np.array([[-30.0, 45.0]])
    He = np.array([[100.0, 0.0]])
    We = np.array([[0.0, 60.0]])
    Re = np.array([[50.0, 0.0]])
    res = run_fwi(np.repeat(Te[None], 5, 0), np.repeat(He[None], 5, 0),
                  np.repeat(We[None], 5, 0), np.repeat(Re[None], 5, 0),
                  np.array([4, 5, 6, 7, 8]))
    finite = all(np.isfinite(v).all() for v in res.values())
    print(f"[{'SUCCESS' if finite else 'FAILURE'}] extreme inputs stay finite")
    if not finite:
        fails += 1

    if fails:
        print(f"[FAILURE] Self-test failed ({fails} checks)")
        return 1
    print("[SUCCESS] Self-test passed")
    return 0


# =====================================================
# PREPROCESSING + DRIVER
# =====================================================

def magnus_es(t_c):
    """Saturation vapour pressure (hPa), Alduchov & Eskridge (1996)."""
    return 6.1094 * np.exp(17.625 * t_c / (t_c + 243.04))


def process_year(year, force):
    import pandas as pd
    import xarray as xr

    inst_path = RAW_DIR / f"era5_instant_{year}_AprSep.nc"
    prec_path = RAW_DIR / f"era5_precip_{year}_AprSep.nc"
    out_path = PROCESSED_DIR / f"fwi_{year}.nc"

    if out_path.exists() and not force:
        print(f"[INFO] Exists, skipping: {out_path.name} (use --force)")
        return True

    for p in (inst_path, prec_path):
        if not p.exists():
            print(f"[FAILURE] Missing input: {p.name}")
            return False

    print(f"[INFO] Year {year}: reading ERA5")
    inst = xr.open_dataset(inst_path, engine="netcdf4")
    prec = xr.open_dataset(prec_path, engine="netcdf4")

    try:
        tname = "valid_time" if "valid_time" in inst.coords else "time"

        times = pd.DatetimeIndex(inst[tname].values)
        dates = times.normalize()

        # float64 so the day-by-day moisture codes do not accumulate rounding
        t_c = inst["t2m"].values.astype("float64") - 273.15
        td_c = inst["d2m"].values.astype("float64") - 273.15
        rh = 100.0 * magnus_es(td_c) / magnus_es(t_c)
        rh = np.clip(rh, 0.0, 100.0)
        wind_kmh = np.hypot(
            inst["u10"].values.astype("float64"),
            inst["v10"].values.astype("float64"),
        ) * 3.6
        swvl1 = np.clip(inst["swvl1"].values.astype("float64"), 0.0, None)

        # 24 h rain: window 21 UTC (D-1) .. 20 UTC (D) -> shift +3 h, group by day
        ptime = pd.DatetimeIndex(prec[("valid_time" if "valid_time" in prec.coords
                                       else "time")].values)
        shifted_day = (ptime + pd.Timedelta(hours=3)).normalize()
        tp_mm = np.clip(prec["tp"].values.astype("float64") * 1000.0, 0.0, None)

        rain = np.zeros_like(t_c)
        n_missing = 0
        for i, d in enumerate(dates):
            sel = np.where(shifted_day == d)[0]
            if len(sel) == 0:
                n_missing += 1
                continue
            rain[i] = tp_mm[sel].sum(axis=0)
        if n_missing:
            print(f"[WARNING] {n_missing} days had no rain data (set to 0)")

        months = dates.month.values

        print(f"[INFO] Year {year}: running FWI over {len(dates)} days")
        codes = run_fwi(t_c, rh, wind_kmh, rain, months)

        lat = inst["latitude"].values
        lon = inst["longitude"].values
        dims = ("date", "latitude", "longitude")

        ds = xr.Dataset(
            {
                "t2m_c": (dims, t_c.astype("float32"),
                          {"units": "degC", "long_name": "2 m temperature at 20 UTC"}),
                "rh": (dims, rh.astype("float32"),
                       {"units": "%", "long_name": "relative humidity at 20 UTC (Magnus)"}),
                "wind_kmh": (dims, wind_kmh.astype("float32"),
                             {"units": "km/h", "long_name": "10 m wind speed at 20 UTC"}),
                "rain24_mm": (dims, rain.astype("float32"),
                              {"units": "mm", "long_name": "24 h rain ending 20 UTC"}),
                "swvl1": (dims, swvl1.astype("float32"),
                          {"units": "m3 m-3",
                           "long_name": "volumetric soil water layer 1 at 20 UTC (clipped >= 0)"}),
                "ffmc": (dims, codes["ffmc"], {"long_name": "Fine Fuel Moisture Code"}),
                "dmc": (dims, codes["dmc"], {"long_name": "Duff Moisture Code"}),
                "dc": (dims, codes["dc"], {"long_name": "Drought Code"}),
                "isi": (dims, codes["isi"], {"long_name": "Initial Spread Index"}),
                "bui": (dims, codes["bui"], {"long_name": "Build-Up Index"}),
                "fwi": (dims, codes["fwi"], {"long_name": "Fire Weather Index"}),
            },
            coords={"date": dates.values, "latitude": lat, "longitude": lon},
            attrs={
                "title": "Canadian FWI System from ERA5, Western United States",
                "source": "ERA5 reanalysis, Copernicus Climate Change Service",
                "method": ("Van Wagner & Pickett (1985); start-up FFMC 85, "
                           "DMC 6, DC 15 on 1 April; inputs at 20 UTC"),
                "note": ("Weather-based hazard indicator, not a validated "
                         "US fire-danger rating"),
            },
        )

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        tmp = out_path.with_suffix(".nc.part")
        ds.to_netcdf(tmp, encoding=enc)
        tmp.replace(out_path)

        jjas = np.isin(months, [6, 7, 8, 9])
        print(
            f"[SUCCESS] {out_path.name}: JJAS mean FWI "
            f"{np.nanmean(codes['fwi'][jjas]):.1f}, "
            f"max FWI {np.nanmax(codes['fwi']):.1f}"
        )
        return True
    finally:
        inst.close()
        prec.close()


def find_years():
    years = set()
    for p in RAW_DIR.glob("era5_instant_*_AprSep.nc"):
        m = re.match(r"era5_instant_(\d{4})_AprSep\.nc", p.name)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("FWI COMPUTATION")
    print("=" * 60)

    if args.selftest:
        return selftest()

    years = [args.year] if args.year else find_years()
    if not years:
        print("[FAILURE] No ERA5 files found in data/raw. Run 01 first.")
        return 1

    print(f"[INFO] Years: {years[0]}-{years[-1]} ({len(years)} years)")
    failed = []
    for year in years:
        try:
            if not process_year(year, args.force):
                failed.append(year)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAILURE] Year {year}: {exc}")
            failed.append(year)

    print("=" * 60)
    if failed:
        print(f"[FAILURE] Failed years: {failed}")
        return 1
    print("[SUCCESS] FWI files written to data/processed/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
