#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_hazard_climatology.py

Turns the daily FWI fields from 03_compute_fwi.py into the hazard indicators
for the Western US study states (CA, OR, WA, NV, ID, AZ).

Season: June-September (JJAS). April-May are spin-up for the moisture codes
and are not analysed.

Indicators (per 0.25 degree cell)
  fwi_mean             mean daily FWI, JJAS, all years
  fwi_p50/90/95/99     the cell's OWN percentiles of daily FWI (all JJAS days
                       of all years): how severe its extreme days are
  days_gt_p90 / p95    per year: number of JJAS days above the cell's own
                       90th / 95th percentile. FWI ignores fuel, so deserts
                       score high on the raw scale; counting days relative to
                       each cell's own climate removes that level difference.
                       Pooled over all years the average is 12.2 days (10% of
                       122) in every cell BY CONSTRUCTION, so the information
                       is in the year-to-year variation, the trend and the
                       timing within the season, not in the average map.
  days_gt_regional_p90 per year: days above ONE fixed FWI value (the 90th
                       percentile of all study-area cells and days). Shown
                       next to the local version to make the fuel-blindness
                       visible: it highlights the desert Southwest.
  trend_*              Theil-Sen slope (days per decade) of the annual counts
                       with a Mann-Kendall p-value (ties corrected).
                       22 years is short: this describes the period, it is
                       not climate-change attribution. Roughly 5% of cells
                       show p < 0.05 by chance alone and neighbouring cells
                       are correlated, so read the spatial pattern, not
                       single dotted cells.

State summaries use cell centres inside Natural Earth state polygons
(1:50m, needs cartopy and shapely >= 2.0) and cos(latitude) area weights.
If the polygons cannot be loaded, the whole land area is used as one group.

This is a hazard indicator from weather alone: no fuels, ignition,
suppression, exposure or vulnerability. It is not a validated fire-danger
rating and hazard is not loss.

Outputs
    data/processed/hazard_climatology.nc
    data/processed/hazard_state_annual.csv      state x year
    data/processed/hazard_state_monthly.csv     state x month (JJAS)
    data/processed/hazard_state_trends.csv      state summary and trends
    figures/hazard_maps.png                     4 maps
    figures/hazard_event_years.png              maps of the most extreme years
    figures/hazard_state_years.png              annual counts per state
    figures/hazard_seasonality.png              monthly FWI and exceedance share
    with --smooth, the two map figures are also written as *_smooth.png
    (smooth contours clipped to the state outlines; the cell versions at the
    true 0.25 deg resolution are always written)

Usage:
    python 05_hazard_climatology.py --selftest
    python 05_hazard_climatology.py
    python 05_hazard_climatology.py --event-years 2018 2020 2021
    python 05_hazard_climatology.py --smooth     # also smooth versions of maps
"""

import argparse
import csv
import math
import re
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np

# cells with no data (all NaN) are expected and give NaN, not an error
warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "figures"

OUT_NC = PROCESSED_DIR / "hazard_climatology.nc"
OUT_ANNUAL = PROCESSED_DIR / "hazard_state_annual.csv"
OUT_MONTHLY = PROCESSED_DIR / "hazard_state_monthly.csv"
OUT_TRENDS = PROCESSED_DIR / "hazard_state_trends.csv"


# =====================================================
# SETTINGS
# =====================================================

STATES = ["CA", "OR", "WA", "NV", "ID", "AZ"]

# fixed colour per state (validated categorical order, light surface)
STATE_COLORS = dict(zip(STATES, ["#2a78d6", "#eb6834", "#1baf7a",
                                 "#eda100", "#e87ba4", "#008300"]))
DEFAULT_COLOR = "#2a78d6"

SEASON_MONTHS = [6, 7, 8, 9]
MONTH_DAYS = [30, 31, 31, 30]
MONTH_LABELS = ["Jun", "Jul", "Aug", "Sep"]
N_DAYS = sum(MONTH_DAYS)  # 122

QUANTILES = (50, 90, 95, 99)
ALPHA = 0.05
MIN_YEARS_TREND = 8
MIN_YEARS_WARN = 15

MEAN_LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]
P95_LEVELS = [0, 20, 40, 60, 80, 100, 120]
REGIONAL_LEVELS = [0, 2, 5, 10, 20, 40, 60, 80, 100]
YEAR_LEVELS = [0, 5, 10, 15, 20, 30, 45, 60, 90, 122]

CAVEAT = ("FWI = weather-driven fire danger only (no fuels, ignition or "
          "exposure); hazard is not loss.")


# =====================================================
# STATISTICS (numpy only, tested by --selftest)
# =====================================================

def season_months():
    """Month number of each of the 122 JJAS days."""
    return np.repeat(SEASON_MONTHS, MONTH_DAYS)


def trend_test(y, x):
    """
    Theil-Sen slope and Mann-Kendall test for series y at times x.
    Returns (slope per x-unit, z, two-sided p). NaN if too short or gappy.
    Normal approximation with tie correction.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = y.size
    if n < MIN_YEARS_TREND or not np.all(np.isfinite(y)):
        return np.nan, np.nan, np.nan
    i, j = np.triu_indices(n, 1)
    dy = y[j] - y[i]
    dx = x[j] - x[i]
    slope = float(np.median(dy / dx))
    s = float(np.sign(dy).sum())
    _, counts = np.unique(y, return_counts=True)
    t = counts[counts > 1].astype(float)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(t * (t - 1) * (2 * t + 5))) / 18.0
    if var <= 0 or s == 0:
        return slope, 0.0, 1.0
    z = (s - np.sign(s)) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return slope, z, p


def trend_field(counts, years, mask):
    """Trend per cell. counts: (year, lat, lon). Returns slope, p (lat, lon)."""
    slope = np.full(mask.shape, np.nan)
    pval = np.full(mask.shape, np.nan)
    for iy, ix in zip(*np.nonzero(mask)):
        sl, _, p = trend_test(counts[:, iy, ix], years)
        slope[iy, ix], pval[iy, ix] = sl, p
    return slope, pval


def area_mean(field, mask, lat):
    """cos(latitude)-weighted mean over mask cells. field: (..., lat, lon)."""
    w = np.cos(np.deg2rad(lat))[:, None] * mask
    f = np.where(mask, field, np.nan)
    ok = np.isfinite(f)
    wsum = (w * ok).sum(axis=(-2, -1))
    total = np.nansum(f * w, axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / wsum


def local_percentiles(stack, qs=QUANTILES):
    """Per-cell percentiles over all days of all years. stack: (Y, D, lat, lon)."""
    flat = stack.reshape(-1, stack.shape[-2], stack.shape[-1])
    return np.nanpercentile(flat, list(qs), axis=0)


def count_above(stack, threshold):
    """Days per year above a threshold. threshold: scalar or (lat, lon)."""
    return (stack > threshold).sum(axis=1).astype(np.int16)


# =====================================================
# STATE MASKS
# =====================================================

def state_masks(lat, lon, states):
    """
    Returns (masks, outline): {postal code: boolean (lat, lon)} from Natural
    Earth state polygons, and the union of those polygons (for clipping smooth
    maps). Returns (None, None) if cartopy / shapely / the shapefile cannot be
    used. A cell belongs to a state if its centre lies inside the polygon.
    """
    try:
        import cartopy.io.shapereader as shpreader
        try:
            from shapely import contains_xy
        except ImportError:  # shapely 1.8
            from shapely.vectorized import contains as contains_xy
    except ImportError as exc:
        print(f"[WARNING] State masks need cartopy and shapely: {exc}")
        return None, None

    try:
        path = shpreader.natural_earth(
            resolution="50m", category="cultural",
            name="admin_1_states_provinces_lakes")
        records = list(shpreader.Reader(path).records())
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Could not load state polygons: {exc}")
        return None, None

    LON, LAT = np.meshgrid(lon, lat)
    masks, geoms = {}, {}
    for rec in records:
        a = {k.lower(): v for k, v in rec.attributes.items()}
        is_us = (a.get("adm0_a3") == "USA"
                 or a.get("admin") == "United States of America")
        code = a.get("postal")
        if is_us and code in states:
            masks[code] = np.asarray(contains_xy(rec.geometry, LON, LAT))
            geoms[code] = rec.geometry

    if not masks:
        keys = sorted({k.lower() for k in records[0].attributes}) if records else []
        print(f"[WARNING] No study states found in the shapefile; fields: {keys}")
        return None, None

    out = {}
    for code in states:
        if code in masks and masks[code].any():
            out[code] = masks[code]
            print(f"[INFO] {code}: {int(masks[code].sum())} cells")
        else:
            print(f"[WARNING] {code}: no grid cells found, skipped")
    if not out:
        return None, None
    from shapely.ops import unary_union
    return out, unary_union([geoms[c] for c in out])


def land_fallback(path):
    """Land mask from soil moisture (NaN over sea) in the first FWI file."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        sw = ds["swvl1"].isel(date=0).values
    land = np.isfinite(sw)
    if land.all():
        print("[WARNING] swvl1 has no sea mask: all cells counted as land")
    return land


# =====================================================
# LOADING
# =====================================================

def find_years():
    years = []
    for p in PROCESSED_DIR.glob("fwi_*.nc"):
        m = re.fullmatch(r"fwi_(\d{4})\.nc", p.name)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def load_year(year):
    """JJAS daily FWI of one year as float32 (122, lat, lon)."""
    import pandas as pd
    import xarray as xr

    with xr.open_dataset(PROCESSED_DIR / f"fwi_{year}.nc") as ds:
        dates = pd.DatetimeIndex(ds["date"].values)
        keep = np.nonzero(np.isin(dates.month, SEASON_MONTHS))[0]
        months = dates.month.values[keep]
        fwi = ds["fwi"].isel(date=keep).values.astype("float32")
        lat = ds["latitude"].values
        lon = ds["longitude"].values
    return fwi, months, lat, lon


# =====================================================
# TABLES AND OUTPUT FILES
# =====================================================

def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def write_netcdf(path, years, lat, lon, fields_2d, fields_3d, study, attrs):
    import xarray as xr

    data_vars = {}
    for name, (arr, meta) in fields_2d.items():
        data_vars[name] = (("latitude", "longitude"), arr, meta)
    for name, (arr, meta) in fields_3d.items():
        data_vars[name] = (("year", "latitude", "longitude"), arr, meta)
    data_vars["study_area"] = (("latitude", "longitude"), study.astype("int8"),
                               {"long_name": "1 = inside study states"})
    ds = xr.Dataset(data_vars, coords={"year": np.asarray(years, dtype="int32"),
                                       "latitude": lat, "longitude": lon},
                    attrs=attrs)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    tmp = path.with_suffix(".nc.part")
    ds.to_netcdf(tmp, encoding=enc)
    tmp.replace(path)


# =====================================================
# FIGURES
# =====================================================

def _panel(fig, nrows, ncols, idx, extent, lon, lat, field, cmap_name, levels,
           extend, title, label, fmt="%g", smooth=False, clip=None):
    import map_utils as mu

    ax = mu.map_axes(fig, nrows, ncols, idx, extent)
    cmap, norm = mu.discrete_cmap(cmap_name, levels, extend)
    mesh = mu.draw_field(ax, lon, lat, field, cmap, norm, smooth=smooth,
                         clip=clip)
    ax.set_title(textwrap.fill(title, 30), fontsize=9)
    mu.add_colorbar(fig, mesh, ax, label, levels, fmt=fmt)
    return ax


def make_hazard_maps(fig_path, lon, lat, show, fields, y0, y1,
                     smooth=False, clip=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import map_utils as mu

    def masked(a):
        return np.where(show, a, np.nan)

    extent = mu.grid_extent(lat, lon)
    fig = plt.figure(figsize=(17, 6.4))

    _panel(fig, 1, 4, 1, extent, lon, lat, masked(fields["fwi_mean"]),
           "YlOrRd", MEAN_LEVELS, "max",
           "(a) Mean daily FWI, JJAS", "FWI", smooth=smooth, clip=clip)

    _panel(fig, 1, 4, 2, extent, lon, lat, masked(fields["fwi_p95"]),
           "YlOrRd", P95_LEVELS, "max",
           "(b) Severity of extreme days: each cell's own 95th percentile of FWI",
           "FWI", smooth=smooth, clip=clip)

    _panel(fig, 1, 4, 3, extent, lon, lat, masked(fields["regional_days"]),
           "YlOrRd", REGIONAL_LEVELS, "max",
           f"(c) Days per JJAS above one fixed FWI of {fields['regional_thr']:.0f}"
           " (regional P90): fuel-blind",
           "days per season", smooth=smooth, clip=clip)

    tr = masked(fields["trend_p90"])
    t_levels = mu.nice_symmetric_levels(tr[np.isfinite(tr)])
    ax = _panel(fig, 1, 4, 4, extent, lon, lat, tr, "RdBu_r", t_levels, "both",
                "(d) Trend in days above each cell's own P90 (dots: p < 0.05)",
                "days per decade", smooth=smooth, clip=clip)
    mu.stipple(ax, lon, lat, show & (fields["trend_p90_p"] < ALPHA))

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mu.save_figure(fig, fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_event_years(fig_path, lon, lat, show, counts_p90, years, picks,
                     smooth=False, clip=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import map_utils as mu

    n = len(picks)
    extent = mu.grid_extent(lat, lon)
    fig = plt.figure(figsize=(4.2 * n + 1.4, 5.8))
    cmap, norm = mu.discrete_cmap("YlOrRd", YEAR_LEVELS, "neither")
    axes, mesh = [], None
    for k, year in enumerate(picks, start=1):
        field = np.where(show, counts_p90[years.index(year)].astype(float), np.nan)
        ax = mu.map_axes(fig, 1, n, k, extent)
        mesh = mu.draw_field(ax, lon, lat, field, cmap, norm, smooth=smooth,
                             clip=clip)
        mean_days = float(np.nanmean(field))
        ax.set_title(f"{year}\nstudy-area mean {mean_days:.1f} days", fontsize=9)
        axes.append(ax)
    mu.add_colorbar(fig, mesh, axes, "days above the cell's own P90 (JJAS)",
                    YEAR_LEVELS, fmt="%d")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mu.save_figure(fig, fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _pfmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def make_state_years(fig_path, years, annual, trends):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import map_utils as mu

    names = list(annual)
    n = len(names)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows),
                             sharey=True, squeeze=False)
    top = max(max(v) for v in annual.values()) * 1.15
    for ax, name in zip(axes.ravel(), names):
        y = np.asarray(annual[name], dtype=float)
        ax.bar(years, y, width=0.75, color=STATE_COLORS.get(name, DEFAULT_COLOR))
        ax.axhline(0.1 * N_DAYS, color="0.3", lw=1, ls=":")
        slope, p = trends[name]
        if np.isfinite(slope):
            icpt = float(np.median(y - slope * np.asarray(years)))
            ax.plot(years, icpt + slope * np.asarray(years), color="0.1", lw=2)
            ax.set_title(f"{name}: {slope * 10:+.1f} days/decade ({_pfmt(p)})",
                         fontsize=10)
        else:
            ax.set_title(name, fontsize=10)
        ax.set_ylim(0, top)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("days above own P90")
    fig.suptitle("Days above each cell's own 90th percentile of FWI, JJAS, "
                 "state average (dotted: 12.2 expected; line: Theil-Sen trend)",
                 fontsize=11)
    fig.text(0.01, 0.005, "Data: ERA5 (C3S), Canadian FWI System. " + CAVEAT,
             fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mu.save_figure(fig, fig_path, dpi=170)
    plt.close(fig)


def make_seasonality(fig_path, monthly):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import map_utils as mu

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(MONTH_LABELS))
    for name, (mean_fwi, frac) in monthly.items():
        color = STATE_COLORS.get(name, DEFAULT_COLOR)
        axes[0].plot(x, mean_fwi, color=color, lw=2, marker="o", ms=6, label=name)
        axes[1].plot(x, frac, color=color, lw=2, marker="o", ms=6, label=name)
    axes[1].axhline(0.10, color="0.3", lw=1, ls=":")
    axes[0].set_title("Mean daily FWI by month")
    axes[0].set_ylabel("FWI")
    axes[1].set_title("Share of days above the cell's own P90 "
                      "(dotted: 10% expected)")
    axes[1].set_ylabel("share of days")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_LABELS)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", lw=0.6)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels),
               loc="lower center", bbox_to_anchor=(0.5, 0.035))
    fig.text(0.01, 0.005, "Data: ERA5 (C3S), Canadian FWI System. Moisture "
             "codes restart on 1 April, so June is the least settled month. "
             + CAVEAT, fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mu.save_figure(fig, fig_path, dpi=170)
    plt.close(fig)


# =====================================================
# SELF-TEST
# =====================================================

def selftest():
    failures = 0

    def check(cond, msg):
        nonlocal failures
        if cond:
            print(f"[SUCCESS] {msg}")
        else:
            failures += 1
            print(f"[FAILURE] {msg}")

    yrs = np.arange(2001, 2023)

    slope, z, p = trend_test(2.0 * (yrs - 2001) + 5, yrs)
    check(abs(slope - 2.0) < 1e-12 and p < 1e-6, "linear increase: slope 2, tiny p")

    slope, z, p = trend_test(-0.5 * (yrs - 2001), yrs)
    check(abs(slope + 0.5) < 1e-12 and z < 0, "linear decrease: slope -0.5, z < 0")

    slope, z, p = trend_test(np.full(yrs.size, 12.0), yrs)
    check(slope == 0.0 and p == 1.0, "constant series: slope 0, p = 1")

    rng = np.random.default_rng(3)
    ps = [trend_test(rng.normal(size=22), yrs)[2] for _ in range(2000)]
    check(0.03 < np.mean(np.array(ps) < 0.05) < 0.08,
          f"white noise: {np.mean(np.array(ps) < 0.05):.3f} of p < 0.05 (about 0.05)")

    tied = np.array([3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12])
    s1, z1, p1 = trend_test(tied, yrs)
    check(np.isfinite(p1) and p1 < 0.001, "tied integer counts handled")

    check(np.isnan(trend_test([1, 2, 3], [1, 2, 3])[0]), "short series gives NaN")

    check(season_months().size == N_DAYS and set(season_months()) == set(SEASON_MONTHS),
          f"season has {N_DAYS} days in months {SEASON_MONTHS}")

    # percentile / count logic on a synthetic stack
    ny, nd, nla, nlo = 6, N_DAYS, 4, 5
    stack = rng.gamma(2.0, 8.0, size=(ny, nd, nla, nlo)).astype("float32")
    stack[..., 0, 0] *= 3.0  # one "desert" cell with much higher FWI
    q = local_percentiles(stack)
    p90 = q[QUANTILES.index(90)]
    cnt = count_above(stack, p90)
    check(cnt.shape == (ny, nla, nlo), "counts have shape (year, lat, lon)")
    frac = cnt.sum(axis=0) / (ny * nd)
    check(np.allclose(frac, 0.10, atol=0.002),
          "local P90 is exceeded on 10% of days in every cell")
    check(abs(float(np.mean(cnt)) - 0.10 * nd) < 0.3, "mean count is about 12.2 days")

    thr = float(np.nanpercentile(stack, 90))
    reg = count_above(stack, thr)
    check(reg[:, 0, 0].mean() > 3 * reg[:, 1, 1].mean(),
          "fixed regional threshold favours the high-FWI cell (fuel-blind)")

    # NaN cells give NaN percentiles and zero counts
    nanstack = stack.copy()
    nanstack[..., 3, 4] = np.nan
    qn = local_percentiles(nanstack)
    check(np.isnan(qn[0][3, 4]) and np.isfinite(qn[0][0, 0]), "all-NaN cell gives NaN")

    # area mean
    lat = np.array([60.0, 0.0])
    field = np.array([[10.0, 10.0], [20.0, 20.0]])
    am = area_mean(field, np.ones((2, 2), bool), lat)
    check(abs(am - (0.5 * 10 + 1.0 * 20) / 1.5) < 1e-9, "area mean uses cos(lat) weights")
    check(abs(area_mean(field, np.array([[True, False], [False, False]]), lat) - 10.0) < 1e-9,
          "area mean respects the mask")

    # trend field
    cnts = np.zeros((22, 2, 2))
    cnts[:, 0, 0] = np.arange(22)
    sl, pv = trend_field(cnts, yrs, np.array([[True, False], [False, True]]))
    check(abs(sl[0, 0] - 1.0) < 1e-12 and np.isnan(sl[0, 1]) and pv[1, 1] == 1.0,
          "trend field only fills masked cells")

    print("=" * 60)
    if failures:
        print(f"[FAILURE] Self-test: {failures} check(s) failed")
        return 1
    print("[SUCCESS] Self-test passed")
    return 0


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--event-years", type=int, nargs="+", default=None,
                        help="years for the event-year maps (max 4)")
    parser.add_argument("--smooth", action="store_true",
                        help="also write smooth (contour) versions of the maps")
    args = parser.parse_args()

    print("=" * 60)
    print("HAZARD CLIMATOLOGY (JJAS)")
    print("=" * 60)

    if args.selftest:
        return selftest()

    # ---- load all years -------------------------------------------------
    years_found = find_years()
    if not years_found:
        print("[FAILURE] No fwi_<year>.nc files in data/processed. Run 03 first.")
        return 1

    expected_months = season_months()
    years, stacks, lat, lon = [], [], None, None
    for year in years_found:
        try:
            fwi, months, la, lo = load_year(year)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] Year {year} skipped: {exc}")
            continue
        if months.size != N_DAYS or not np.array_equal(months, expected_months):
            print(f"[WARNING] Year {year} skipped: {months.size} JJAS days, "
                  f"expected {N_DAYS}")
            continue
        if lat is None:
            lat, lon = la, lo
        elif not (np.allclose(lat, la) and np.allclose(lon, lo)):
            print(f"[WARNING] Year {year} skipped: grid differs")
            continue
        years.append(year)
        stacks.append(fwi)

    if len(years) < MIN_YEARS_TREND:
        print(f"[FAILURE] Only {len(years)} usable years; need >= {MIN_YEARS_TREND}")
        return 1
    stack = np.stack(stacks)  # (year, day, lat, lon)
    del stacks
    y0, y1 = years[0], years[-1]
    print(f"[INFO] Years: {y0}-{y1} ({len(years)} years), grid {len(lat)} x {len(lon)}")
    if len(years) < MIN_YEARS_WARN:
        print(f"[WARNING] Percentiles and trends from only {len(years)} years "
              "are preliminary")

    # ---- study area -----------------------------------------------------
    masks, outline = state_masks(lat, lon, STATES)
    if masks:
        groups = masks
        study = np.any(list(masks.values()), axis=0)
    else:
        print("[WARNING] Using the whole land area as one group")
        study = land_fallback(PROCESSED_DIR / f"fwi_{years[0]}.nc")
        groups = {"Land": study}
    print(f"[INFO] Study area: {int(study.sum())} cells in {list(groups)}")

    import map_utils as mu

    clip = mu.geom_to_path(outline) if outline is not None else None

    # ---- indicators -----------------------------------------------------
    q = local_percentiles(stack)
    pct = {f"fwi_p{k}": q[i] for i, k in enumerate(QUANTILES)}
    p90, p95 = pct["fwi_p90"], pct["fwi_p95"]
    fwi_mean = np.nanmean(stack, axis=(0, 1))

    counts_p90 = count_above(stack, p90)
    counts_p95 = count_above(stack, p95)
    regional_thr = float(np.nanpercentile(stack[..., study], 90))
    counts_reg = count_above(stack, regional_thr)
    print(f"[INFO] Regional P90 threshold (study cells, all days): {regional_thr:.1f}")

    yr = np.asarray(years)
    tr90, tr90_p = trend_field(counts_p90, yr, study)
    tr95, tr95_p = trend_field(counts_p95, yr, study)
    tr90, tr95 = tr90 * 10.0, tr95 * 10.0  # per decade
    sig = float(np.mean(tr90_p[study] < ALPHA))
    print(f"[INFO] Cells with p < {ALPHA} for the P90-count trend: {sig:.1%} "
          f"(about {ALPHA:.0%} would occur by chance)")

    # ---- state tables ---------------------------------------------------
    mean_year = np.nanmean(stack, axis=1)  # (year, lat, lon)
    months = expected_months
    month_mean = {m: np.nanmean(stack[:, months == m], axis=(0, 1))
                  for m in SEASON_MONTHS}
    month_frac = {m: (stack[:, months == m] > p90).mean(axis=(0, 1))
                  for m in SEASON_MONTHS}

    annual_rows, monthly_rows, trend_rows = [], [], []
    annual_plot, monthly_plot, trend_plot = {}, {}, {}
    for name, mask in groups.items():
        s_mean = area_mean(mean_year, mask, lat)
        s90 = area_mean(counts_p90.astype(float), mask, lat)
        s95 = area_mean(counts_p95.astype(float), mask, lat)
        sreg = area_mean(counts_reg.astype(float), mask, lat)
        for k, year in enumerate(years):
            annual_rows.append([name, year, round(float(s_mean[k]), 3),
                                round(float(s90[k]), 3), round(float(s95[k]), 3),
                                round(float(sreg[k]), 3)])
        mm = [float(area_mean(month_mean[m], mask, lat)) for m in SEASON_MONTHS]
        mf = [float(area_mean(month_frac[m], mask, lat)) for m in SEASON_MONTHS]
        for k, m in enumerate(SEASON_MONTHS):
            monthly_rows.append([name, m, round(mm[k], 3), round(mf[k], 4)])
        sl90, _, pp90 = trend_test(s90, yr)
        sl95, _, pp95 = trend_test(s95, yr)
        slm, _, ppm = trend_test(s_mean, yr)
        trend_rows.append([
            name, int(mask.sum()),
            round(float(np.nanmean(s_mean)), 2),
            round(float(area_mean(pct["fwi_p95"], mask, lat)), 2),
            round(sl90 * 10, 3), round(pp90, 4),
            round(sl95 * 10, 3), round(pp95, 4),
            round(slm * 10, 3), round(ppm, 4),
        ])
        annual_plot[name] = s90
        monthly_plot[name] = (mm, mf)
        trend_plot[name] = (sl90, pp90)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_ANNUAL, ["state", "year", "mean_fwi_jjas", "days_gt_local_p90",
                           "days_gt_local_p95", "days_gt_regional_p90"],
              annual_rows)
    write_csv(OUT_MONTHLY, ["state", "month", "mean_fwi", "share_days_gt_local_p90"],
              monthly_rows)
    write_csv(OUT_TRENDS, ["state", "n_cells", "mean_fwi_jjas", "mean_local_p95",
                           "trend_days_gt_p90_per_decade", "p_days_gt_p90",
                           "trend_days_gt_p95_per_decade", "p_days_gt_p95",
                           "trend_mean_fwi_per_decade", "p_mean_fwi"],
              trend_rows)
    print(f"[INFO] Wrote {OUT_ANNUAL.name}, {OUT_MONTHLY.name}, {OUT_TRENDS.name}")

    # ---- printed summary ------------------------------------------------
    print("=" * 60)
    print(f"{'group':6s} {'cells':>6s} {'meanFWI':>8s} {'P95':>7s} "
          f"{'dP90/dec':>9s} {'p':>6s} {'dP95/dec':>9s} {'p':>6s}")
    for r in trend_rows:
        print(f"{r[0]:6s} {r[1]:6d} {r[2]:8.1f} {r[3]:7.1f} "
              f"{r[4]:9.2f} {r[5]:6.3f} {r[6]:9.2f} {r[7]:6.3f}")
    study_90 = area_mean(counts_p90.astype(float), study, lat)
    order = np.argsort(study_90)[::-1]
    print("[INFO] Years ranked by study-area days above local P90:")
    for k in order[:5]:
        print(f"       {years[k]}: {study_90[k]:.1f} days")

    # ---- NetCDF ---------------------------------------------------------
    f2d = {
        "fwi_mean": (fwi_mean.astype("float32"), {"long_name": "Mean daily FWI, JJAS"}),
        "trend_days_gt_p90_per_decade": (tr90.astype("float32"), {
            "long_name": "Theil-Sen trend of days above local P90", "units": "days/decade"}),
        "trend_days_gt_p90_pvalue": (tr90_p.astype("float32"), {
            "long_name": "Mann-Kendall p-value (two-sided)"}),
        "trend_days_gt_p95_per_decade": (tr95.astype("float32"), {
            "long_name": "Theil-Sen trend of days above local P95", "units": "days/decade"}),
        "trend_days_gt_p95_pvalue": (tr95_p.astype("float32"), {
            "long_name": "Mann-Kendall p-value (two-sided)"}),
    }
    for k in QUANTILES:
        f2d[f"fwi_p{k}"] = (pct[f"fwi_p{k}"].astype("float32"), {
            "long_name": f"Local {k}th percentile of daily FWI, JJAS, all years"})
    f3d = {
        "days_gt_p90": (counts_p90, {"long_name": "JJAS days above local P90", "units": "days"}),
        "days_gt_p95": (counts_p95, {"long_name": "JJAS days above local P95", "units": "days"}),
        "days_gt_regional_p90": (counts_reg, {
            "long_name": "JJAS days above the fixed regional P90 value", "units": "days"}),
    }
    try:
        write_netcdf(OUT_NC, years, lat, lon, f2d, f3d, study, {
            "title": "Western US JJAS fire weather hazard indicators",
            "source": "ERA5 (C3S) driven Canadian FWI System, 03_compute_fwi.py",
            "regional_p90_threshold": regional_thr,
            "note": "Weather-driven indicator only; not a validated fire-danger "
                    "rating; hazard is not loss.",
        })
        print(f"[SUCCESS] Wrote {OUT_NC}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAILURE] NetCDF not written: {exc}")
        return 1

    # ---- figures --------------------------------------------------------
    fields = {
        "fwi_mean": fwi_mean, "fwi_p95": p95,
        "regional_days": counts_reg.mean(axis=0), "regional_thr": regional_thr,
        "trend_p90": tr90, "trend_p90_p": tr90_p,
    }
    if args.event_years:
        picks = [y for y in args.event_years if y in years][:4]
        for y in args.event_years:
            if y not in years:
                print(f"[WARNING] Event year {y} not available, skipped")
    else:
        picks = [years[k] for k in order[:4]]
    picks = sorted(picks)

    variants = [False, True] if args.smooth else [False]
    jobs = []
    for sm in variants:
        jobs.append((mu.variant_name("hazard_maps.png", sm),
                     lambda p, sm=sm: make_hazard_maps(p, lon, lat, study, fields,
                                                       y0, y1, smooth=sm, clip=clip)))
        if picks:
            jobs.append((mu.variant_name("hazard_event_years.png", sm),
                         lambda p, sm=sm: make_event_years(p, lon, lat, study,
                                                           counts_p90, years, picks,
                                                           smooth=sm, clip=clip)))
    jobs.append(("hazard_state_years.png",
                 lambda p: make_state_years(p, years, annual_plot, trend_plot)))
    jobs.append(("hazard_seasonality.png",
                 lambda p: make_seasonality(p, monthly_plot)))
    for fname, func in jobs:
        path = FIG_DIR / fname
        try:
            func(path)
            print(f"[SUCCESS] Figure: {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] {fname} not written: {exc}")

    print("=" * 60)
    print("[SUCCESS] Hazard climatology finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
