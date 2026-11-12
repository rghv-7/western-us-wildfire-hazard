#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_benchmark_cems.py

Benchmarks the FWI System computed in 03_compute_fwi.py against the
Copernicus Emergency Management Service (CEMS) historical fire danger
dataset (also ERA5-driven, from EWDS, downloaded by 01b).

Matching variables
    ours   CEMS
    fwi    fwinx      Fire Weather Index
    ffmc   ffmcode    Fine Fuel Moisture Code
    dmc    dufmcode   Duff Moisture Code
    dc     drtcode    Drought Code
    isi    infsinx    Initial Spread Index
    bui    fbupinx    Build-Up Index

Outputs
    data/processed/benchmark_cems_summary.csv   pooled statistics
    figures/benchmark_fwi_vs_cems.png           4-panel summary figure
    figures/benchmark_fwi_mean_maps.png         JJAS-mean FWI maps, ours vs CEMS
    with --smooth, each also as *_smooth.png (maps drawn as smooth contours;
    the cell versions at the true 0.25 deg resolution are always written)
Printed: correlation, bias and RMSE per variable and month, plus for the
FWI the quantities the final hazard product relies on (per-cell temporal
correlation, agreement of top-decile days, agreement of yearly counts).

Expected differences (so a perfect match is NOT expected)
  - CEMS runs continuously; this project restarts the moisture codes on
    1 April (FFMC 85, DMC 6, DC 15). The slow codes (DMC, DC, BUI) keep
    that starting offset: in dry weather they only accumulate, and are
    reduced only by rain, so the offset fades slowly (DC) or hardly at
    all (DMC) through summer. The fast code (FFMC) forgets it in days.
  - Our inputs are a single 20 UTC afternoon snapshot per day. CEMS may
    use different observation timing, so day-to-day differences remain.
  - Ocean cells are excluded: CEMS has no values there.

Usage:
    python 04_benchmark_cems.py --selftest    # check the statistics code
    python 04_benchmark_cems.py               # all years with both files
    python 04_benchmark_cems.py --year 2020   # one year
    python 04_benchmark_cems.py --smooth      # also write the smooth versions
                                              # of the map figures (display only)
"""

import argparse
import csv
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
FIG_DIR = PROJECT_ROOT / "figures"


# =====================================================
# SETTINGS
# =====================================================

VARMAP = {
    "fwi": "fwinx",
    "ffmc": "ffmcode",
    "dmc": "dufmcode",
    "dc": "drtcode",
    "isi": "infsinx",
    "bui": "fbupinx",
}

MONTHS = [4, 5, 6, 7, 8, 9]
JJAS = [6, 7, 8, 9]
EXCEED_PERCENTILE = 90.0
MIN_YEARS_FOR_COUNTS = 5


# =====================================================
# STATISTICS (numpy only, so they can be self-tested)
# =====================================================

def pearson_from_sums(n, sx, sy, sxx, syy, sxy):
    """Pearson r from running sums. Works on scalars or arrays."""
    num = n * sxy - sx * sy
    den = np.sqrt(
        np.maximum(n * sxx - sx ** 2, 0.0) * np.maximum(n * syy - sy ** 2, 0.0)
    )
    safe = np.where(den > 0, den, 1.0)
    return np.where(den > 0, num / safe, np.nan)


class Acc:
    """Running sums for pooled statistics of x (ours) against y (CEMS)."""

    def __init__(self):
        self.n = 0.0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def add(self, x, y):
        x = np.asarray(x, dtype="float64").ravel()
        y = np.asarray(y, dtype="float64").ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        self.n += x.size
        self.sx += x.sum()
        self.sy += y.sum()
        self.sxx += (x * x).sum()
        self.syy += (y * y).sum()
        self.sxy += (x * y).sum()

    def metrics(self):
        if self.n == 0:
            return dict(n=0, r=np.nan, bias=np.nan, rmse=np.nan,
                        mean_ours=np.nan, mean_cems=np.nan)
        r = float(pearson_from_sums(self.n, self.sx, self.sy,
                                    self.sxx, self.syy, self.sxy))
        mse = (self.sxx + self.syy - 2.0 * self.sxy) / self.n
        return dict(
            n=int(self.n),
            r=r,
            bias=(self.sx - self.sy) / self.n,
            rmse=float(np.sqrt(max(mse, 0.0))),
            mean_ours=self.sx / self.n,
            mean_cems=self.sy / self.n,
        )


class CellAcc:
    """Per-cell running sums over time (axis 0), grid shape (ny, nx)."""

    def __init__(self, shape):
        self.n = np.zeros(shape)
        self.sx = np.zeros(shape)
        self.sy = np.zeros(shape)
        self.sxx = np.zeros(shape)
        self.syy = np.zeros(shape)
        self.sxy = np.zeros(shape)

    def add(self, x, y):
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        ok = np.isfinite(x) & np.isfinite(y)
        x = np.where(ok, x, 0.0)
        y = np.where(ok, y, 0.0)
        self.n += ok.sum(axis=0)
        self.sx += x.sum(axis=0)
        self.sy += y.sum(axis=0)
        self.sxx += (x * x).sum(axis=0)
        self.syy += (y * y).sum(axis=0)
        self.sxy += (x * y).sum(axis=0)

    def r_map(self):
        r = pearson_from_sums(self.n, self.sx, self.sy,
                              self.sxx, self.syy, self.sxy)
        return np.where(self.n > 0, r, np.nan)

    def mean_maps(self):
        n = np.where(self.n > 0, self.n, np.nan)
        return self.sx / n, self.sy / n


def exceedance_stats(O, C, lengths, q=EXCEED_PERCENTILE):
    """
    O, C: arrays (days, cells), days concatenated year after year.
    lengths: number of days of each year, in order.
    Returns (hit_rate, chance_rate, median_annual_count_r, n_years).
    hit_rate: of the days our FWI is above its own per-cell q-th percentile,
    the fraction on which CEMS is also above its own percentile.
    """
    po = np.percentile(O, q, axis=0)
    pc = np.percentile(C, q, axis=0)
    eo = O > po
    ec = C > pc

    hit = (eo & ec).sum() / max(int(eo.sum()), 1)
    chance = 1.0 - q / 100.0

    edges = np.concatenate([[0], np.cumsum(lengths)])
    co = np.stack([eo[a:b].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    cc = np.stack([ec[a:b].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])

    if co.shape[0] < MIN_YEARS_FOR_COUNTS:
        return float(hit), chance, np.nan, co.shape[0]

    a = co - co.mean(axis=0)
    b = cc - cc.mean(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    ok = den > 0
    r = (a * b).sum(axis=0)[ok] / den[ok]
    med = float(np.median(r)) if r.size else np.nan
    return float(hit), chance, med, co.shape[0]


# =====================================================
# SELF-TEST
# =====================================================

def selftest():
    print("[INFO] Running benchmark self-test")
    rng = np.random.default_rng(0)
    fails = 0

    def ok(cond, msg):
        nonlocal fails
        print(f"[{'SUCCESS' if cond else 'FAILURE'}] {msg}")
        if not cond:
            fails += 1

    # pooled metrics against numpy, fed in chunks with NaNs
    x = rng.normal(20, 5, 5000)
    y = 0.8 * x + rng.normal(0, 2, 5000) + 3.0
    x[::97] = np.nan
    acc = Acc()
    for i in range(0, 5000, 1000):
        acc.add(x[i:i + 1000], y[i:i + 1000])
    m = acc.metrics()
    good = np.isfinite(x)
    ok(abs(m["r"] - np.corrcoef(x[good], y[good])[0, 1]) < 1e-9,
       f"pooled r {m['r']:.4f} matches numpy")
    ok(abs(m["bias"] - (x[good] - y[good]).mean()) < 1e-9, "bias matches numpy")
    ok(abs(m["rmse"] - np.sqrt(((x[good] - y[good]) ** 2).mean())) < 1e-9,
       "RMSE matches numpy")

    # per-cell correlation
    X = rng.normal(size=(200, 3, 4))
    Y = X * 2.0 + rng.normal(scale=0.1, size=X.shape)
    Y[:, 0, 0] = np.nan                        # an 'ocean' cell
    cell = CellAcc((3, 4))
    cell.add(X[:100], Y[:100])
    cell.add(X[100:], Y[100:])
    rm = cell.r_map()
    ref = np.corrcoef(X[:, 1, 2], Y[:, 1, 2])[0, 1]
    ok(abs(rm[1, 2] - ref) < 1e-9, "per-cell r matches numpy")
    ok(np.isnan(rm[0, 0]), "all-NaN cell gives NaN, not an error")

    # exceedance statistics: identical series must agree perfectly
    years, ndays, ncell = 8, 122, 6
    base = rng.gamma(2.0, 10.0, size=(years * ndays, ncell))
    hit, chance, med, ny = exceedance_stats(base, base.copy(), [ndays] * years)
    ok(abs(hit - 1.0) < 1e-12, f"identical series: hit rate {hit:.3f} = 1")
    ok(abs(med - 1.0) < 1e-9 or np.isnan(med),
       f"identical series: yearly-count r {med:.3f} = 1")
    # independent series must score near chance
    other = rng.gamma(2.0, 10.0, size=base.shape)
    hit, chance, med, ny = exceedance_stats(base, other, [ndays] * years)
    ok(abs(hit - chance) < 0.04, f"independent series: hit {hit:.3f} near chance {chance:.2f}")

    if fails:
        print(f"[FAILURE] Self-test failed ({fails} checks)")
        return 1
    print("[SUCCESS] Self-test passed")
    return 0


# =====================================================
# DATA LOADING
# =====================================================

def load_pair(year):
    """Returns (ours Dataset, dict of CEMS arrays aligned to ours) or None."""
    import pandas as pd
    import xarray as xr

    ours_path = PROCESSED_DIR / f"fwi_{year}.nc"
    cems_path = RAW_DIR / f"cems_fwi_{year}_AprSep.nc"
    if not ours_path.exists() or not cems_path.exists():
        return None

    ours = xr.open_dataset(ours_path, engine="netcdf4").load()
    cems = xr.open_dataset(cems_path, engine="netcdf4").load()

    tname = "valid_time" if "valid_time" in cems.coords else "time"
    cems = cems.rename({tname: "date"})
    cems = cems.assign_coords(date=pd.DatetimeIndex(cems["date"].values).normalize())

    lon = ((cems["longitude"].values + 180.0) % 360.0) - 180.0
    cems = cems.assign_coords(longitude=lon).sortby("longitude")

    cems = cems.sel(date=ours["date"].values)
    cems = cems.reindex(
        latitude=ours["latitude"].values,
        longitude=ours["longitude"].values,
        method="nearest",
        tolerance=0.01,
    )
    arrays = {v: cems[c].values for v, c in VARMAP.items()}
    return ours, arrays


def find_years():
    years = set()
    for p in PROCESSED_DIR.glob("fwi_*.nc"):
        m = re.match(r"fwi_(\d{4})\.nc", p.name)
        if m and (RAW_DIR / f"cems_fwi_{m.group(1)}_AprSep.nc").exists():
            years.add(int(m.group(1)))
    return sorted(years)


# =====================================================
# FIGURES
# =====================================================

COUNT_LEVELS = [1, 10, 100, 1000, 10000, 100000]
R_LEVELS = [0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 1.0]
MEAN_LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]

def _footer(fig, extra=""):
    import map_utils as mu

    mu.footer(fig, "Data: ERA5 (C3S); CEMS fire danger (EWDS).", extra)


def make_figure(fig_path, O, C, land, domain_series, r_map, diff_map, lat, lon,
                smooth=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import map_utils as mu

    fig = plt.figure(figsize=(13, 10.5))
    extent = mu.grid_extent(lat, lon)

    # (a) daily JJAS FWI, ours vs CEMS (random subsample, land cells)
    ax = fig.add_subplot(2, 2, 1)
    rng = np.random.default_rng(1)
    o = O[:, land].ravel()
    c = C[:, land].ravel()
    ok = np.isfinite(o) & np.isfinite(c)
    o, c = o[ok], c[ok]
    idx = rng.choice(o.size, size=min(400000, o.size), replace=False)
    lim = float(np.percentile(np.concatenate([o[idx], c[idx]]), 99.9))
    cmap, norm = mu.discrete_cmap("viridis", COUNT_LEVELS, "max")
    hb = ax.hexbin(c[idx], o[idx], gridsize=60, extent=(0, lim, 0, lim),
                   mincnt=1, cmap=cmap, norm=norm)
    ax.plot([0, lim], [0, lim], color="k", lw=1, ls="--")
    ax.set_xlabel("CEMS FWI")
    ax.set_ylabel("This project FWI")
    ax.set_title("Daily FWI, JJAS, all years (land cells)")
    mu.add_colorbar(fig, hb, ax, "points per bin", COUNT_LEVELS, fmt="%d")

    # (b) mean seasonal cycle of domain-mean FWI
    ax = fig.add_subplot(2, 2, 2)
    mo = np.nanmean(np.stack([s[0] for s in domain_series]), axis=0)
    mc = np.nanmean(np.stack([s[1] for s in domain_series]), axis=0)
    days = np.arange(mo.size)
    ax.plot(days, mc, label="CEMS", lw=2)
    ax.plot(days, mo, label="This project", lw=2)
    ax.set_xticks(np.cumsum([0, 30, 31, 30, 31, 31]))
    ax.set_xticklabels(["Apr", "May", "Jun", "Jul", "Aug", "Sep"])
    ax.set_ylabel("Domain-mean FWI (land)")
    ax.set_title("Mean seasonal cycle (spin-up visible in April)")
    ax.legend()

    # (c) per-cell temporal correlation, binned
    ax = mu.map_axes(fig, 2, 2, 3, extent)
    cmap, norm = mu.discrete_cmap("viridis", R_LEVELS, "min")
    mesh = mu.draw_field(ax, lon, lat, np.clip(r_map, None, 1.0), cmap, norm,
                         smooth=smooth)
    ax.set_title("Per-cell correlation of daily FWI, JJAS")
    mu.add_colorbar(fig, mesh, ax, "Pearson r", R_LEVELS)

    # (d) mean difference, binned and symmetric around zero
    ax = mu.map_axes(fig, 2, 2, 4, extent)
    d_levels = mu.nice_symmetric_levels(diff_map)
    cmap, norm = mu.discrete_cmap("RdBu_r", d_levels, "both")
    mesh = mu.draw_field(ax, lon, lat, diff_map, cmap, norm, smooth=smooth)
    ax.set_title("Mean JJAS FWI: this project minus CEMS")
    mu.add_colorbar(fig, mesh, ax, "FWI difference", d_levels)

    _footer(fig, mu.SMOOTH_NOTE if smooth else "")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mu.save_figure(fig, fig_path, dpi=170)
    plt.close(fig)


def make_mean_maps(fig_path, mean_ours, mean_cems, lat, lon, smooth=False):
    """
    Side-by-side JJAS-mean FWI maps with one shared binned colorbar.
    smooth=True draws filled contours instead of one block per grid cell.
    That looks nicer but interpolates between the 0.25 deg cells, so it is
    for presentation only; the cell version shows the true resolution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import map_utils as mu

    fig = plt.figure(figsize=(13, 5.8))
    extent = mu.grid_extent(lat, lon)
    cmap, norm = mu.discrete_cmap("YlOrRd", MEAN_LEVELS, "max")

    axes = []
    for i, (data, title) in enumerate(
        [(mean_ours, "This project"), (mean_cems, "CEMS")], start=1
    ):
        ax = mu.map_axes(fig, 1, 2, i, extent)
        mesh = mu.draw_field(ax, lon, lat, data, cmap, norm, smooth=smooth)
        ax.set_title(f"Mean JJAS FWI: {title}")
        axes.append(ax)
    mu.add_colorbar(fig, mesh, axes, "Mean FWI", MEAN_LEVELS)

    _footer(fig, mu.SMOOTH_NOTE if smooth else "")
    mu.save_figure(fig, fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--smooth", action="store_true",
                        help="also write smooth (contour) mean maps")
    args = parser.parse_args()

    print("=" * 60)
    print("BENCHMARK AGAINST CEMS")
    print("=" * 60)

    if args.selftest:
        return selftest()

    years = [args.year] if args.year else find_years()
    if not years:
        print("[FAILURE] No year has both data/processed/fwi_<y>.nc and the CEMS file.")
        return 1
    print(f"[INFO] Years: {years[0]}-{years[-1]} ({len(years)} years)")

    accs = {(v, k): Acc() for v in VARMAP for k in MONTHS + ["JJAS"]}
    cells = {}
    O_list, C_list, lengths, domain_series = [], [], [], []
    lat = lon = None

    for year in years:
        try:
            pair = load_pair(year)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAILURE] Year {year}: {exc}")
            return 1
        if pair is None:
            print(f"[WARNING] Year {year}: files missing, skipped")
            continue
        ours, cems = pair
        if lat is None:
            lat, lon = ours["latitude"].values, ours["longitude"].values
            shape = (len(lat), len(lon))
            cells = {v: CellAcc(shape) for v in VARMAP}

        months = ours["date"].dt.month.values
        jj = np.isin(months, JJAS)

        for var in VARMAP:
            x = ours[var].values
            y = cems[var]
            for m in MONTHS:
                sel = months == m
                accs[(var, m)].add(x[sel], y[sel])
            accs[(var, "JJAS")].add(x[jj], y[jj])
            cells[var].add(x[jj], y[jj])

            if var == "fwi":
                O_list.append(x[jj].astype("float32"))
                C_list.append(y[jj].astype("float32"))
                lengths.append(int(jj.sum()))
                both = np.isfinite(x) & np.isfinite(y)
                xm = np.where(both, x, np.nan)
                ym = np.where(both, y, np.nan)
                domain_series.append(
                    (np.nanmean(xm, axis=(1, 2)), np.nanmean(ym, axis=(1, 2)))
                )
        print(f"[INFO] Year {year} processed")

    # ---------- table ----------
    rows = []
    print("=" * 60)
    print(f"{'var':<5}{'period':<7}{'n':>10}{'r':>8}{'bias':>9}"
          f"{'rmse':>9}{'ours':>9}{'cems':>9}")
    for var in VARMAP:
        for key in MONTHS + ["JJAS"]:
            m = accs[(var, key)].metrics()
            label = key if key == "JJAS" else f"{key:02d}"
            rows.append([var, label, m["n"], m["r"], m["bias"], m["rmse"],
                         m["mean_ours"], m["mean_cems"]])
            print(f"{var:<5}{label:<7}{m['n']:>10}{m['r']:>8.3f}{m['bias']:>9.2f}"
                  f"{m['rmse']:>9.2f}{m['mean_ours']:>9.2f}{m['mean_cems']:>9.2f}")
        print("-" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROCESSED_DIR / "benchmark_cems_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variable", "period", "n", "pearson_r", "bias_ours_minus_cems",
                    "rmse", "mean_ours", "mean_cems"])
        w.writerows(rows)
    print(f"[INFO] Wrote {csv_path.name}")

    # ---------- FWI-specific diagnostics ----------
    r_map = cells["fwi"].r_map()
    mo, mc = cells["fwi"].mean_maps()
    diff_map = mo - mc

    valid = np.isfinite(r_map)
    print("=" * 60)
    print(f"[INFO] FWI, JJAS, land cells: {int(valid.sum())}")
    print(f"[INFO] Per-cell daily correlation: median {np.nanmedian(r_map):.3f}, "
          f"10th percentile {np.nanpercentile(r_map, 10):.3f}")
    both = np.isfinite(mo) & np.isfinite(mc)
    if both.sum() > 2:
        sp = np.corrcoef(mo[both], mc[both])[0, 1]
        print(f"[INFO] Spatial correlation of JJAS-mean FWI maps: {sp:.3f}")

    O = np.concatenate(O_list, axis=0)
    C = np.concatenate(C_list, axis=0)
    land = np.isfinite(C).all(axis=0) & np.isfinite(O).all(axis=0)
    hit, chance, med, ny = exceedance_stats(
        O[:, land], C[:, land], lengths, EXCEED_PERCENTILE
    )
    print(f"[INFO] Top-{100 - EXCEED_PERCENTILE:.0f}% days (each cell's own "
          f"{EXCEED_PERCENTILE:.0f}th percentile): ours flags CEMS-flagged days "
          f"{hit:.1%} of the time (chance {chance:.0%})")
    if np.isfinite(med):
        print(f"[INFO] Yearly count of such days, ours vs CEMS: median per-cell "
              f"correlation {med:.3f} over {ny} years")
    else:
        print(f"[INFO] Yearly-count correlation needs >= {MIN_YEARS_FOR_COUNTS} "
              f"years (have {ny})")

    import map_utils as mu

    variants = [False, True] if args.smooth else [False]
    for sm in variants:
        figures = (
            ("benchmark_fwi_vs_cems.png",
             lambda p, sm=sm: make_figure(p, O, C, land, domain_series,
                                          r_map, diff_map, lat, lon, smooth=sm)),
            ("benchmark_fwi_mean_maps.png",
             lambda p, sm=sm: make_mean_maps(p, mo, mc, lat, lon, smooth=sm)),
        )
        for fname, func in figures:
            path = FIG_DIR / mu.variant_name(fname, sm)
            try:
                func(path)
                print(f"[SUCCESS] Figure: {path}")
            except Exception as exc:  # noqa: BLE001
                print(f"[WARNING] {path.name} not written: {exc}")

    print("=" * 60)
    print("[SUCCESS] Benchmark finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
