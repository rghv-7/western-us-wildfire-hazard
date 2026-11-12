#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_fire_overlay.py

Compares the fire-weather hazard from 05_hazard_climatology.py with where fires
actually burned, using NIFC fire perimeters (from 06a_download_fire_perimeters.py)
for the Western US study states (CA, OR, WA, NV, ID, AZ).

Steps
  1. Read the two perimeter files: the history layer (2001-2019, year from
     FIRE_YEAR_INT) and WFIGS (2020 onward, year from the discovery date).
     Keep wildfires of at least --min-acres; drop prescribed burns and records
     flagged for deletion. The load summary shows what was kept and dropped.
  2. Rasterise the perimeters onto the 0.25 deg ERA5 grid: burned fraction per
     cell and year. Overlapping or duplicated perimeters count once.
  3. Compare with days above each cell's own P90 (hazard_climatology.nc).

Outputs
    figures/fire_overlay_years.png    hazard map with that year's perimeters on it
    figures/fire_gain_curve.png       share of burned area against hazard rank
    figures/fire_state_scatter.png    state-year burned area vs days above P90
    data/processed/fire_state_annual.csv      state x year burned km2 and hazard
    data/processed/fire_hazard_summary.csv    rank correlations and gain-curve scores
    data/processed/fire_burned_fraction.nc    burned fraction per cell and year
    with --smooth, the overlay figure is also written as *_smooth.png

How to read the results
  Gain curve: cell-years are ranked from the highest to the lowest hazard (days
  above the cell's own P90 in that year). If hazard says nothing about fire,
  the curve follows the diagonal (Gini 0). Gini is 2 x (area under the curve)
  - 1, with a 95% interval from resampling whole years.
  State scatter: Spearman rank correlation over the years. Both burned area
  and hazard may share a trend, so a detrended version is reported as well.

Limits (they apply to every figure)
  FWI is weather only: no fuels, ignition, suppression or exposure. Perimeters
  are final footprints without reliable fire dates in the history layer, so the
  burned area includes fires outside June-September, among them the wind-driven
  autumn fires that the JJAS window does not describe. Coverage differs by
  contributing agency, so burned area is a lower bound. A correlation is not
  proof of cause, and hazard is not loss.

Usage:
    python 06_fire_overlay.py --selftest
    python 06_fire_overlay.py
    python 06_fire_overlay.py --years 2018 2020 2021 2022 --smooth
"""

import argparse
import importlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


# =====================================================
# PATHS AND SETTINGS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
FIG_DIR = PROJECT_ROOT / "figures"

HAZARD_NC = PROCESSED_DIR / "hazard_climatology.nc"
HIST_PATH = EXTERNAL_DIR / "InterAgencyFirePerimeterHistory_All_Years_View.geojson"
WFIGS_PATH = EXTERNAL_DIR / "WFIGS_Interagency_Perimeters.geojson"

OUT_ANNUAL = PROCESSED_DIR / "fire_state_annual.csv"
OUT_SUMMARY = PROCESSED_DIR / "fire_hazard_summary.csv"
OUT_NC = PROCESSED_DIR / "fire_burned_fraction.nc"

HIST_FIRST, HIST_LAST = 2001, 2019   # years taken from the history layer
WFIGS_FIRST = 2020                   # years taken from WFIGS
WF_DATE_FIELDS = ("attr_FireDiscoveryDateTime", "poly_PolygonDateTime",
                  "poly_CreateDate")
WF_ACRE_FIELDS = ("poly_GISAcres", "poly_Acres_AutoCalc", "attr_FinalAcres",
                  "attr_IncidentSize")
DELETE_FLAGS = {"yes", "y", "true", "1"}

ACRE_KM2 = 0.00404686
EARTH_RADIUS_KM = 6371.0088

N_PERM = 10000
N_BOOT = 1000
MIN_YEARS = 8

PERIM_FACE = (0.04, 0.12, 0.45, 0.28)
PERIM_EDGE = "#0b1f66"

CAVEAT = ("FWI = weather only (no fuels, ignition or exposure); hazard is not loss. "
          "Perimeters: NIFC, wildfires only, coverage varies by agency "
          "(burned area is a lower bound and includes fires outside Jun-Sep).")


def _hz():
    """The hazard script (its file name starts with a digit, so import by name)."""
    return importlib.import_module("05_hazard_climatology")


# =====================================================
# READING THE PERIMETERS
# =====================================================

def to_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def parse_date(value):
    """datetime from an ArcGIS date (epoch milliseconds or ISO text), else None."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        if value != value:
            return None
        try:
            return datetime(1970, 1, 1) + timedelta(milliseconds=float(value))
        except (OverflowError, OSError, ValueError):
            return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(value))
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def parse_geometry(geom):
    """
    GeoJSON Polygon / MultiPolygon -> list of (exterior, holes), each ring a
    float array (n, 2) of lon, lat. Rings with fewer than 4 points are dropped.
    """
    if not geom:
        return []
    kind, coords = geom.get("type"), geom.get("coordinates")
    if kind == "Polygon":
        parts = [coords]
    elif kind == "MultiPolygon":
        parts = coords
    else:
        return []
    out = []
    for part in parts or []:
        rings = [np.asarray(r, dtype=float)[:, :2] for r in part if len(r) >= 4]
        if rings:
            out.append((rings[0], rings[1:]))
    return out


def load_geojson(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("features", [])


def history_records(features, min_acres):
    """Wildfire polygons of the history layer (2001-2019). Returns kept, dropped, feature counts."""
    kept, dropped, feat_counts = [], Counter(), Counter()
    for f in features:
        p = f.get("properties") or {}
        feat = str(p.get("FEATURE_CA") or "").strip()
        feat_counts[feat or "(empty)"] += 1
        if "prescribed" in feat.lower():
            dropped["prescribed burn"] += 1
            continue
        year = to_float(p.get("FIRE_YEAR_INT"))
        if year is None:
            year = to_float(p.get("FIRE_YEAR"))
        if year is None:
            dropped["no year"] += 1
            continue
        year = int(year)
        if not HIST_FIRST <= year <= HIST_LAST:
            dropped[f"outside {HIST_FIRST}-{HIST_LAST}"] += 1
            continue
        acres = to_float(p.get("GIS_ACRES"))
        if acres is None or acres < min_acres:
            dropped["below min acres"] += 1
            continue
        polys = parse_geometry(f.get("geometry"))
        if not polys:
            dropped["no polygon"] += 1
            continue
        kept.append({"year": year, "acres": acres, "month": None,
                     "name": str(p.get("INCIDENT") or "").strip(),
                     "polys": polys, "source": "history"})
    return kept, dropped, feat_counts


def wfigs_records(features, min_acres):
    """Wildfire polygons of WFIGS (2020 onward). Returns kept, dropped, value counts."""
    kept, dropped = [], Counter()
    type_counts, feat_counts = Counter(), Counter()
    for f in features:
        p = f.get("properties") or {}
        cat = str(p.get("attr_IncidentTypeCategory") or "").strip().upper()
        feat = str(p.get("poly_FeatureCategory") or "").strip()
        type_counts[cat or "(empty)"] += 1
        feat_counts[feat or "(empty)"] += 1
        if cat == "RX" or "prescribed" in feat.lower():
            dropped["prescribed burn"] += 1
            continue
        if str(p.get("poly_DeleteThis") or "").strip().lower() in DELETE_FLAGS:
            dropped["flagged for deletion"] += 1
            continue
        date = None
        for key in WF_DATE_FIELDS:
            date = parse_date(p.get(key))
            if date is not None:
                break
        if date is None:
            dropped["no date"] += 1
            continue
        if date.year < WFIGS_FIRST:
            dropped[f"before {WFIGS_FIRST} (history layer used)"] += 1
            continue
        acres = None
        for key in WF_ACRE_FIELDS:
            acres = to_float(p.get(key))
            if acres:
                break
        if not acres or acres < min_acres:
            dropped["below min acres"] += 1
            continue
        polys = parse_geometry(f.get("geometry"))
        if not polys:
            dropped["no polygon"] += 1
            continue
        known = parse_date(p.get("attr_FireDiscoveryDateTime"))
        name = str(p.get("poly_IncidentName") or p.get("attr_IncidentName") or "")
        kept.append({"year": date.year, "acres": acres,
                     "month": known.month if known else None,
                     "name": name.strip(), "polys": polys, "source": "WFIGS"})
    return kept, dropped, type_counts, feat_counts


def fire_groups(records, lat, lon_w, study):
    """
    One entry per (year, fire name): largest acreage, a label position and
    whether it lies in the study area. Used for counts and labels only; the
    burned area comes from the rasterised union of all polygons.
    """
    dlat = float(np.median(np.abs(np.diff(lat))))
    dlon = float(np.median(np.abs(np.diff(lon_w))))
    groups = {}
    for k, r in enumerate(records):
        key = (r["year"], r["name"].lower() if r["name"] else f"#{k}")
        best = groups.get(key)
        if best is not None and best["acres"] >= r["acres"]:
            continue
        ext = max((p[0] for p in r["polys"]), key=len)
        cx, cy = float(ext[:, 0].mean()), float(ext[:, 1].mean())
        iy = int(np.argmin(np.abs(lat - cy)))
        ix = int(np.argmin(np.abs(lon_w - cx)))
        inside = (abs(lat[iy] - cy) <= dlat and abs(lon_w[ix] - cx) <= dlon
                  and bool(study[iy, ix]))
        groups[key] = {"year": r["year"], "name": r["name"], "acres": r["acres"],
                       "cx": cx, "cy": cy, "in_study": inside}
    by_year = defaultdict(list)
    for g in groups.values():
        by_year[g["year"]].append(g)
    return by_year


# =====================================================
# RASTERISING AND STATISTICS (numpy and matplotlib only, tested by --selftest)
# =====================================================

def burned_fraction(polys, lat, lon, sub=8):
    """
    Fraction of every grid cell covered by at least one polygon, estimated on a
    sub x sub sample of points per cell. polys: list of (exterior, holes).
    Overlapping polygons are not double counted.
    """
    from matplotlib.path import Path as MplPath

    lat = np.asarray(lat, dtype=float)
    lon_w = np.where(np.asarray(lon, dtype=float) > 180, np.asarray(lon) - 360, lon)
    dlat = float(np.median(np.abs(np.diff(lat))))
    dlon = float(np.median(np.abs(np.diff(lon_w))))
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    fine_lat = (lat[:, None] + offs[None, :] * dlat).ravel()
    fine_lon = (lon_w[:, None] + offs[None, :] * dlon).ravel()
    fine = np.zeros((fine_lat.size, fine_lon.size), dtype=bool)
    for ext, holes in polys:
        iy = np.nonzero((fine_lat >= ext[:, 1].min()) & (fine_lat <= ext[:, 1].max()))[0]
        ix = np.nonzero((fine_lon >= ext[:, 0].min()) & (fine_lon <= ext[:, 0].max()))[0]
        if iy.size == 0 or ix.size == 0:
            continue
        gx, gy = np.meshgrid(fine_lon[ix], fine_lat[iy])
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        inside = MplPath(ext).contains_points(pts)
        for hole in holes:
            inside &= ~MplPath(hole).contains_points(pts)
        fine[np.ix_(iy, ix)] |= inside.reshape(iy.size, ix.size)
    return fine.reshape(len(lat), sub, len(lon_w), sub).mean(axis=(1, 3)).astype("float32")


def cell_area_km2(lat, dlat, dlon):
    """Area (km2) of a dlat x dlon cell for each latitude (spherical Earth)."""
    phi = np.deg2rad(np.asarray(lat, dtype=float))
    half = np.deg2rad(dlat) / 2.0
    return EARTH_RADIUS_KM ** 2 * np.deg2rad(dlon) * (np.sin(phi + half) - np.sin(phi - half))


def rankdata(a):
    """Ranks starting at 1, ties get the average rank."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(a.size)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_perm(x, y, n_perm=N_PERM, seed=0):
    """Spearman rho and a two-sided permutation p-value."""
    rho = spearman(x, y)
    if not np.isfinite(rho):
        return rho, float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((n_perm, rx.size)), axis=1)
    cx = rx - rx.mean()
    cy = ry[perm] - ry.mean()
    r = (cx * cy).sum(axis=1) / (rx.size * rx.std() * ry.std())
    p = (1.0 + np.sum(np.abs(r) >= abs(rho) - 1e-12)) / (n_perm + 1.0)
    return rho, float(p)


def detrend(y, x):
    """y minus its Theil-Sen line (minus the mean if the series is too short)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    slope = _hz().trend_test(y, x)[0]
    if not np.isfinite(slope):
        return y - y.mean()
    return y - (slope * x + np.median(y - slope * x))


def gain_points(n_by_value, b_by_value):
    """
    Cumulative area share (x) and burned-area share (y), cells ranked from the
    highest to the lowest hazard value. n_by_value[v], b_by_value[v]: area and
    burned area summed over cells with hazard value v. Starts at (0, 0).
    """
    n = np.asarray(n_by_value, dtype=float)[::-1]
    b = np.asarray(b_by_value, dtype=float)[::-1]
    keep = n > 0
    n, b = n[keep], b[keep]
    if n.sum() <= 0 or b.sum() <= 0:
        return None
    x = np.concatenate([[0.0], np.cumsum(n) / n.sum()])
    y = np.concatenate([[0.0], np.cumsum(b) / b.sum()])
    return x, y


def gain_stats(n_by_value, b_by_value):
    """(Gini, share of burned area in the highest-hazard 20% of the area)."""
    pts = gain_points(n_by_value, b_by_value)
    if pts is None:
        return float("nan"), float("nan")
    x, y = pts
    area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    return 2.0 * area - 1.0, float(np.interp(0.2, x, y))


def gain_bootstrap(n_yv, b_yv, n_boot=N_BOOT, seed=0):
    """95% intervals (Gini low, high, top-20% low, high) from resampling years."""
    rng = np.random.default_rng(seed)
    ny = n_yv.shape[0]
    g, t = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, ny, ny)
        gi, ti = gain_stats(n_yv[idx].sum(axis=0), b_yv[idx].sum(axis=0))
        if np.isfinite(gi):
            g.append(gi)
            t.append(ti)
    if not g:
        return (float("nan"),) * 4
    return (float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5)),
            float(np.percentile(t, 2.5)), float(np.percentile(t, 97.5)))


def hazard_bins(days_year, area_cells, burned_km2, mask):
    """Area and burned area summed by hazard value (days above P90) for one year."""
    hz = _hz()
    valid = mask & np.isfinite(days_year)
    v = np.clip(np.nan_to_num(days_year[valid]), 0, hz.N_DAYS).astype(int)
    n = np.bincount(v, weights=area_cells[valid], minlength=hz.N_DAYS + 1)
    b = np.bincount(v, weights=burned_km2[valid], minlength=hz.N_DAYS + 1)
    return n, b


# =====================================================
# FIGURES
# =====================================================

def rings_to_path(polys):
    """One matplotlib Path of all outer rings (holes ignored: display only)."""
    from matplotlib.path import Path as MplPath

    verts, codes = [], []
    for ext, _ in polys:
        c = np.full(len(ext), MplPath.LINETO, dtype=MplPath.code_type)
        c[0] = MplPath.MOVETO
        c[-1] = MplPath.CLOSEPOLY
        verts.append(ext)
        codes.append(c)
    return MplPath(np.concatenate(verts), np.concatenate(codes))


def pick_labels(candidates, n, min_sep=2.8):
    """
    The n largest fires to name on a map, skipping any whose label position lies
    within min_sep degrees of a larger fire already named (labels would overlap).
    """
    chosen = []
    for g in sorted(candidates, key=lambda g: -g["acres"]):
        if len(chosen) >= n:
            break
        if all(math.hypot(g["cx"] - c["cx"], g["cy"] - c["cy"]) >= min_sep
               for c in chosen):
            chosen.append(g)
    return chosen


def _pfmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def make_overlay_figure(fig_path, lon, lat, show, days, years, picks, perim,
                        groups, burned_total, min_acres, label_top,
                        smooth=False, clip=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, PathPatch
    import map_utils as mu

    hz = _hz()
    n = len(picks)
    extent = mu.grid_extent(lat, lon)
    fig = plt.figure(figsize=(4.2 * n + 1.4, 5.8))
    cmap, norm = mu.discrete_cmap("YlOrRd", hz.YEAR_LEVELS, "neither")
    axes, mesh = [], None
    for k, year in enumerate(picks, start=1):
        field = np.where(show, days[years.index(year)].astype(float), np.nan)
        ax = mu.map_axes(fig, 1, n, k, extent)
        mesh = mu.draw_field(ax, lon, lat, field, cmap, norm, smooth=smooth,
                             clip=clip)
        if year in perim:
            patch = PathPatch(perim[year], facecolor=PERIM_FACE,
                              edgecolor=PERIM_EDGE, linewidth=0.7, zorder=4)
            ax.add_patch(patch)
            if clip is not None:
                patch.set_clip_path(PathPatch(clip, transform=ax.transData,
                                              facecolor="none", edgecolor="none"))
        inside = [g for g in groups.get(year, []) if g["in_study"]]
        for g in pick_labels(inside, label_top):
            label = f"{g['name'].title() or 'unnamed'}\n{g['acres'] / 1000:,.0f}k ac"
            ax.text(g["cx"], g["cy"], label, fontsize=6.5, ha="center",
                    va="center", zorder=6, clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                              alpha=0.75))
        ax.set_title(f"{year}\n{len(inside)} fires, "
                     f"{burned_total.get(year, 0):,.0f} km\u00b2 burned", fontsize=9)
        axes.append(ax)
    axes[0].legend(handles=[Patch(facecolor=PERIM_FACE, edgecolor=PERIM_EDGE,
                                  label=f"fire perimeter, {min_acres:,.0f}+ acres")],
                   loc="lower left", fontsize=7, framealpha=0.85)
    mu.add_colorbar(fig, mesh, axes, "days above the cell's own P90 (JJAS)",
                    hz.YEAR_LEVELS, fmt="%d")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mu.save_figure(fig, fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_gain_figure(fig_path, curves, stats, y0, y1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import map_utils as mu

    hz = _hz()
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot([0, 1], [0, 1], color="0.4", lw=1, ls=":", label="no skill (diagonal)")
    for name, pts in curves.items():
        if pts is None:
            continue
        pooled = name == "ALL"
        gini, top20 = stats[name][:2]
        label = (f"all study states: top 20% holds {top20:.0%}, Gini {gini:.2f}"
                 if pooled else f"{name}: {top20:.0%}")
        ax.plot(pts[0], pts[1], lw=2.8 if pooled else 1.3,
                color="0.1" if pooled else hz.STATE_COLORS.get(name, hz.DEFAULT_COLOR),
                label=label, zorder=3 if pooled else 2)
    ax.axvline(0.2, color="0.8", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("cumulative share of area, highest to lowest hazard")
    ax.set_ylabel("cumulative share of burned area")
    ax.set_title(f"Where did the burned area fall? Cell-years {y0}-{y1}\n"
                 "(hazard = days above the cell's own P90 that year)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ci = stats.get("ALL")
    if ci is not None and np.isfinite(ci[2]):
        ax.text(0.02, 0.97, f"Gini 95% interval {ci[2]:.2f} to {ci[3]:.2f} "
                            "(resampling years)", transform=ax.transAxes,
                fontsize=8, va="top", color="0.25")
    fig.text(0.01, 0.005, CAVEAT, fontsize=6.5, color="0.4", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mu.save_figure(fig, fig_path, dpi=170)
    plt.close(fig)


def make_scatter_figure(fig_path, years, series, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import map_utils as mu

    hz = _hz()
    names = list(series)
    ncols = min(3, len(names))
    nrows = int(math.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows),
                             squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        haz, burned = series[name]
        y = np.maximum(burned, 1.0)
        ax.scatter(haz, y, s=26, color=hz.STATE_COLORS.get(name, hz.DEFAULT_COLOR),
                   edgecolor="white", linewidth=0.5, zorder=3)
        for k in np.argsort(-burned)[:3]:
            ax.annotate(str(years[k]), (haz[k], y[k]), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, color="0.25")
        ax.axvline(0.1 * hz.N_DAYS, color="0.6", lw=1, ls=":")
        ax.set_yscale("log")
        fmt = FuncFormatter(lambda v, _: f"{v:,.0f}")
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(fmt)
        ax.tick_params(axis="y", which="minor", labelsize=7)
        r = results[name]
        ax.set_title(f"{name}: rho = {r['rho']:+.2f} ({_pfmt(r['p'])}); "
                     f"detrended {r['rho_d']:+.2f}", fontsize=9)
        ax.set_xlabel("days above own P90 (state mean)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="0.92", lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("burned area, km2 (log; < 1 shown at 1)")
    fig.suptitle("Burned area against fire-weather hazard, one point per year "
                 "(dotted: 12.2 days expected)", fontsize=11)
    fig.text(0.01, 0.005, CAVEAT, fontsize=6.5, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
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

    # dates
    check(parse_date(1597017600000).year == 2020, "epoch milliseconds -> 2020")
    check(parse_date("2019-07-04T12:00:00Z").month == 7, "ISO text parsed")
    check(parse_date(None) is None and parse_date("") is None
          and parse_date("n/a") is None, "missing dates give None")

    # geometry
    ring = [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]
    hole = [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]
    poly = parse_geometry({"type": "Polygon", "coordinates": [ring, hole]})
    multi = parse_geometry({"type": "MultiPolygon",
                            "coordinates": [[ring], [ring[:3]], [ring, hole]]})
    check(len(poly) == 1 and len(poly[0][1]) == 1, "polygon with a hole parsed")
    check(len(multi) == 2, "multipolygon: short ring dropped")
    check(parse_geometry(None) == [] and parse_geometry({"type": "Point"}) == [],
          "non-polygon geometry ignored")

    # rasterising: cell edges at 0.125 offsets, descending latitude
    lat = np.arange(40.0, 37.99, -0.25)
    lon = np.arange(-120.0, -117.99, 0.25)
    sq = np.array([[-119.625, 38.875], [-118.625, 38.875], [-118.625, 39.875],
                   [-119.625, 39.875], [-119.625, 38.875]])
    f1 = burned_fraction([(sq, [])], lat, lon, sub=4)
    check(abs(float(f1.sum()) - 16.0) < 1e-4 and float(f1.max()) <= 1.0,
          "1 deg square covers 16 cells exactly")
    f2 = burned_fraction([(sq, []), (sq, [])], lat, lon, sub=4)
    check(np.allclose(f1, f2), "duplicate polygons are not double counted")
    h = np.array([[-119.375, 39.125], [-118.875, 39.125], [-118.875, 39.625],
                  [-119.375, 39.625], [-119.375, 39.125]])
    f3 = burned_fraction([(sq, [h])], lat, lon, sub=4)
    check(abs(float(f3.sum()) - 12.0) < 1e-4, "a 2 x 2 cell hole removes 4 cells")
    f4 = burned_fraction([(sq, [])], lat, lon + 360.0, sub=4)
    check(np.allclose(f1, f4), "longitudes in 0-360 are handled")
    half = np.array([[-119.5, 39.0], [-119.0, 39.0], [-119.0, 39.5], [-119.5, 39.5],
                     [-119.5, 39.0]])
    check(abs(float(burned_fraction([(half, [])], lat, lon, sub=8).sum()) - 4.0) < 0.3,
          "partial cell cover adds up (0.5 deg square = 4 cells)")

    # areas
    a = cell_area_km2(np.array([0.0, 60.0]), 0.25, 0.25)
    check(abs(a[0] - 772.8) < 8.0 and abs(a[1] / a[0] - 0.5) < 0.01,
          f"cell area {a[0]:.0f} km2 at the equator, half at 60 N")

    # ranks and correlation
    check(np.allclose(rankdata([1, 2, 2, 3]), [1, 2.5, 2.5, 4]), "tied ranks averaged")
    x = np.arange(22.0)
    check(abs(spearman(x, x ** 2) - 1.0) < 1e-12 and abs(spearman(x, -x) + 1.0) < 1e-12,
          "Spearman is +1 / -1 for monotone series")
    rho, p = spearman_perm(x, x ** 2)
    check(p < 0.001, "monotone series: tiny permutation p")
    rng = np.random.default_rng(1)
    ps = [spearman_perm(rng.normal(size=22), rng.normal(size=22), n_perm=999, seed=i)[1]
          for i in range(150)]
    check(0.02 < np.mean(np.array(ps) < 0.05) < 0.10,
          f"independent noise: {np.mean(np.array(ps) < 0.05):.2f} below p 0.05 (about 0.05)")
    yrs = np.arange(2001, 2023)
    d = detrend(3.0 * (yrs - 2001) + rng.normal(0, 0.1, yrs.size), yrs)
    check(abs(np.polyfit(yrs, d, 1)[0]) < 0.05, "detrend removes a linear trend")

    # gain curve
    n = np.ones(123)
    perfect = np.zeros(123)
    perfect[122] = 1.0
    g, t = gain_stats(n, perfect)
    check(g > 0.98 and abs(t - 1.0) < 1e-9, "all burning at the top hazard: Gini near 1")
    g, t = gain_stats(n, n.copy())
    check(abs(g) < 1e-9 and abs(t - 0.2) < 1e-9, "burning proportional to area: Gini 0")
    g, t = gain_stats(n, np.arange(123.0))
    check(0.2 < g < 0.5 and t > 0.2, "burning rising with hazard: positive Gini")
    ny = 10
    lo, hi, tlo, thi = gain_bootstrap(np.tile(n, (ny, 1)), np.tile(np.arange(123.0), (ny, 1)),
                                      n_boot=200)
    check(abs(lo - hi) < 1e-9, "identical years: zero-width bootstrap interval")
    check(gain_points(n, np.zeros(123)) is None, "no burned area: no curve")

    # records from both layers
    ring = [[-120, 40], [-119, 40], [-119, 41], [-120, 41], [-120, 40]]
    geom = {"type": "Polygon", "coordinates": [ring]}

    def feat(props):
        return {"type": "Feature", "properties": props, "geometry": geom}

    hist = [feat({"FIRE_YEAR_INT": 2015, "GIS_ACRES": 5000, "INCIDENT": "A"}),
            feat({"FIRE_YEAR_INT": 2015, "GIS_ACRES": 500, "INCIDENT": "B"}),
            feat({"FIRE_YEAR_INT": 1999, "GIS_ACRES": 9000, "INCIDENT": "C"}),
            feat({"FIRE_YEAR": "2018", "GIS_ACRES": 7000, "INCIDENT": "D"}),
            feat({"GIS_ACRES": 7000})]
    hist.append(feat({"FIRE_YEAR_INT": 2016, "GIS_ACRES": 8000, "FEATURE_CA": "Prescribed Fire"}))
    kept, dropped, _ = history_records(hist, 1000)
    check([r["year"] for r in kept] == [2015, 2018] and dropped["below min acres"] == 1
          and dropped["no year"] == 1 and dropped["prescribed burn"] == 1,
          "history layer filtered by year, size and prescribed burns")
    wf = [feat({"attr_FireDiscoveryDateTime": 1597017600000, "poly_GISAcres": 90000,
                "poly_IncidentName": "BIG", "attr_IncidentTypeCategory": "WF"}),
          feat({"attr_FireDiscoveryDateTime": 1597017600000, "poly_GISAcres": 90000,
                "attr_IncidentTypeCategory": "RX"}),
          feat({"poly_PolygonDateTime": "2021-10-02T00:00:00Z", "poly_GISAcres": 3000,
                "poly_FeatureCategory": "Wildfire Final Fire Perimeter"}),
          feat({"poly_PolygonDateTime": "2021-10-02T00:00:00Z", "poly_GISAcres": 3000,
                "poly_FeatureCategory": "Prescribed Fire Final Perimeter"}),
          feat({"poly_PolygonDateTime": 1597017600000, "poly_GISAcres": 3000,
                "poly_DeleteThis": "Yes"}),
          feat({"poly_PolygonDateTime": 1400000000000, "poly_GISAcres": 3000}),
          feat({"poly_PolygonDateTime": 1597017600000, "poly_GISAcres": 10})]
    kept, dropped, tc, fc = wfigs_records(wf, 1000)
    check(len(kept) == 2 and sum(dropped.values()) == 5
          and dropped["prescribed burn"] == 2 and dropped["flagged for deletion"] == 1,
          "WFIGS: prescribed burns, deleted, old and small records dropped")
    check(kept[0]["year"] == 2020 and kept[0]["month"] == 8 and kept[1]["month"] is None,
          "WFIGS year and month taken from the dates")
    study = np.ones((len(lat), len(lon)), dtype=bool)
    recs = [{"year": 2020, "name": "Big", "acres": 100.0, "polys": parse_geometry(geom)},
            {"year": 2020, "name": "BIG", "acres": 900.0, "polys": parse_geometry(geom)},
            {"year": 2020, "name": "", "acres": 50.0, "polys": parse_geometry(geom)}]
    gr = fire_groups(recs, lat, lon, study)
    check(len(gr[2020]) == 2 and max(g["acres"] for g in gr[2020]) == 900.0,
          "same-named perimeters merged for counts, largest kept")

    cands = [{"acres": 900.0, "cx": 0.0, "cy": 0.0}, {"acres": 800.0, "cx": 0.5, "cy": 0.5},
             {"acres": 700.0, "cx": 3.0, "cy": 0.0}, {"acres": 600.0, "cx": 6.0, "cy": 0.0}]
    got = [g["acres"] for g in pick_labels(cands, 3)]
    check(got == [900.0, 700.0, 600.0], "labels: a fire next to a larger named fire is skipped")

    print("=" * 60)
    if failures:
        print(f"[FAILURE] Self-test: {failures} check(s) failed")
        return 1
    print("[SUCCESS] Self-test passed")
    return 0


# =====================================================
# MAIN
# =====================================================

def load_hazard():
    """days_gt_p90 (year, lat, lon), years, lat, lon, study mask from step 05."""
    import xarray as xr

    with xr.open_dataset(HAZARD_NC) as ds:
        days = ds["days_gt_p90"].values.astype("float32")
        years = [int(y) for y in ds["year"].values]
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        study = ds["study_area"].values.astype(bool)
    return days, years, lat, lon, study


def write_burned_nc(burned, years, lat, lon, study):
    hz = _hz()
    hz.write_netcdf(OUT_NC, years, lat, lon, {}, {
        "burned_fraction": (burned, {
            "long_name": "Fraction of the cell inside wildfire perimeters",
            "units": "1"})},
        study, {"title": "NIFC wildfire perimeters on the ERA5 0.25 deg grid",
                "note": "Perimeters of the year's wildfires; overlaps counted once; "
                        "coverage varies by agency."})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="years for the overlay maps (max 4)")
    parser.add_argument("--min-acres", type=float, default=1000.0,
                        help="smallest perimeter to use (default 1000 acres)")
    parser.add_argument("--sub", type=int, default=8,
                        help="sample points per cell side when rasterising")
    parser.add_argument("--label-top", type=int, default=3,
                        help="largest fires named on each map (0 = none)")
    parser.add_argument("--history", type=Path, default=HIST_PATH)
    parser.add_argument("--wfigs", type=Path, default=WFIGS_PATH)
    parser.add_argument("--smooth", action="store_true",
                        help="also write a smooth version of the overlay map")
    args = parser.parse_args()

    print("=" * 60)
    print("FIRE OVERLAY AND VALIDATION")
    print("=" * 60)

    if args.selftest:
        return selftest()

    hz = _hz()
    import map_utils as mu

    # ---- hazard fields --------------------------------------------------
    if not HAZARD_NC.exists():
        print(f"[FAILURE] {HAZARD_NC.name} not found. Run 05_hazard_climatology.py first.")
        return 1
    days, years, lat, lon, study = load_hazard()
    lon_w = np.where(lon > 180, lon - 360, lon)
    print(f"[INFO] Hazard: {years[0]}-{years[-1]} ({len(years)} years), "
          f"grid {len(lat)} x {len(lon)}, {int(study.sum())} study cells")

    masks, outline = hz.state_masks(lat, lon, hz.STATES)
    groups = masks if masks else {"Study area": study}
    clip = mu.geom_to_path(outline) if outline is not None else None

    # ---- perimeters -----------------------------------------------------
    for path in (args.history, args.wfigs):
        if not Path(path).exists():
            print(f"[FAILURE] {path} not found. Run 06a_download_fire_perimeters.py.")
            return 1
    hist, dropped, hist_feats = history_records(load_geojson(args.history),
                                                args.min_acres)
    print(f"[INFO] History layer: kept {len(hist)} wildfire polygons of at least "
          f"{args.min_acres:,.0f} acres; dropped {dict(dropped)}")
    print(f"[INFO] History feature categories: {dict(hist_feats.most_common(6))}")
    wf, dropped, type_counts, feat_counts = wfigs_records(load_geojson(args.wfigs),
                                                          args.min_acres)
    print(f"[INFO] WFIGS: kept {len(wf)} wildfire polygons; dropped {dict(dropped)}")
    print(f"[INFO] WFIGS incident types: {dict(type_counts.most_common(6))}")
    print(f"[INFO] WFIGS feature categories: {dict(feat_counts.most_common(6))}")
    records = [r for r in hist + wf if r["year"] in years]
    dated = [r for r in wf if r["month"] is not None and r["year"] in years]
    if dated:
        off = sum(r["acres"] for r in dated if r["month"] not in hz.SEASON_MONTHS)
        tot = sum(r["acres"] for r in dated)
        print(f"[INFO] WFIGS acres discovered outside Jun-Sep: {off / tot:.0%} "
              "(not described by the JJAS hazard; typically wind-driven autumn fires)")
    print("[INFO] The history layer has no fire dates: its burned area covers all seasons")

    by_year = defaultdict(list)
    for r in records:
        by_year[r["year"]].append(r)
    missing = [y for y in years if not by_year.get(y)]
    if missing:
        print(f"[WARNING] No perimeters for {missing}: those years are left out")
    years_used = [y for y in years if by_year.get(y)]
    if len(years_used) < MIN_YEARS:
        print(f"[FAILURE] Only {len(years_used)} years with perimeters")
        return 1

    # ---- rasterise ------------------------------------------------------
    dlat = float(np.median(np.abs(np.diff(lat))))
    dlon = float(np.median(np.abs(np.diff(lon_w))))
    area_cells = np.repeat(cell_area_km2(lat, dlat, dlon)[:, None], len(lon), axis=1)
    burned = np.zeros((len(years), len(lat), len(lon)), dtype="float32")
    for k, y in enumerate(years):
        if by_year.get(y):
            burned[k] = burned_fraction([p for r in by_year[y] for p in r["polys"]],
                                        lat, lon, sub=args.sub)
    burned_km2 = burned * area_cells[None]
    print("[INFO] Perimeters rasterised")

    fgroups = fire_groups(records, lat, lon_w, study)
    total_by_year = {y: float(burned_km2[years.index(y)][study].sum()) for y in years}
    print("[INFO] year  fires(in study)  burned km2 (study states)")
    for y in years_used:
        n_in = sum(g["in_study"] for g in fgroups.get(y, []))
        src = {r["source"] for r in by_year[y]}
        print(f"       {y}  {n_in:6d}           {total_by_year[y]:9,.0f}   {'/'.join(sorted(src))}")

    # ---- state series and statistics -------------------------------------
    sel = [years.index(y) for y in years_used]
    yr = np.asarray(years_used)
    all_groups = dict(groups)
    all_groups["ALL"] = study
    series, results, curves, stats = {}, {}, {}, {}
    annual_rows, summary_rows = [], []
    n_yv, b_yv = {}, {}
    for name, mask in all_groups.items():
        haz = hz.area_mean(days[sel].astype(float), mask, lat)
        bkm = np.array([float(burned_km2[i][mask].sum()) for i in sel])
        if name != "ALL":
            series[name] = (haz, bkm)
            for y, h, b in zip(years_used, haz, bkm):
                annual_rows.append([name, y, round(float(b), 1), round(float(h), 3)])
        rho, p = spearman_perm(haz, bkm)
        rho_d, p_d = spearman_perm(detrend(haz, yr), detrend(bkm, yr), seed=1)
        results[name] = {"rho": rho, "p": p, "rho_d": rho_d, "p_d": p_d}

        nb = [hazard_bins(days[i], area_cells, burned_km2[i], mask) for i in sel]
        n_yv[name] = np.array([a for a, _ in nb])
        b_yv[name] = np.array([b for _, b in nb])
        gini, top20 = gain_stats(n_yv[name].sum(0), b_yv[name].sum(0))
        lo, hi, tlo, thi = gain_bootstrap(n_yv[name], b_yv[name])
        stats[name] = (gini, top20, lo, hi, tlo, thi)
        curves[name] = gain_points(n_yv[name].sum(0), b_yv[name].sum(0))
        summary_rows.append([name, len(yr), round(rho, 3), round(p, 4),
                             round(rho_d, 3), round(p_d, 4), round(top20, 3),
                             round(gini, 3), round(lo, 3), round(hi, 3),
                             round(float(bkm.sum()), 1)])

    # ---- tables ---------------------------------------------------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    hz.write_csv(OUT_ANNUAL, ["state", "year", "burned_km2", "days_gt_local_p90"],
                 annual_rows)
    hz.write_csv(OUT_SUMMARY, ["group", "n_years", "spearman_rho", "p_perm",
                               "spearman_rho_detrended", "p_perm_detrended",
                               "share_burned_in_top20pct_hazard", "gini",
                               "gini_ci_low", "gini_ci_high", "total_burned_km2"],
                 summary_rows)
    print(f"[INFO] Wrote {OUT_ANNUAL.name}, {OUT_SUMMARY.name}")
    try:
        write_burned_nc(burned, years, lat, lon, study)
        print(f"[SUCCESS] Wrote {OUT_NC}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] {OUT_NC.name} not written: {exc}")

    print("=" * 60)
    print(f"{'group':11s} {'rho':>6s} {'p':>7s} {'rho_det':>8s} {'p':>7s} "
          f"{'top20%':>7s} {'Gini':>6s}  95% interval")
    for name in list(series) + ["ALL"]:
        r, s = results[name], stats[name]
        print(f"{name:11s} {r['rho']:6.2f} {r['p']:7.3f} {r['rho_d']:8.2f} "
              f"{r['p_d']:7.3f} {s[1]:7.0%} {s[0]:6.2f}  {s[2]:.2f} to {s[3]:.2f}")
    print("[INFO] top20% = share of burned area in the 20% of cell-years with the "
          "highest hazard (20% if hazard said nothing)")

    # ---- figures --------------------------------------------------------
    study_90 = hz.area_mean(days.astype(float), study, lat)
    if args.years:
        picks = [y for y in args.years if y in years][:4]
        for y in args.years:
            if y not in years:
                print(f"[WARNING] Year {y} not available, skipped")
    else:
        picks = [years[k] for k in np.argsort(study_90)[::-1][:4]]
    picks = sorted(picks)
    perim = {y: rings_to_path([p for r in by_year[y] for p in r["polys"]])
             for y in picks if by_year.get(y)}
    for y in picks:
        top = sorted((g for g in fgroups.get(y, []) if g["in_study"]),
                     key=lambda g: -g["acres"])[:3]
        print(f"[INFO] {y} largest: " + ", ".join(
            f"{g['name'].title() or 'unnamed'} ({g['acres']:,.0f} ac)" for g in top))

    jobs = []
    for sm in ([False, True] if args.smooth else [False]):
        jobs.append((mu.variant_name("fire_overlay_years.png", sm),
                     lambda p, sm=sm: make_overlay_figure(
                         p, lon, lat, study, days, years, picks, perim, fgroups,
                         total_by_year, args.min_acres, args.label_top,
                         smooth=sm, clip=clip)))
    jobs.append(("fire_gain_curve.png",
                 lambda p: make_gain_figure(p, curves, stats, years_used[0],
                                            years_used[-1])))
    jobs.append(("fire_state_scatter.png",
                 lambda p: make_scatter_figure(p, years_used, series, results)))
    for fname, func in jobs:
        path = FIG_DIR / fname
        try:
            func(path)
            print(f"[SUCCESS] Figure: {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] {fname} not written: {exc}")

    print("=" * 60)
    print("[SUCCESS] Fire overlay finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
