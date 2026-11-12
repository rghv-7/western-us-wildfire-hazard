#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_utils.py

Shared plotting helpers for the wildfire hazard project (imported by the
benchmark and, later, the hazard-map scripts). Not run on its own.

  discrete_cmap   colormap + norm with distinct colour bins (no smooth
                  gradient), so small differences between classes stay visible
  map_axes        map axes with land/ocean, coastline, country borders and
                  state/province lines (Natural Earth 1:50m via cartopy)
  draw_field      draws a lat/lon grid as coloured cells (the data's true
                  0.25 deg resolution), or optionally as smooth filled
                  contours for presentation only (smooth=True); smooth maps
                  can be clipped to a polygon (clip=geom_to_path(...))
  variant_name    "x.png" -> "x_smooth.png" for the smooth versions
  add_colorbar    discrete colorbar with a tick at every bin edge
  nice_symmetric_levels  bins symmetric around zero with a round width
  stipple         dots on cells (e.g. statistically significant trends)
  footer          small source note at the bottom of a figure
  save_figure     savefig that writes a numbered copy if the target PNG is
                  locked (e.g. open in a Windows viewer while running in WSL)

Boundaries come from Natural Earth (public domain). cartopy downloads the
needed files on first use and caches them. In this map window (US West plus
edges of Canada and Mexico) the national borders shown are not disputed.

If cartopy is not installed, plots fall back to plain axes without
boundaries and a [WARNING] is printed once.
"""

import warnings
from pathlib import Path

import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import DownloadWarning
    # Natural Earth files are downloaded once on first use, then cached;
    # the warning is expected and harmless.
    warnings.filterwarnings("ignore", category=DownloadWarning)
    HAVE_CARTOPY = True
except ImportError:  # pragma: no cover
    HAVE_CARTOPY = False

_WARNED = False

OCEAN_COLOR = "#dbe7f1"

SMOOTH_NOTE = (" Smoothed for display: maps use contours interpolated "
               "between 0.25 deg grid cells.")


# =====================================================
# COLOUR BINS
# =====================================================

def discrete_cmap(name, levels, extend="neither"):
    """
    Returns (cmap, norm) with one colour per bin between consecutive levels.
    extend: 'neither', 'min', 'max' or 'both' adds a colour for values
    below the first / above the last level.
    """
    import matplotlib
    from matplotlib.colors import BoundaryNorm

    levels = list(levels)
    extra = {"neither": 0, "min": 1, "max": 1, "both": 2}[extend]
    cmap = matplotlib.colormaps[name].resampled(len(levels) - 1 + extra)
    norm = BoundaryNorm(levels, ncolors=cmap.N, extend=extend)
    return cmap, norm


def add_colorbar(fig, mappable, ax, label, levels, fmt="%g", **kwargs):
    """Colorbar with equal-width bins and a tick at every bin edge."""
    cb = fig.colorbar(
        mappable, ax=ax, ticks=list(levels), spacing="uniform",
        format=fmt, shrink=0.85, pad=0.03, **kwargs,
    )
    cb.set_label(label)
    return cb


def nice_symmetric_levels(values, n_side=5):
    """
    Bin edges symmetric around zero with n_side bins each side and a
    round bin width (0.5, 1, 2, 2.5, 5, 10, ...), covering the 98th
    percentile of |values|.
    """
    top = float(np.nanpercentile(np.abs(values), 98))
    raw = max(top / n_side, 1e-9)
    mag = 10.0 ** np.floor(np.log10(raw))
    width = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return np.arange(-n_side, n_side + 1) * width


def variant_name(name, smooth):
    """'maps.png' -> 'maps_smooth.png' when smooth is True."""
    return name.replace(".png", "_smooth.png") if smooth else name


def geom_to_path(geom):
    """
    matplotlib Path (lon/lat) from a shapely Polygon or MultiPolygon, used to
    clip smooth maps to the study area. Only outer boundaries are used.
    """
    from matplotlib.path import Path as MplPath

    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    verts, codes = [], []
    for poly in polys:
        xy = np.asarray(poly.exterior.coords)
        c = np.full(len(xy), MplPath.LINETO, dtype=MplPath.code_type)
        c[0] = MplPath.MOVETO
        c[-1] = MplPath.CLOSEPOLY
        verts.append(xy)
        codes.append(c)
    return MplPath(np.concatenate(verts), np.concatenate(codes))


def footer(fig, note, extra=""):
    """Source note at the bottom left; credits Natural Earth when used."""
    if HAVE_CARTOPY:
        note += " Boundaries: Natural Earth."
    fig.text(0.01, 0.005, note + extra, fontsize=7, color="0.4")


def save_figure(fig, path, **kwargs):
    """
    fig.savefig(path). If the file cannot be overwritten (on /mnt drives in
    WSL this happens while a Windows program has the PNG open and shows up
    as 'Errno 22 Invalid argument'), writes <name>_1.png, _2.png, ... beside
    it instead and says so.
    """
    path = Path(path)
    try:
        fig.savefig(path, **kwargs)
        return path
    except OSError as exc:
        for k in range(1, 100):
            alt = path.with_name(f"{path.stem}_{k}{path.suffix}")
            if alt.exists():
                continue
            try:
                fig.savefig(alt, **kwargs)
            except OSError:
                continue
            print(f"[WARNING] Could not overwrite {path.name} ({exc}); close it "
                  f"in any viewer. Saved {alt.name} instead")
            return alt
        raise


# =====================================================
# GRID HELPERS
# =====================================================

def cell_edges(centers):
    """Cell edges from 1-D cell centres (ascending or descending)."""
    c = np.asarray(centers, dtype=float)
    mid = (c[:-1] + c[1:]) / 2.0
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def grid_extent(lat, lon):
    """[west, east, south, north] covering all grid cells."""
    ex, ey = cell_edges(lon), cell_edges(lat)
    return [float(ex.min()), float(ex.max()), float(ey.min()), float(ey.max())]


# =====================================================
# MAP AXES
# =====================================================

def _decorate(ax):
    import matplotlib.ticker as mticker

    ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                   facecolor=OCEAN_COLOR, edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"),
                   facecolor=OCEAN_COLOR, edgecolor="none", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="0.15", linewidth=0.6, zorder=3)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural", "admin_1_states_provinces_lakes", "50m",
            edgecolor="0.3", facecolor="none", linewidth=0.5),
        zorder=3)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural", "admin_0_boundary_lines_land", "50m",
            edgecolor="black", facecolor="none", linewidth=0.9),
        zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(5)
    gl.ylocator = mticker.MultipleLocator(5)


def map_axes(fig, nrows, ncols, index, extent):
    """Map axes at the given subplot position. extent = [W, E, S, N]."""
    global _WARNED
    if HAVE_CARTOPY:
        ax = fig.add_subplot(nrows, ncols, index, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        _decorate(ax)
        return ax

    if not _WARNED:
        print("[WARNING] cartopy not installed: maps drawn without state or "
              "land boundaries (pip install cartopy)")
        _WARNED = True
    ax = fig.add_subplot(nrows, ncols, index)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_facecolor(OCEAN_COLOR)
    ax.set_aspect(1.0 / np.cos(np.deg2rad((extent[2] + extent[3]) / 2.0)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return ax


def fill_coast(data, n_iter=2):
    """
    Fills NaN cells next to valid cells with the mean of their valid
    neighbours (n_iter cells deep). Used only for smooth display, so the
    coloured field reaches the true coastline before the ocean is drawn
    over it. Never use the filled array for statistics.
    """
    out = np.array(data, dtype=float)
    for _ in range(n_iter):
        pad = np.pad(out, 1, constant_values=np.nan)
        stack = np.stack([
            pad[1 + dy: pad.shape[0] - 1 + dy, 1 + dx: pad.shape[1] - 1 + dx]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            nb = np.nanmean(stack, axis=0)
        out = np.where(np.isnan(out), nb, out)
    return out


def draw_field(ax, lon, lat, data, cmap, norm, smooth=False, clip=None):
    """
    Colours the grid by data[lat, lon]; NaN cells stay transparent.
    smooth=False: one flat-coloured rectangle per grid cell (true resolution).
    smooth=True:  filled contours with the same colour bins; contour lines
                  are interpolated between cell centres (display only).
    clip:         optional matplotlib Path (lon/lat, see geom_to_path) that
                  cuts a smooth map exactly along its outline, e.g. the study
                  states. Without it, the ocean is drawn over the field so
                  the colours stop at the real coastline.
    """
    transform = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}

    if not smooth:
        x, y = cell_edges(lon), cell_edges(lat)
        return ax.pcolormesh(x, y, np.ma.masked_invalid(data), cmap=cmap,
                             norm=norm, shading="flat", zorder=1, **transform)

    field = fill_coast(data) if HAVE_CARTOPY else np.asarray(data, float)
    mesh = ax.contourf(lon, lat, np.ma.masked_invalid(field),
                       levels=list(norm.boundaries), cmap=cmap, norm=norm,
                       extend=norm.extend, zorder=1, **transform)
    if clip is not None:
        from matplotlib.patches import PathPatch

        patch = PathPatch(clip, transform=ax.transData,
                          facecolor="none", edgecolor="none")
        try:
            mesh.set_clip_path(patch)
        except AttributeError:  # matplotlib < 3.8
            for coll in mesh.collections:
                coll.set_clip_path(patch)
    elif HAVE_CARTOPY:
        # hide the coastal cells filled above by drawing the ocean on top
        ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                       facecolor=OCEAN_COLOR, edgecolor="none", zorder=1.5)
    return mesh


def stipple(ax, lon, lat, mask, size=1.5, color="0.15"):
    """Small dots at the centres of cells where mask[lat, lon] is True."""
    iy, ix = np.nonzero(mask)
    if iy.size == 0:
        return
    transform = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
    ax.scatter(np.asarray(lon)[ix], np.asarray(lat)[iy], s=size, c=color,
               marker=".", linewidths=0, zorder=5, **transform)
