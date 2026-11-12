# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:47:05 2026

@author: ragha
"""

#!/usr/bin/env python3

from pathlib import Path
import zipfile
import xarray as xr

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path.cwd().parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"

ZIPFILE = RAW_DIR / "era5_test_201006.nc"

EXTRACT_DIR = RAW_DIR / "extracted"

EXTRACT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# EXTRACT
# --------------------------------------------------

print(f"Input file: {ZIPFILE}")

with zipfile.ZipFile(ZIPFILE, "r") as z:
    z.extractall(EXTRACT_DIR)

print("\nExtracted files:")

for f in EXTRACT_DIR.glob("*.nc"):
    print(f.name)

# --------------------------------------------------
# INSPECT
# --------------------------------------------------

for f in EXTRACT_DIR.glob("*.nc"):

    print("\n" + "=" * 70)
    print(f"Opening: {f.name}")

    ds = xr.open_dataset(
        f,
        engine="netcdf4"
    )

    print(ds)

    print("\nVariables:")
    print(list(ds.data_vars))