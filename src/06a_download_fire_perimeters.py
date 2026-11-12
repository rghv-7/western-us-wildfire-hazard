"""Download fire perimeter polygons from an ArcGIS feature layer (REST query API).

Pages through the layer, keeps polygons that intersect the study bounding box,
simplifies the geometry lightly (display and 0.25 deg gridding do not need
metre-level edges) and writes one GeoJSON file to data/external/.

It works for any ArcGIS feature layer, so the same script serves both NIFC
layers: the history layer (up to 2019) and the WFIGS layer (2020 onward).
Run with --count-only first to check the layer and the count before downloading.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "external"

# west, south, east, north (covers CA, OR, WA, NV, ID, AZ)
DEFAULT_BBOX = "-125,31,-109,49.1"


class ServerError(Exception):
    """The server answered, but with an error message (not worth retrying)."""


def layer_url(url):
    """Normalise a pasted service or layer URL to .../FeatureServer/<n>."""
    url = url.split("?")[0].rstrip("/")
    url = re.sub(r"/query$", "", url)
    if re.search(r"/(Feature|Map)Server$", url):
        url += "/0"
    return url


def default_name(url):
    """Service name taken from the URL, used for the default output file."""
    parts = [p for p in urlparse(url).path.split("/") if p]
    for i, part in enumerate(parts):
        if part in ("FeatureServer", "MapServer") and i > 0:
            return parts[i - 1]
    return "fire_perimeters"


def query(url, params, tries=3):
    """One call to the layer's /query endpoint, returning parsed JSON."""
    full = url + "/query?" + urlencode(params)
    last = None
    for k in range(tries):
        try:
            req = Request(full, headers={"User-Agent": "wildfire-hazard-proj"})
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if "error" in data:
                raise ServerError(str(data["error"]))
            return data
        except (URLError, OSError, ValueError) as exc:
            last = exc
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"Request failed after {tries} tries: {last}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="feature service or layer URL")
    parser.add_argument("--out", help="output .geojson (default: data/external/<service>.geojson)")
    parser.add_argument("--where", default="1=1", help="attribute filter, e.g. \"GIS_ACRES>=1000\"")
    parser.add_argument("--bbox", default=DEFAULT_BBOX, help="west,south,east,north")
    parser.add_argument("--fields", default="*", help="comma-separated fields or *")
    parser.add_argument("--page", type=int, default=1000, help="records requested per call")
    parser.add_argument("--simplify", type=float, default=0.0005,
                        help="geometry tolerance in degrees (0.0005 is about 50 m, 0 disables)")
    parser.add_argument("--order-field", default="OBJECTID",
                        help="field for stable paging ('' to disable)")
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("DOWNLOAD FIRE PERIMETERS")
    print("=" * 60)

    url = layer_url(args.url)
    base = {
        "where": args.where,
        "geometry": args.bbox,
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
    }
    print(f"[INFO] Layer: {url}")
    try:
        count = int(query(url, {**base, "returnCountOnly": "true", "f": "json"})["count"])
    except (RuntimeError, ServerError, KeyError) as exc:
        print(f"[FAILURE] Count request failed: {exc}")
        return 1
    print(f"[INFO] Features matching filter and bounding box: {count}")
    if args.count_only:
        print("[SUCCESS] Count only, nothing written")
        return 0
    if count == 0:
        print("[FAILURE] Nothing to download; check --where and --bbox")
        return 1

    params = {
        **base,
        "outFields": args.fields,
        "returnGeometry": "true",
        "outSR": 4326,
        "geometryPrecision": 4,
        "f": "geojson",
        "resultRecordCount": args.page,
    }
    if args.simplify > 0:
        params["maxAllowableOffset"] = args.simplify
    order = {"orderByFields": args.order_field} if args.order_field else {}

    features = []
    offset = 0
    while offset < count:
        page_params = {**params, **order, "resultOffset": offset}
        try:
            data = query(url, page_params)
        except ServerError as exc:
            if not order:
                print(f"[FAILURE] Server error: {exc}")
                return 1
            print(f"[WARNING] Server rejected ordering by {args.order_field}; retrying without")
            order = {}
            continue
        except RuntimeError as exc:
            print(f"[FAILURE] {exc}")
            return 1
        page = data.get("features", [])
        if not page:
            break
        features.extend(page)
        offset += len(page)
        print(f"[INFO] {len(features)} / {count}")

    if len(features) < count:
        print(f"[WARNING] Got {len(features)} of {count} features; the file is incomplete")

    out = Path(args.out) if args.out else OUT_DIR / f"{default_name(url)}.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh, separators=(",", ":"))

    geom_types = Counter((f.get("geometry") or {}).get("type") for f in features)
    print(f"[INFO] Geometry types: {dict(geom_types)}")
    if features:
        print(f"[INFO] Fields: {list(features[0].get('properties', {}))}")
    print(f"[SUCCESS] Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
