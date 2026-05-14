"""
imd_international.py — compute IMD-4 station-level features for any
city given a list of (station_id, lat, lon) tuples.

Inputs sourced internationally :
  M (heavy-transit access)   <- OSM Overpass query for railway=station,
                               subway, tram_stop within 300 m of each station
  I (cycle-infrastructure)   <- OSM Overpass query for highway=cycleway
                               or cycleway=track|lane within 300 m
  T (topography)             <- Open-Elevation API (SRTM 30 m) at the
                               station coordinate
  D (intra-system density)   <- count of other system stations within 1 km

The script is deliberately country-agnostic : it consumes only the
station list extracted from the trip log and OpenStreetMap.

Output : a parquet under
  paper_demand/data_collection/imd_international/<city>.parquet
with columns (station_id, lat, lon,
              gtfs_heavy_stops_300m, infra_cyclable_features_300m,
              elevation_m, topography_roughness_index,
              n_stations_within_500m, n_stations_within_1km,
              catchment_density_per_km2).
"""
from __future__ import annotations

import argparse
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from overpy import Overpass

DATA_ROOT = Path("/home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/data_collection")
OUT_DIR = DATA_ROOT / "imd_international"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = DATA_ROOT / "_cache_osm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distance in meters between two (lat, lon) pairs."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def extract_stations_from_trip_zip(zip_path: Path, city_slug: str) -> pd.DataFrame:
    """Read trip CSVs from a Lyft-format zip and build the station list
    from start_station_* columns. Aggregates across the whole zip."""
    seen = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("__MACOSX") or not name.endswith(".csv"):
                continue
            with z.open(name) as f:
                # Stream in chunks; keep just station-level info
                for chunk in pd.read_csv(
                    f,
                    chunksize=200_000,
                    usecols=["start_station_id", "start_station_name",
                             "start_lat", "start_lng"],
                    low_memory=False,
                ):
                    chunk = chunk.dropna(subset=["start_station_id"])
                    grouped = chunk.groupby("start_station_id").agg(
                        name=("start_station_name", "first"),
                        lat=("start_lat", "mean"),
                        lng=("start_lng", "mean"),
                        n=("start_lat", "count"),
                    ).reset_index()
                    for _, row in grouped.iterrows():
                        sid = row["start_station_id"]
                        if sid in seen:
                            old = seen[sid]
                            new_n = old["n"] + row["n"]
                            seen[sid] = {
                                "name": old["name"] or row["name"],
                                "lat": (old["lat"]*old["n"] + row["lat"]*row["n"]) / new_n,
                                "lng": (old["lng"]*old["n"] + row["lng"]*row["n"]) / new_n,
                                "n": new_n,
                            }
                        else:
                            seen[sid] = {"name": row["name"], "lat": row["lat"],
                                         "lng": row["lng"], "n": row["n"]}
    out = (pd.DataFrame.from_dict(seen, orient="index")
           .reset_index().rename(columns={"index": "station_id"}))
    out["station_id"] = out["station_id"].astype(str)
    out["city"] = city_slug
    return out


def overpass_query_bbox(query_body: str, bbox: tuple[float, float, float, float],
                        cache_key: str, want_centers: bool = True,
                        retries: int = 3) -> list[tuple[float, float]]:
    """Run a city-wide bbox Overpass query, return list of (lat, lon) of matches."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        return [tuple(p) for p in json.loads(cache_file.read_text())]

    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"
    q = f"""
    [out:json][timeout:180];
    (
      {query_body}
    );
    out tags center;
    """
    q = q.replace("{bbox}", bbox_str)

    headers = {
        "User-Agent": "imd-international-research/0.1 (rfosse@cesi.fr)",
        "Accept": "application/json",
    }
    for attempt in range(retries):
        try:
            r = requests.post(OVERPASS_URL, data={"data": q},
                              headers=headers, timeout=200)
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            pts = []
            for el in data.get("elements", []):
                if "lat" in el and "lon" in el:
                    pts.append((el["lat"], el["lon"]))
                elif "center" in el:
                    pts.append((el["center"]["lat"], el["center"]["lon"]))
            cache_file.write_text(json.dumps(pts))
            return pts
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
            else:
                print(f"      [overpass error] {type(e).__name__}: {e}")
                return []
    return []


def count_within(target_lat: float, target_lon: float,
                 points: list[tuple[float, float]], radius_m: int) -> int:
    """Count `points` within `radius_m` of (target_lat, target_lon)."""
    if not points:
        return 0
    c = 0
    for plat, plon in points:
        if haversine_m(target_lat, target_lon, plat, plon) <= radius_m:
            c += 1
    return c


def get_elevation(lat: float, lon: float) -> float:
    """Open-Elevation: single-point query (SRTM 30 m)."""
    cache_file = CACHE_DIR / f"elev_{lat:.5f}_{lon:.5f}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())["elevation"]
    try:
        r = requests.get(OPEN_ELEVATION_URL,
                         params={"locations": f"{lat},{lon}"},
                         timeout=30)
        r.raise_for_status()
        elev = r.json()["results"][0]["elevation"]
        cache_file.write_text(json.dumps({"elevation": elev}))
        return float(elev)
    except Exception as e:
        print(f"      [elev error] {type(e).__name__}: {e}")
        return float("nan")


def get_elevation_batch(coords: list[tuple[float, float]]) -> list[float]:
    """Batch elevation lookup (Open-Elevation supports up to ~100 per req)."""
    out = []
    BATCH = 50
    for i in range(0, len(coords), BATCH):
        batch = coords[i:i+BATCH]
        # Check cache
        cached = []
        uncached = []
        for j, (lat, lon) in enumerate(batch):
            cf = CACHE_DIR / f"elev_{lat:.5f}_{lon:.5f}.json"
            if cf.exists():
                cached.append((j, json.loads(cf.read_text())["elevation"]))
            else:
                uncached.append((j, lat, lon))
        # Query the uncached
        if uncached:
            payload = {"locations": [{"latitude": lat, "longitude": lon}
                                     for _, lat, lon in uncached]}
            try:
                r = requests.post(OPEN_ELEVATION_URL, json=payload, timeout=60)
                r.raise_for_status()
                results = r.json()["results"]
                for (j, lat, lon), res in zip(uncached, results):
                    cf = CACHE_DIR / f"elev_{lat:.5f}_{lon:.5f}.json"
                    cf.write_text(json.dumps({"elevation": res["elevation"]}))
                    cached.append((j, res["elevation"]))
            except Exception as e:
                print(f"      [elev batch error] {type(e).__name__}: {e}")
                for j, _, _ in uncached:
                    cached.append((j, float("nan")))
        cached.sort()
        out.extend([v for _, v in cached])
    return out


def compute_imd_for_city(stations: pd.DataFrame, city_slug: str,
                         skip_osm: bool = False,
                         skip_elevation: bool = False) -> pd.DataFrame:
    """Compute IMD-4 station-level features for one city."""
    n = len(stations)
    print(f"\nProcessing {city_slug} : {n} stations")
    out = stations.copy()

    # Compute bbox with a small margin
    lat_min, lat_max = float(out["lat"].min()), float(out["lat"].max())
    lng_min, lng_max = float(out["lng"].min()), float(out["lng"].max())
    margin = 0.02  # ~2 km
    bbox = (lat_min - margin, lng_min - margin,
            lat_max + margin, lng_max + margin)
    print(f"  [{city_slug}] bbox = {bbox}")

    # ── M-axis : heavy-transit stops in bbox ──────────────────────────────
    if not skip_osm:
        print(f"  [{city_slug}] Overpass M-axis (bbox query for all transit)...")
        M_query = """
          node["railway"="station"]({bbox});
          node["railway"="halt"]({bbox});
          node["station"="subway"]({bbox});
          node["station"="light_rail"]({bbox});
          node["railway"="tram_stop"]({bbox});
        """
        m_points = overpass_query_bbox(M_query, bbox, f"M_bbox_{city_slug}")
        print(f"      {len(m_points)} transit nodes in bbox")
        out["gtfs_heavy_stops_300m"] = [
            count_within(r["lat"], r["lng"], m_points, 300)
            for _, r in out.iterrows()
        ]

        # ── I-axis : cycle infra in bbox ──────────────────────────────────
        print(f"  [{city_slug}] Overpass I-axis (bbox query for all cycle infra)...")
        I_query = """
          way["highway"="cycleway"]({bbox});
          way["cycleway"="track"]({bbox});
          way["cycleway"="lane"]({bbox});
          way["bicycle"="designated"]({bbox});
        """
        i_points = overpass_query_bbox(I_query, bbox, f"I_bbox_{city_slug}")
        print(f"      {len(i_points)} cycle-infra ways in bbox")
        out["infra_cyclable_features_300m"] = [
            count_within(r["lat"], r["lng"], i_points, 300)
            for _, r in out.iterrows()
        ]
    else:
        out["gtfs_heavy_stops_300m"] = np.nan
        out["infra_cyclable_features_300m"] = np.nan

    # ── T-axis : SRTM elevation via Open-Elevation ────────────────────────
    if not skip_elevation:
        print(f"  [{city_slug}] Elevation lookups (SRTM 30 m)...")
        coords = list(zip(out["lat"].astype(float), out["lng"].astype(float)))
        elevs = get_elevation_batch(coords)
        out["elevation_m"] = elevs
        # Local roughness : std-dev of elevation in a 3-station neighbourhood
        out["topography_roughness_index"] = np.nan  # filled after we know
    else:
        out["elevation_m"] = np.nan
        out["topography_roughness_index"] = np.nan

    # ── D-axis : intra-system station density ─────────────────────────────
    print(f"  [{city_slug}] Intra-system density (500 m, 1 km)...")
    lats = out["lat"].astype(float).values
    lngs = out["lng"].astype(float).values
    n500, n1k = np.zeros(n), np.zeros(n)
    rough = np.zeros(n)
    for i in range(n):
        # Pairwise haversine — O(N²) but N≤1500 typical
        for j in range(n):
            if i == j: continue
            d = haversine_m(lats[i], lngs[i], lats[j], lngs[j])
            if d < 500: n500[i] += 1
            if d < 1000: n1k[i] += 1
        # Local roughness : std-dev of elevation of nearest-5 stations
        if not skip_elevation:
            distances = [(j, haversine_m(lats[i], lngs[i], lats[j], lngs[j]))
                         for j in range(n) if j != i]
            distances.sort(key=lambda x: x[1])
            knn = [out.iloc[j]["elevation_m"] for j, _ in distances[:5]]
            knn = [v for v in knn if not np.isnan(v)]
            rough[i] = np.std(knn) if len(knn) >= 2 else 0.0
    out["n_stations_within_500m"] = n500.astype(int)
    out["n_stations_within_1km"] = n1k.astype(int)
    out["topography_roughness_index"] = rough
    out["catchment_density_per_km2"] = n1k / (math.pi * 1.0**2)  # 1 km buffer

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True,
                   help="City slug (must match Tier 1 download folder)")
    p.add_argument("--month", default="202408",
                   help="Reference month to read for station extraction (YYYYMM)")
    p.add_argument("--skip-osm", action="store_true")
    p.add_argument("--skip-elev", action="store_true")
    args = p.parse_args()

    zip_path = (DATA_ROOT / "tier1_trip_logs" / args.city
                / f"{args.month}-{args.city.split('_', 1)[1]}-tripdata.zip")
    if not zip_path.exists():
        # Fallback patterns
        for pat in [f"{args.month}-citibike-tripdata.zip",
                    f"{args.month}-capitalbikeshare-tripdata.zip",
                    f"{args.month}-divvy-tripdata.zip",
                    f"{args.month}-baywheels-tripdata.csv.zip",
                    f"{args.month}-bluebikes-tripdata.zip"]:
            cand = DATA_ROOT / "tier1_trip_logs" / args.city / pat
            if cand.exists():
                zip_path = cand; break
    if not zip_path.exists():
        print(f"Trip zip not found for {args.city} {args.month}")
        return

    print(f"Reading stations from {zip_path}")
    t0 = time.time()
    stations = extract_stations_from_trip_zip(zip_path, args.city)
    print(f"Extracted {len(stations)} unique stations in {time.time()-t0:.1f}s")
    print(stations.head(5))

    imd = compute_imd_for_city(stations, args.city,
                                skip_osm=args.skip_osm,
                                skip_elevation=args.skip_elev)
    out_path = OUT_DIR / f"{args.city}.parquet"
    imd.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path} ({len(imd)} stations × {len(imd.columns)} cols)")
    print(imd.describe(include="all").T[["count", "mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
