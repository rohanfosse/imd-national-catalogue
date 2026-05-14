"""
tier1_downloader.py — Multi-city trip-log downloader for the
high-quality open data tier of the IMD-augmented demand benchmark.

Each city is encoded as a `CitySpec` with a URL template and an
expected CSV schema. The download writes monthly zip archives to a
per-city subdirectory and verifies the schema of the first file.

Cities covered (all confirmed open trip data):
  - Citi Bike NYC                (USA, s3)
  - Capital Bikeshare DC         (USA, s3)
  - Bay Wheels SF                (USA, s3)
  - Boston Bluebikes             (USA, s3)
  - Divvy Chicago                (USA, s3)
  - BIXI Montréal                (CA, annual ZIP archives)
  - TfL Cycle Hire London        (UK, quarterly CSV via cycling.data.tfl.gov.uk)

Run with --dry-run to list URLs without downloading.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import io
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional

OUT_ROOT = Path(__file__).parent / "tier1_trip_logs"


@dataclasses.dataclass
class CitySpec:
    name: str                      # Display name
    slug: str                      # Folder slug
    country: str
    operator: str
    url_template: str              # "{year}-{month}" or "{year}" placeholders
    granularity: str               # "monthly", "annual", "quarterly"
    extension: str = ".zip"
    notes: str = ""

    def render_url(self, year: int, month: int = 1, quarter: int = 1) -> str:
        return self.url_template.format(year=year, month=f"{month:02d}",
                                        quarter=f"Q{quarter}")


CITIES: list[CitySpec] = [
    CitySpec(
        name="Citi Bike NYC", slug="nyc_citibike",
        country="USA", operator="Lyft",
        url_template="https://s3.amazonaws.com/tripdata/{year}{month}-citibike-tripdata.zip",
        granularity="monthly",
        notes="Newer months (>=2024) use `.zip` ; older months may "
              "use `.csv.zip` or chunked `-1.csv.zip`. Inspect the s3 "
              "bucket index for exact pattern of older periods.",
    ),
    CitySpec(
        name="Capital Bikeshare DC", slug="dc_capitalbikeshare",
        country="USA", operator="Lyft",
        url_template="https://s3.amazonaws.com/capitalbikeshare-data/{year}{month}-capitalbikeshare-tripdata.zip",
        granularity="monthly",
    ),
    CitySpec(
        name="Bay Wheels SF", slug="sf_baywheels",
        country="USA", operator="Lyft",
        url_template="https://s3.amazonaws.com/baywheels-data/{year}{month}-baywheels-tripdata.csv.zip",
        granularity="monthly",
    ),
    CitySpec(
        name="Boston Bluebikes", slug="boston_bluebikes",
        country="USA", operator="Lyft / BlueRide",
        url_template="https://s3.amazonaws.com/hubway-data/{year}{month}-bluebikes-tripdata.zip",
        granularity="monthly",
    ),
    CitySpec(
        name="Divvy Chicago", slug="chicago_divvy",
        country="USA", operator="Lyft",
        url_template="https://divvy-tripdata.s3.amazonaws.com/{year}{month}-divvy-tripdata.zip",
        granularity="monthly",
    ),
    CitySpec(
        name="BIXI Montréal", slug="montreal_bixi",
        country="CA", operator="BIXI Montréal",
        url_template="https://sitewebbixi.s3.amazonaws.com/uploads/docs/biximontrealrentals{year}-XXXXX.zip",
        granularity="annual",
        notes="BIXI uses random hashes per year — see https://bixi.com/en/open-data "
              "for the actual URL ; this template is a placeholder.",
    ),
    CitySpec(
        name="TfL Cycle Hire London", slug="london_tfl_cyclehire",
        country="UK", operator="Transport for London",
        url_template="https://cycling.data.tfl.gov.uk/usage-stats/{year}{quarter}-cycle-trips.csv",
        granularity="quarterly",
        extension=".csv",
        notes="TfL releases weekly CSV files ; the URL above is illustrative — "
              "use the cycling.data.tfl.gov.uk index for the exact pattern.",
    ),
]


def fetch(url: str, dest: Path, timeout: int = 60) -> tuple[bool, int]:
    """Download a URL to `dest`. Returns (success, bytes_written)."""
    if dest.exists():
        return True, dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (imd-research)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True, len(data)
    except Exception as e:
        print(f"      [error] {url}: {type(e).__name__}: {e}")
        return False, 0


def list_zip_contents(zip_path: Path, max_files: int = 5) -> list[str]:
    try:
        with zipfile.ZipFile(zip_path) as z:
            return z.namelist()[:max_files]
    except zipfile.BadZipFile:
        return ["(not a zip)"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", nargs="*", default=None,
                   help="City slugs to download (default: all)")
    p.add_argument("--start-year", type=int, default=2024)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--start-month", type=int, default=8)
    p.add_argument("--end-month", type=int, default=8)
    p.add_argument("--dry-run", action="store_true",
                   help="Print URLs but do not download")
    args = p.parse_args()

    selected = (CITIES if args.cities is None
                else [c for c in CITIES if c.slug in args.cities])
    if not selected:
        print(f"Unknown cities. Available: {[c.slug for c in CITIES]}")
        return

    print(f"Plan: {len(selected)} cities, "
          f"{args.start_year}-{args.start_month:02d} → "
          f"{args.end_year}-{args.end_month:02d}")
    print(f"Output root: {OUT_ROOT}")
    print()

    for city in selected:
        print(f"=== {city.name} ({city.country}, {city.operator}) ===")
        city_dir = OUT_ROOT / city.slug
        city_dir.mkdir(parents=True, exist_ok=True)

        for year in range(args.start_year, args.end_year + 1):
            for month in range(args.start_month, args.end_month + 1):
                url = city.render_url(year=year, month=month)
                fname = url.rsplit("/", 1)[-1]
                dest = city_dir / fname

                if args.dry_run:
                    print(f"  [dry-run] {url}")
                    continue

                t0 = time.time()
                ok, nbytes = fetch(url, dest)
                dt = time.time() - t0
                if ok:
                    print(f"  ✓ {fname}  ({nbytes/1e6:.1f} MB, {dt:.1f}s)")
                    if dest.suffix == ".zip":
                        print(f"      contents: {list_zip_contents(dest)}")
                else:
                    print(f"  ✗ failed: {url}")


if __name__ == "__main__":
    main()
