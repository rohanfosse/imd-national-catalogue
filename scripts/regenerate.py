"""Rebuild the catalogue Parquet + audit-summary CSV from the bundled sources.

Usage:
    python scripts/regenerate.py [--source-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imd_pipeline import (  # noqa: E402
    CATALOGUE_DIR,
    SOURCE_DIR,
    audit_summary,
    build_catalogue,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=str(SOURCE_DIR),
                        help="Folder containing imd4_national_communes.csv"
                             " and ies_national_communes.csv")
    parser.add_argument("--out-dir", default=str(CATALOGUE_DIR),
                        help="Folder to write the Parquet + summary CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Reading sources from %s", args.source_dir)
    catalogue = build_catalogue(args.source_dir)
    log.info("Built catalogue: %d rows × %d cols", *catalogue.shape)

    parquet_path = out_dir / "communes_imd_national.parquet"
    catalogue.to_parquet(parquet_path, index=False, compression="zstd")
    log.info("Wrote %s (%.1f kB)",
             parquet_path,
             parquet_path.stat().st_size / 1024)

    summary = audit_summary(catalogue)
    summary_path = out_dir / "communes_imd_audit_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Wrote %s", summary_path)

    log.info("\nAudit summary (per population stratum):")
    log.info("%s", summary.to_string(index=False))

    log.info("\nHeadline counts:")
    log.info("  total communes ........ %d", len(catalogue))
    log.info("  with IES (income) ..... %d",
             int(catalogue["ies_available"].sum()))
    log.info("  cycling-poverty deserts %d",
             int(catalogue["double_penalty"].sum()))


if __name__ == "__main__":
    main()
