"""IMD National Catalogue — scoring pipeline.

This package is the v0.1 scaffold. The full implementation will port
`b20_national_imd4.py` and `b21_national_ies.py` from the
BikeShare-ICT working tree into a tested, packaged API:

    from imd_pipeline import score_communes, audit_summary

    df = score_communes()          # 34,858 rows
    summary = audit_summary(df)    # per-region/strata aggregates

See the README roadmap for the staging plan.
"""

__version__ = "0.1.0"
