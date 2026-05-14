"""Smoke tests for the v0.1 scaffold.

These tests assert the public import surface only. Real scoring tests
land in v0.2 alongside the b20/b21 port.
"""

import imd_pipeline


def test_version_pinned():
    assert imd_pipeline.__version__ == "0.1.0"


def test_public_api_exposes_placeholders():
    from imd_pipeline.core import score_communes, audit_summary

    assert callable(score_communes)
    assert callable(audit_summary)
