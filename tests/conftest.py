"""Pytest configuration for test collection."""

import os

# Skip GCS-dependent test files entirely when SKIP_GCS_TESTS is set
# This prevents import errors from modules that require GCS credentials
collect_ignore = []
if "SKIP_GCS_TESTS" in os.environ:
    collect_ignore.extend(
        [
            "test_gcs.py",
            "test_tracking_pipeline.py",
        ]
    )
