"""Custom exceptions used across the dashboard's core modules.

Keeping these in one place lets app.py catch a small, predictable set of
error types and turn them into friendly Streamlit messages, instead of
guessing what a bare ``Exception`` from deep inside pandas actually means.
"""

from __future__ import annotations


class DataValidationError(Exception):
    """Raised when an uploaded CSV fails required validation checks.

    The message is written to be shown directly to a forest-department
    end user, so it should stay short and avoid stack-trace language.
    """


class SpatialEnrichmentError(Exception):
    """Raised when village-centroid enrichment cannot proceed safely.

    This is intentionally *not* raised for "no centroids provided" -
    that is a normal, supported mode of running the app. It is raised
    only when a centroids source was supplied but is unusable (e.g.
    missing required columns), so the caller can warn the user instead
    of silently skipping enrichment.
    """
