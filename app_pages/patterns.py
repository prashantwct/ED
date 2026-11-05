"""Patterns: is it getting worse, when does it happen, what is new or growing.

Three questions that were three tabs, now sections of one page, each
leading with its main chart. All three read the analysis history -- the
register up to the end of the period -- because a trend needs the months
before it. The 3D space-time cube is on the Analyst page.
"""

from __future__ import annotations

import streamlit as st

from app_pages import emerging, trend, when
from core.ui import section


def render(c) -> None:
    first = c.history["Date"].min()
    span = f"{first.day} {first:%b %Y} to {c.window.end.day} {c.window.end:%b %Y}"
    section("trend", "Is it getting worse?", f"Every month from {span}")
    trend.render(c)
    section("clock", "When?", "The hours, nights and months conflict concentrates in")
    when.render(c)
    section("hotspot", "New or growing?",
            "Places tested step by step for whether they are becoming hotspots")
    emerging.render(c)
    st.caption("The space-time cube, in 3D, is on the Analyst page.")
