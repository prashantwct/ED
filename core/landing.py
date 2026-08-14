"""The screen before any data is loaded.

A forest officer opening this for the first time should be able to tell
whose tool it is, what it does with their export, and what it will not
do, before uploading anything. That is the whole job of this page.

Partner marks are real files or they are typography. An official
emblem is not something to approximate: a drawn-from-memory state seal
is a misrepresentation, so a partner without a supplied logo file gets a
clean wordmark instead. Drop the official PNG or SVG into ``assets/``
under the name in :data:`PARTNERS` and it is picked up automatically.
"""

from __future__ import annotations

import base64
import logging
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from core.ui import icon

logger = logging.getLogger(__name__)

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"

HERO_IMAGE = "hero-elephant.jpg"
HERO_ALT = (
    "A herd of Asian elephants crossing open grassland beside a reservoir "
    "in eastern Madhya Pradesh"
)
HERO_CREDIT = "Photograph © Dr Anish Andheria"

PROJECT = "MOVE-MP"
PROJECT_FULL = "Monitoring and Understanding Vital Elephant Movements in Madhya Pradesh"

# (display name, short name, logo filename). A missing file falls back to
# the wordmark rather than to an invented emblem.
PARTNERS: Tuple[Tuple[str, str, str], ...] = (
    ("Madhya Pradesh Forest Department", "MPFD", "logo-mpfd.png"),
    ("Wildlife Conservation Trust", "WCT", "logo-wct.png"),
)

CAPABILITIES: Tuple[Tuple[str, str, str], ...] = (
    ("target", "Beat priorities",
     "Every beat ranked into a decision tier, with the evidence beside the "
     "ranking: casualties, adjusted conflict rate, night share and which "
     "animals are driving it."),
    ("hotspot", "Movement hotspots",
     "Density clusters found in the point data rather than the "
     "administrative grid, so a herd working a boundary reads as one place."),
    ("village", "Villages at risk",
     "Conflict counted around each settlement, ranked by tier, with the "
     "distance to the nearest hotspot."),
    ("broadcast", "Early-warning coverage",
     "The same ranking read against the villager registry, so a village that "
     "keeps losing people with nobody enrolled to warn stops being invisible."),
    ("clock", "Timing and season",
     "The hours that hold most of the conflict, and the months, reported "
     "with how concentrated they actually are."),
    ("map", "Maps and boundaries",
     "Sightings, hotspot footprints and village risk over division, range "
     "and beat outlines from the forest department shapefiles."),
    ("report", "A brief that prints",
     "One self-contained HTML file with the maps embedded, for circulating "
     "to divisions that will not open a dashboard."),
)

REQUIRED_COLUMNS = ("Date", "Latitude", "Longitude", "Division", "Range", "Beat")
OPTIONAL_COLUMNS = (
    "Time or Hour", "Total Count", "Male / Female / Calf Count", "Crop Damage",
    "Grain Damage", "House Damage", "Injury", "Death",
    "Male / Female / Children Death Count",
)

ACCESS_NOTE = (
    "Field exports carry victim names in the free-text description and the "
    "name of each reporter. This app has no access control, so treat a link "
    "to it as you would the export itself."
)


def _data_uri(filename: str) -> Optional[str]:
    """Base64 asset, or None when the file is not shipped."""
    path = ASSET_DIR / filename
    if not path.exists():
        logger.info("Landing asset not present: %s", filename)
        return None
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else f"image/{suffix[1:]}"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _partner_marks() -> str:
    marks: List[str] = []
    for name, short, filename in PARTNERS:
        uri = _data_uri(filename)
        if uri:
            marks.append(
                f'<img class="lp-logo" src="{uri}" alt="{escape(name)}" '
                f'title="{escape(name)}"/>'
            )
        else:
            marks.append(
                f'<span class="lp-wordmark" title="{escape(name)}">'
                f'<span class="lp-wordmark__short">{escape(short)}</span>'
                f'<span class="lp-wordmark__full">{escape(name)}</span></span>'
            )
    return (
        '<div class="lp-partners"><span class="lp-partners__label">'
        'A joint initiative of</span>' + "".join(marks) + "</div>"
    )


def _capability_cards() -> str:
    cards = [
        f'<div class="lp-cap"><span class="lp-cap__icon">{icon(name, 18)}</span>'
        f'<div><div class="lp-cap__title">{escape(title)}</div>'
        f'<div class="lp-cap__body">{escape(body)}</div></div></div>'
        for name, title, body in CAPABILITIES
    ]
    return '<div class="lp-caps">' + "".join(cards) + "</div>"


def _column_chips(columns: Tuple[str, ...], kind: str) -> str:
    chips = "".join(
        f'<span class="lp-chip lp-chip--{kind}">{escape(c)}</span>' for c in columns
    )
    return f'<div class="lp-chips">{chips}</div>'


def hero() -> None:
    """Masthead. Drawn above the uploader, which is the next action."""
    image = _data_uri(HERO_IMAGE)
    st.markdown(_CSS, unsafe_allow_html=True)

    if image:
        # role="img" plus a label: the photograph carries meaning here, and
        # a CSS background is invisible to a screen reader otherwise.
        backdrop = (
            f'<div class="lp-hero__image" role="img" '
            f'aria-label="{escape(HERO_ALT)}" '
            f'style="background-image:url({image})"></div>'
            f'<div class="lp-hero__credit">{escape(HERO_CREDIT)}</div>'
        )
    else:
        backdrop = '<div class="lp-hero__image lp-hero__image--plain"></div>'

    st.markdown(
        f"""
        <section class="lp-hero">
          {backdrop}
          <div class="lp-hero__scrim"></div>
          <div class="lp-hero__body">
            <div class="lp-eyebrow">{escape(PROJECT)} &middot; Elephant Monitoring Cell</div>
            <h1 class="lp-title">Elephant Conflict Intelligence</h1>
            <p class="lp-sub">
              Turning Gajrakshak field reports into the decisions a division
              actually makes: which beats to resource, which shift to staff,
              which villages to warn.
            </p>
          </div>
        </section>
        {_partner_marks()}
        """,
        unsafe_allow_html=True,
    )


def details() -> None:
    """Everything below the uploader on the pre-upload screen."""
    st.markdown(
        f'<p class="lp-lede">{escape(PROJECT_FULL)}. Nothing is stored: the '
        f'file stays in this session and leaves when you close the tab.</p>',
        unsafe_allow_html=True,
    )

    st.markdown(_capability_cards(), unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<div class="lp-h2">What to upload</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="lp-label">Required</div>'
            + _column_chips(REQUIRED_COLUMNS, "req"),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="lp-label">Optional, and each one unlocks something</div>'
            + _column_chips(OPTIONAL_COLUMNS, "opt"),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="lp-h2">Before you share it</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="lp-note">{escape(ACCESS_NOTE)}</div>',
                    unsafe_allow_html=True)


def header() -> None:
    """Compact identity bar, once an export is loaded.

    The hero is gone by this point and the screen belongs to the data, so
    the branding shrinks to a line: whose tool, which project, and the
    partner marks small enough to ignore.
    """
    st.markdown(_CSS, unsafe_allow_html=True)
    marks: List[str] = []
    for name, short, filename in PARTNERS:
        uri = _data_uri(filename)
        marks.append(
            f'<img class="lp-bar__logo" src="{uri}" alt="{escape(name)}"/>'
            if uri else
            f'<span class="lp-bar__mark" title="{escape(name)}">{escape(short)}</span>'
        )
    st.markdown(
        f'<div class="lp-bar">'
        f'  <div class="lp-bar__id">'
        f'    <span class="lp-bar__title">Elephant Conflict Intelligence</span>'
        f'    <span class="lp-bar__project">{escape(PROJECT)} &middot; '
        f'Elephant Monitoring Cell</span>'
        f'  </div>'
        f'  <div class="lp-bar__partners">{"".join(marks)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


_CSS = """
<style>
.lp-bar {
  display: flex; align-items: center; justify-content: space-between;
  gap: var(--ci-space-4); flex-wrap: wrap;
  padding: 0 0 var(--ci-space-3) 0; margin-bottom: var(--ci-space-3);
  border-bottom: 1px solid var(--ci-border);
}
.lp-bar__title {
  display: block; font-size: 1.3rem; font-weight: 700;
  color: var(--ci-text); letter-spacing: -0.01em; line-height: 1.2;
}
.lp-bar__project {
  display: block; font-size: .72rem; font-weight: 600; letter-spacing: .1em;
  text-transform: uppercase; color: var(--ci-text-muted); margin-top: 3px;
}
.lp-bar__partners { display: flex; align-items: center; gap: 16px; }
.lp-bar__logo { height: 30px; width: auto; display: block; }
.lp-bar__mark {
  font-size: .82rem; font-weight: 700; letter-spacing: .05em;
  color: var(--ci-text-muted);
  padding: 4px 9px; border: 1px solid var(--ci-border); border-radius: 6px;
}

.lp-hero {
  position: relative; border-radius: 14px 14px 0 0; overflow: hidden;
  min-height: 340px; display: flex; align-items: flex-end;
  margin: 0; background: #0d1a14;
}
.lp-hero__image {
  position: absolute; inset: 0;
  background-size: cover; background-position: 62% 42%;
}
.lp-hero__image--plain { background: linear-gradient(120deg, #17301f, #24452e); }
/* Scrim carries the text contrast. Without it the title sits on sunlit
   grass at roughly 2:1 and fails at any size. */
.lp-hero__scrim {
  position: absolute; inset: 0;
  background: linear-gradient(100deg,
    rgba(8,20,14,.94) 0%, rgba(8,20,14,.88) 34%,
    rgba(8,20,14,.55) 62%, rgba(8,20,14,.20) 100%);
}
.lp-hero__credit {
  position: absolute; right: 10px; bottom: 8px; z-index: 2;
  font-size: .66rem; color: rgba(255,255,255,.62); letter-spacing: .01em;
}
.lp-hero__body {
  position: relative; z-index: 1;
  padding: clamp(22px, 4vw, 40px); max-width: 660px;
}
.lp-eyebrow {
  font-size: .72rem; font-weight: 700; letter-spacing: .14em;
  text-transform: uppercase; color: #9fd6b4; margin-bottom: 10px;
}
.lp-hero .lp-title, .lp-hero h1.lp-title {
  margin: 0; padding: 0; color: #ffffff; font-weight: 700; line-height: 1.06;
  font-size: clamp(1.9rem, 4.4vw, 3rem); letter-spacing: -0.02em;
}
.lp-sub {
  margin: 12px 0 0 0; color: rgba(255,255,255,.90);
  font-size: clamp(.94rem, 1.5vw, 1.06rem); line-height: 1.55; max-width: 52ch;
}

.lp-partners {
  display: flex; align-items: center; gap: 28px; flex-wrap: wrap;
  background: #ffffff; border: 1px solid var(--ci-border); border-top: none;
  border-radius: 0 0 14px 14px; padding: 14px 22px;
  margin: -14px 0 var(--ci-space-4) 0;
}
.lp-partners__label {
  font-size: .68rem; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: #7a8a80; margin-right: 4px;
}
.lp-logo { height: 38px; width: auto; display: block; }
.lp-wordmark { display: flex; flex-direction: column; line-height: 1.2; }
.lp-wordmark__short {
  font-size: 1rem; font-weight: 700; color: #16221c; letter-spacing: .04em;
}
.lp-wordmark__full { font-size: .68rem; color: #5b6b62; letter-spacing: .02em; }

.lp-lede {
  color: var(--ci-text-muted); font-size: .94rem; line-height: 1.6;
  max-width: 74ch; margin: 0 0 var(--ci-space-4) 0;
}
.lp-h2 {
  font-size: 1rem; font-weight: 650; color: var(--ci-text);
  margin: var(--ci-space-4) 0 var(--ci-space-2) 0;
}
.lp-label {
  font-size: .72rem; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; color: var(--ci-text-muted);
  margin: var(--ci-space-3) 0 6px 0;
}

.lp-caps {
  display: grid; gap: var(--ci-space-3);
  grid-template-columns: repeat(auto-fit, minmax(268px, 1fr));
  margin-bottom: var(--ci-space-4);
}
.lp-cap {
  display: flex; gap: 12px; align-items: flex-start;
  background: var(--ci-bg-raised); border: 1px solid var(--ci-border);
  border-radius: var(--ci-radius); padding: 14px 16px;
}
.lp-cap__icon {
  display: inline-flex; align-items: center; justify-content: center;
  width: 32px; height: 32px; flex: 0 0 32px; border-radius: 8px;
  background: var(--ci-brand-soft); color: var(--ci-brand);
}
.lp-cap__title {
  font-size: .9rem; font-weight: 650; color: var(--ci-text); margin-bottom: 2px;
}
.lp-cap__body { font-size: .8rem; color: var(--ci-text-muted); line-height: 1.5; }

.lp-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.lp-chip {
  font-size: .74rem; padding: 3px 9px; border-radius: 999px;
  border: 1px solid var(--ci-border); white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
.lp-chip--req {
  background: var(--ci-brand-soft); color: var(--ci-brand);
  border-color: transparent; font-weight: 650;
}
.lp-chip--opt { background: var(--ci-bg-sunken); color: var(--ci-text-muted); }

.lp-note {
  background: #fbfaf4; border: 1px solid #ece7d5; border-left: 3px solid #c9a227;
  border-radius: 8px; padding: 12px 14px;
  font-size: .82rem; line-height: 1.55; color: #5a5545;
}
</style>
"""
