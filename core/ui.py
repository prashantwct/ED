"""Design tokens, icons and reusable Streamlit blocks.

Separate from the analytics so changing how something looks cannot
change what it says.

Rules this enforces:

* Tier is never colour alone. Every badge carries a shape, a glyph and a
  word, so the ranking survives greyscale printing and colour vision
  deficiency.
* No emoji as structural icons. Icons are inline SVG on a 24px grid.
* Data columns use tabular figures so counts line up when scanned.
* Both light and dark themes are defined; Streamlit follows the system.
* Anything drawn from the CSV is escaped before it reaches HTML.
"""

from __future__ import annotations

from html import escape
from typing import Optional

import streamlit as st

# Contrast checked against the badge background: all pairs clear 4.5:1.
TIER_STYLE = {
    "Critical": {
        "bg": "#8C1D18",
        "fg": "#FFFFFF",
        "accent": "#B3261E",
        "glyph": "▲",
        "label": "Critical",
        "meaning": "A person was killed or injured here recently.",
    },
    "High": {
        "bg": "#A44708",
        "fg": "#FFFFFF",
        "accent": "#C2570C",
        "glyph": "◆",
        "label": "High",
        "meaning": "Casualties earlier in the period, or sustained damage.",
    },
    "Watch": {
        "bg": "#F2C94C",
        "fg": "#3D2E00",
        "accent": "#C9A227",
        "glyph": "●",
        "label": "Watch",
        "meaning": "Rising, or above the landscape conflict rate.",
    },
    "Routine": {
        "bg": "#E3E9E5",
        "fg": "#33463B",
        "accent": "#6B8578",
        "glyph": "○",
        "label": "Routine",
        "meaning": "No current signal beyond background activity.",
    },
}

# Okabe-Ito colourblind-safe sequence, not a red-green ramp. The map
# pairs it with a size channel so colour is never the only cue.
CATEGORY_STYLE = {
    "Death": {"color": "#7A1E00", "label": "Human fatality"},
    "Injury": {"color": "#D55E00", "label": "Human injury"},
    "House": {"color": "#E69F00", "label": "House / store damage"},
    "Crop": {"color": "#F0E442", "label": "Crop or grain loss"},
    "Presence": {"color": "#56B4E9", "label": "Presence only"},
}

# Lucide-style: 24px grid, 1.75 stroke, currentColor for theme adaptation.
_ICON_PATHS = {
    "assessment": '<path d="M12 2v4"/><path d="m16.2 7.8 2.9-2.9"/><path d="M18 12h4"/>'
                  '<path d="m16.2 16.2 2.9 2.9"/><path d="M12 18v4"/>'
                  '<path d="m4.9 19.1 2.9-2.9"/><path d="M2 12h4"/>'
                  '<path d="m4.9 4.9 2.9 2.9"/><circle cx="12" cy="12" r="3"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/>'
              '<circle cx="12" cy="12" r="1.5"/>',
    "trend": '<path d="M22 7 13.5 15.5 8.5 10.5 2 17"/><path d="M16 7h6v6"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.5 2"/>',
    "village": '<path d="M3 21h18"/><path d="M5 21V9l7-5 7 5v12"/>'
               '<path d="M10 21v-6h4v6"/>',
    "map": '<path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3z"/><path d="M9 3v15"/>'
           '<path d="M15 6v15"/>',
    "hotspot": '<path d="M12 21s7-5.6 7-11a7 7 0 1 0-14 0c0 5.4 7 11 7 11z"/>'
               '<circle cx="12" cy="10" r="2.5"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M7 15v-4"/><path d="M12 15V7"/>'
             '<path d="M17 15v-7"/>',
    "table": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18"/>'
             '<path d="M9 10v10"/>',
    "report": '<path d="M15 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V6z"/>'
              '<path d="M14 2v5h5"/><path d="M9 13h6"/><path d="M9 17h4"/>',
}


def icon(name: str, size: int = 20, color: str = "currentColor") -> str:
    """Inline SVG icon. Inherits text colour so it adapts to the theme."""
    path = _ICON_PATHS.get(name, _ICON_PATHS["assessment"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.75" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" '
        f'focusable="false" style="flex:0 0 auto;">{path}</svg>'
    )


def inject_theme() -> None:
    """Install design tokens and component CSS. Call once, early."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


_THEME_CSS = """
<style>
:root {
  --ci-bg-raised:   #ffffff;
  --ci-bg-sunken:   #f4f6f5;
  --ci-border:      #dde5e0;
  --ci-text:        #16221c;
  --ci-text-muted:  #5b6b62;
  --ci-brand:       #1f5f3f;
  --ci-brand-soft:  #eef4f0;
  --ci-space-1: 4px;
  --ci-space-2: 8px;
  --ci-space-3: 12px;
  --ci-space-4: 16px;
  --ci-space-5: 24px;
  --ci-radius:  10px;
}

@media (prefers-color-scheme: dark) {
  :root {
    --ci-bg-raised:  #182420;
    --ci-bg-sunken:  #101815;
    --ci-border:     #2c3b34;
    --ci-text:       #e8efea;
    --ci-text-muted: #9db0a5;
    --ci-brand:      #6fbf8f;
    --ci-brand-soft: #1c2b24;
  }
}

/* Tabular figures wherever numbers form a column. */
.ci-num, [data-testid="stMetricValue"], .stDataFrame td {
  font-variant-numeric: tabular-nums;
  font-feature-settings: "tnum" 1;
}

.ci-section {
  display: flex; align-items: center; gap: var(--ci-space-3);
  margin: var(--ci-space-5) 0 var(--ci-space-2) 0; color: var(--ci-text);
}
.ci-section__icon {
  display: inline-flex; align-items: center; justify-content: center;
  width: 34px; height: 34px; border-radius: var(--ci-radius);
  background: var(--ci-brand-soft); color: var(--ci-brand);
}
.ci-section__title { font-size: 1.12rem; font-weight: 650; line-height: 1.25; }
.ci-section__sub   { font-size: 0.8rem; color: var(--ci-text-muted); margin-top: 1px; }

.ci-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 2px 9px; border-radius: 999px;
  font-size: 0.72rem; font-weight: 700;
  letter-spacing: 0.03em; text-transform: uppercase; white-space: nowrap;
}
.ci-badge__glyph { font-size: 0.72rem; line-height: 1; }

.ci-tierbar { display: flex; gap: var(--ci-space-3); flex-wrap: wrap; margin: var(--ci-space-3) 0; }
.ci-tiercard {
  flex: 1 1 130px; min-width: 130px;
  background: var(--ci-bg-raised);
  border: 1px solid var(--ci-border);
  border-left: 4px solid var(--ci-tier-accent, var(--ci-border));
  border-radius: var(--ci-radius);
  padding: var(--ci-space-3) var(--ci-space-4);
}
.ci-tiercard__label {
  display: flex; align-items: center; gap: 6px;
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--ci-text-muted);
}
.ci-tiercard__value {
  font-size: 1.7rem; font-weight: 700; line-height: 1.1;
  margin-top: 2px; color: var(--ci-text); font-variant-numeric: tabular-nums;
}
.ci-tiercard__meaning {
  font-size: 0.72rem; color: var(--ci-text-muted); margin-top: 3px; line-height: 1.35;
}

.ci-card {
  background: var(--ci-bg-raised);
  border: 1px solid var(--ci-border);
  border-left: 4px solid var(--ci-tier-accent, var(--ci-border));
  border-radius: var(--ci-radius);
  padding: var(--ci-space-4);
  margin-bottom: var(--ci-space-3);
}
.ci-card__head {
  display: flex; align-items: center; justify-content: space-between;
  gap: var(--ci-space-3); margin-bottom: var(--ci-space-2);
}
.ci-card__title { font-size: 0.98rem; font-weight: 650; color: var(--ci-text); }
.ci-card__where { font-size: 0.76rem; color: var(--ci-text-muted); margin-top: 1px; }
.ci-card__grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(84px, 1fr));
  gap: var(--ci-space-2) var(--ci-space-3); margin-top: var(--ci-space-3);
}
.ci-stat__label {
  font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--ci-text-muted);
}
.ci-stat__value {
  font-size: 1.05rem; font-weight: 650; color: var(--ci-text);
  font-variant-numeric: tabular-nums;
}
.ci-stat__value--alert { color: #C2410C; }
@media (prefers-color-scheme: dark) { .ci-stat__value--alert { color: #FF9A62; } }

.ci-findings {
  background: var(--ci-brand-soft);
  border-left: 4px solid var(--ci-brand);
  border-radius: 0 var(--ci-radius) var(--ci-radius) 0;
  padding: var(--ci-space-4) var(--ci-space-5);
  margin: var(--ci-space-3) 0;
}
.ci-findings ul { margin: 0; padding-left: 18px; }
.ci-findings li { margin: 5px 0; font-size: 0.92rem; line-height: 1.5; color: var(--ci-text); }

.ci-legend { display: flex; flex-wrap: wrap; gap: var(--ci-space-4); margin-top: var(--ci-space-2); }
.ci-legend__item {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 0.78rem; color: var(--ci-text-muted);
}
.ci-legend__dot {
  width: 11px; height: 11px; border-radius: 50%;
  border: 1px solid rgba(0,0,0,.35); flex: 0 0 auto;
}
</style>
"""


def section(name: str, title: str, subtitle: Optional[str] = None) -> None:
    """Section header with an SVG icon."""
    sub = f'<div class="ci-section__sub">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="ci-section">'
        f'  <span class="ci-section__icon">{icon(name, 19)}</span>'
        f'  <div><div class="ci-section__title">{escape(title)}</div>{sub}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def tier_badge(tier: str) -> str:
    """Badge HTML carrying colour, glyph and word."""
    style = TIER_STYLE.get(tier, TIER_STYLE["Routine"])
    return (
        f'<span class="ci-badge" style="background:{style["bg"]};color:{style["fg"]};">'
        f'<span class="ci-badge__glyph" aria-hidden="true">{style["glyph"]}</span>'
        f'{style["label"]}</span>'
    )


def tier_summary(counts: dict) -> None:
    """Four-tier count strip, each card stating what its tier means."""
    cards = []
    for tier, style in TIER_STYLE.items():
        cards.append(
            f'<div class="ci-tiercard" style="--ci-tier-accent:{style["accent"]};">'
            f'  <div class="ci-tiercard__label">'
            f'    <span aria-hidden="true">{style["glyph"]}</span>{style["label"]}'
            f"  </div>"
            f'  <div class="ci-tiercard__value">{int(counts.get(tier, 0))}</div>'
            f'  <div class="ci-tiercard__meaning">{style["meaning"]}</div>'
            f"</div>"
        )
    st.markdown(f'<div class="ci-tierbar">{"".join(cards)}</div>', unsafe_allow_html=True)


def findings(lines) -> None:
    """Assessment block at the top of the page."""
    if not lines:
        return
    items = "".join(f"<li>{escape(str(line))}</li>" for line in lines)
    st.markdown(f'<div class="ci-findings"><ul>{items}</ul></div>', unsafe_allow_html=True)


def _stat(label: str, value: str, alert: bool = False) -> str:
    cls = "ci-stat__value ci-stat__value--alert" if alert else "ci-stat__value"
    return (
        f'<div><div class="ci-stat__label">{escape(label)}</div>'
        f'<div class="{cls}">{escape(value)}</div></div>'
    )


def hotspot_card(row: dict) -> None:
    """One hotspot as a card.

    Cards rather than table rows: a hotspot's location, footprint and
    casualty count belong together, and the reader is choosing between a
    handful rather than scanning 157 beats.

    Beat and division names come from the CSV, so they are escaped.
    """
    style = TIER_STYLE.get(str(row["Tier"]), TIER_STYLE["Routine"])
    deaths = int(row["Human Deaths"])
    injured = int(row["People Injured"])
    night = row["Night Share %"]
    night_str = "-" if night != night else f"{night:.0f}%"

    stats = "".join([
        _stat("Sightings", f"{int(row['Sightings']):,}"),
        _stat("Conflicts", f"{int(row['Conflict Events']):,}"),
        _stat("Conflict share", f"{row['Conflict Share %']:.0f}%"),
        _stat("Killed", str(deaths), alert=deaths > 0),
        _stat("Injured", str(injured), alert=injured > 0),
        _stat("At night", night_str),
        _stat("Radius", f"{row['Radius (km)']:.1f} km"),
    ])

    st.markdown(
        f'<div class="ci-card" style="--ci-tier-accent:{style["accent"]};">'
        f'  <div class="ci-card__head">'
        f"    <div>"
        f'      <div class="ci-card__title">{escape(str(row["Hotspot"]))} &middot; '
        f'{escape(str(row["Divisions"]))}</div>'
        f'      <div class="ci-card__where">Beats: {escape(str(row["Beats"]))}</div>'
        f"    </div>"
        f"    {tier_badge(str(row['Tier']))}"
        f"  </div>"
        f'  <div class="ci-card__grid">{stats}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def map_legend(title: str, styles: dict) -> None:
    """Legend for a map layer keyed on tier, glyph included."""
    items = []
    for entry in styles.values():
        items.append(
            f'<span class="ci-legend__item">'
            f'<span class="ci-legend__dot" style="background:{entry["accent"]};"></span>'
            f'{entry["glyph"]} {escape(entry["label"])}</span>'
        )
    st.markdown(
        f'<div class="ci-legend"><span class="ci-legend__item">'
        f"<b>{escape(title)}</b></span>{''.join(items)}</div>",
        unsafe_allow_html=True,
    )


def category_legend(counts: dict) -> None:
    """Map legend: colour swatch plus the category name in words."""
    items = []
    for key, style in CATEGORY_STYLE.items():
        count = int(counts.get(key, 0))
        if not count:
            continue
        items.append(
            f'<span class="ci-legend__item">'
            f'<span class="ci-legend__dot" style="background:{style["color"]};"></span>'
            f'{escape(style["label"])} ({count:,})</span>'
        )
    if items:
        st.markdown(f'<div class="ci-legend">{"".join(items)}</div>', unsafe_allow_html=True)
