"""Download any chart or map as SVG, high-resolution PNG, or PDF.

Every figure on the page has an SVG twin -- the brief's and the damage
study's chart builders -- and that is what gets exported, not a capture
of the interactive Plotly or pydeck canvas, which the server cannot
render without a browser. From the one SVG:

* **SVG** is the vector original, for editing or dropping into a report.
* **PDF** is the same drawing as vector operators (svglib + ReportLab).
* **PNG** is rasterised at ``PNG_SCALE`` times its drawn size by resvg,
  with the fonts shipped in ``assets/fonts`` so text renders on a host
  that has none installed.

Files are built only when their button is pressed (Streamlit runs the
callable on click), so a page with thirty figures does not draw ninety
files on every rerun.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_FAMILY = "DejaVu Sans"
# 4x the drawn size: a 900-wide chart becomes 3,600 px, about 300 dpi
# across a 12-inch print.
PNG_SCALE = 4


def standalone(svg: str) -> str:
    """The bare <svg> element, with a pixel width and height.

    Map SVGs arrive wrapped in a <div> and sized width="100%" for the
    brief's layout; a downloaded file needs its own size, taken from the
    viewBox.
    """
    start, end = svg.find("<svg"), svg.rfind("</svg>")
    if start < 0 or end < 0:
        raise ValueError("not an SVG")
    svg = svg[start:end + len("</svg>")]
    head_end = svg.find(">")
    head = svg[:head_end]
    view = re.search(r'viewBox="\s*[-\d.]+\s+[-\d.]+\s+([\d.]+)\s+([\d.]+)\s*"', head)
    if view:
        head = re.sub(r'\s(width|height)="[^"]*"', "", head)
        head += f' width="{view.group(1)}" height="{view.group(2)}"'
    if "xmlns=" not in head:
        head += ' xmlns="http://www.w3.org/2000/svg"'
    return head + svg[head_end:]


def to_png(svg: str, scale: int = PNG_SCALE) -> bytes:
    import resvg_py

    fonts = [str(p) for p in sorted(FONT_DIR.glob("*.ttf"))]
    return bytes(resvg_py.svg_to_bytes(
        svg_string=standalone(svg), zoom=scale, background="#ffffff",
        # Only the shipped fonts, and every text element set in them: an
        # SVG naming just "Helvetica" otherwise draws no text at all on a
        # host without it, and the same file renders the same everywhere.
        skip_system_fonts=bool(fonts), font_files=fonts or None,
        font_family=FONT_FAMILY, sans_serif_family=FONT_FAMILY,
        style_sheet=f'* {{ font-family: "{FONT_FAMILY}"; }}' if fonts else None,
    ))


def to_pdf(svg: str) -> bytes:
    from reportlab.graphics import renderPDF
    from svglib.svglib import svg2rlg

    drawing = svg2rlg(io.BytesIO(standalone(svg).encode("utf-8")))
    if drawing is None:
        raise ValueError("the SVG could not be converted")
    out = io.BytesIO()
    renderPDF.drawToFile(drawing, out)
    return out.getvalue()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "figure"


def download_menu(name: str, make_svg: Callable[[], Optional[str]]) -> None:
    """A small Download menu under a figure: SVG, PNG and PDF.

    ``make_svg`` is called only when a button is pressed, once per press.
    A figure that cannot be drawn downloads a one-line explanation
    rather than an empty or broken file.
    """
    import streamlit as st

    slug = _slug(name)
    cache: dict = {}

    def svg() -> str:
        if "svg" not in cache:
            cache["svg"] = standalone(make_svg() or "")
        return cache["svg"]

    def guarded(build: Callable[[], bytes]) -> Callable[[], bytes]:
        def run() -> bytes:
            try:
                return build()
            except Exception as exc:  # noqa: BLE001 - a download must not crash the page
                logger.exception("Export of %s failed", name)
                return f"{name} could not be exported: {type(exc).__name__}: {exc}\n".encode()
        return run

    with st.popover("Download", icon=":material/download:"):
        st.caption(name)
        st.download_button("SVG (vector)", guarded(lambda: svg().encode("utf-8")),
                           f"{slug}.svg", mime="image/svg+xml", on_click="ignore",
                           key=f"dl_svg_{slug}", width="stretch")
        st.download_button(f"PNG ({PNG_SCALE}x resolution)", guarded(lambda: to_png(svg())),
                           f"{slug}.png", mime="image/png", on_click="ignore",
                           key=f"dl_png_{slug}", width="stretch")
        st.download_button("PDF (vector)", guarded(lambda: to_pdf(svg())),
                           f"{slug}.pdf", mime="application/pdf", on_click="ignore",
                           key=f"dl_pdf_{slug}", width="stretch")
