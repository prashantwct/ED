"""The space-time cube in 3D."""

from __future__ import annotations

import logging

import streamlit as st

from app_pages.common import CUBE_CONFIG
from core import charts, spacetime


logger = logging.getLogger(__name__)


def render(c) -> None:
    cube = c.cube
    cube_fdr = c.cube_fdr
    emerging = c.emerging
    layer_boundaries = c.layer_boundaries
    villages = c.villages

    named = emerging[emerging["Place"] != ""]
    cube_hot = spacetime.hot_bins(cube, fdr=cube_fdr)

    if named.empty:
        st.info("No place is a significant hotspot in any step, so there is no cube to "
                "draw. Try a larger hexagon, a longer step, or the strict test off.")
    else:
        view_col, focus_col = st.columns([3, 2], vertical_alignment="bottom")
        camera = view_col.segmented_control(
            "View", list(charts.CUBE_CAMERAS), default="Oblique", key="cube_camera",
        ) or "Oblique"
        focus_names = {
            row["Place"]: f"{row['Place']} {row['Pattern']}"
            + (f" - {str(row['Nearest Village']).split(' (')[0]}"
               if row["Nearest Village"] else "")
            for row in named.to_dict("records")
        }
        # A string, not None, for "all": Streamlit reads a None option as
        # nothing selected and shows an empty box.
        all_places = "All places"
        picked_focus = focus_col.selectbox(
            "Focus", [all_places, *focus_names], key="cube_focus",
            format_func=lambda p, n=focus_names: n.get(p, p),
            help="Zoom the cube onto one place and its neighbours.",
        )
        focus = None if picked_focus == all_places else picked_focus
        step_col, ctx_col = st.columns([6, 1], vertical_alignment="bottom")
        step_fmt = "%b %Y" if cube.step == "Month" else "%d %b %Y"
        slice_labels = ["None"] + [f"{s:{step_fmt}}" for s in cube.steps]
        picked_slice = step_col.select_slider(
            "Highlight a step", slice_labels, value="None", key="cube_slice",
        )
        show_context = ctx_col.toggle("Others", value=True, key="cube_context",
                                      help="Faint dots for places with no pattern.")
        highlight = (None if picked_slice == "None"
                     else cube.steps[slice_labels.index(picked_slice) - 1])
        # Like the maps, a cube that fails to draw says so in its own
        # place: an exception here would stop the script and take the
        # Villages, When and Data tabs down with it.
        try:
            cube_figure = charts.space_time_cube(
                cube, emerging, cube_hot, villages,
                camera=camera, highlight_step=highlight, context=show_context,
                floor_boundaries=layer_boundaries, focus=focus,
            )
        except Exception as exc:  # noqa: BLE001 - the page must survive
            logger.exception("Space-time cube failed")
            st.error(
                f"The space-time cube could not be drawn: {type(exc).__name__}. "
                "Everything else on this page is unaffected."
            )
        else:
            st.plotly_chart(cube_figure, width="stretch", config=CUBE_CONFIG,
                            key="cube_chart")
        st.caption(
            "Each column is a place; each hexagon, a step with events, its area "
            "following the count. Coloured hexagons are steps the place was a "
            "significant hotspot, grey ones steps it was not, so a New place is grey "
            "with a coloured top and a Diminishing one the reverse. The floor carries "
            "the forest boundaries. Drag to rotate, scroll to zoom, click a legend "
            "entry to hide it; the buttons at the top right reset the view and save "
            "it as an image."
        )
