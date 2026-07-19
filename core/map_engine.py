import streamlit as st
import pydeck as pdk

from core.analytics import classify_conflict

# Most severe first -- also the legend order.
CATEGORY_COLORS = {
    "Death":    [120, 0, 10, 210],
    "Injury":   [214, 39, 40, 195],
    "House":    [230, 140, 20, 180],
    "Crop":     [200, 170, 40, 165],
    "Presence": [95, 140, 95, 130],
}

# Pixel floor/ceiling for point size. This is what keeps a severe point
# visible at landscape zoom -- previously radius was geographic metres
# only, which shrinks to a fraction of a screen pixel once the view
# spans 100+ km, making every point on the map look identical.
RADIUS_MIN_PIXELS = 3
RADIUS_MAX_PIXELS = 16


def render_map(df):
    """
    Renders a GPU-accelerated map, colored by conflict category
    (death/injury/house/crop/presence) and sized by severity, with a
    pixel floor so the most severe points stay visible regardless of
    zoom level.
    """
    if df.empty:
        st.info("No data available to display on map.")
        return

    df = df.copy()
    if "Conflict Category" not in df.columns:
        df["Conflict Category"] = classify_conflict(df)

    df["fill_color"] = df["Conflict Category"].map(CATEGORY_COLORS)
    df["fill_color"] = df["fill_color"].apply(
        lambda c: c if isinstance(c, list) else CATEGORY_COLORS["Presence"]
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[Longitude, Latitude]",
        get_radius="Severity Score * 40",
        radius_min_pixels=RADIUS_MIN_PIXELS,
        radius_max_pixels=RADIUS_MAX_PIXELS,
        get_fill_color="fill_color",
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=float(df["Latitude"].mean()),
        longitude=float(df["Longitude"].mean()),
        zoom=8,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>{Conflict Category}</b><br/>"
                    "Division: {Division} | Range: {Range}<br/>"
                    "Beat: {Beat}<br/>"
                    "Severity: {Severity Score}"
        },
    )
    st.pydeck_chart(deck, width='stretch')

    legend = " &nbsp;&nbsp; ".join(
        f'<span style="color:rgb({r},{g},{b});font-size:16px">&#9679;</span> {cat}'
        for cat, (r, g, b, a) in CATEGORY_COLORS.items()
    )
    st.markdown(legend, unsafe_allow_html=True)
