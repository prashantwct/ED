import streamlit as st
import pydeck as pdk

def render_map(df):
    """
    Renders a fast GPU-accelerated map using Deck.GL.
    """
    if df.empty:
        st.info("No data available to display on map.")
        return

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[Longitude, Latitude]',
        get_radius="Severity Score * 40",
        get_fill_color=[200, 30, 0, 160],
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=df["Latitude"].mean(),
        longitude=df["Longitude"].mean(),
        zoom=8,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Severity:</b> {Severity Score}<br/>"
                    "<b>Lat:</b> {Latitude}<br/>"
                    "<b>Lon:</b> {Longitude}"
        },
    )

    st.pydeck_chart(deck, use_container_width=True)
