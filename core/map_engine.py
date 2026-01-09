import pydeck as pdk
import streamlit as st


def render_map(df):
if df.empty:
st.info("No map data")
return


layer = pdk.Layer(
'ScatterplotLayer',
df,
get_position='[Longitude, Latitude]',
get_radius='Severity Score * 40',
get_fill_color='[200, 30, 0, 160]',
pickable=True
)


view = pdk.ViewState(
latitude=df['Latitude'].mean(),
longitude=df['Longitude'].mean(),
zoom=8
)


st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))
