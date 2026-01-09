import streamlit as st
from core.data_loader import load_and_validate_csv
from core.analytics import compute_severity, compute_kpis
from core.spatial import attach_nearest_village
from core.map_engine import render_map
from core.report import generate_html_report

st.set_page_config(page_title="Elephant Dashboard", layout="wide", page_icon="🐘")
st.title("🐘 Elephant Sighting & Conflict Dashboard")

uploaded = st.file_uploader("Upload CSV", type="csv")

if not uploaded:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = load_and_validate_csv(uploaded)

df["Severity Score"] = compute_severity(df)
df = attach_nearest_village(df)

kpis = compute_kpis(df)

c1, c2, c3 = st.columns(3)
c1.metric("Entries", kpis["entries"])
c2.metric("Conflicts", kpis["conflicts"])
c3.metric("Severity", f"{kpis['severity']:.1f}")

st.subheader("📍 Spatial View")
render_map(df)

if st.button("Generate Report"):
    html = generate_html_report(df, df["Date"].min(), df["Date"].max())
    st.download_button(
        "Download Report",
        html,
        "Elephant_Report.html",
        mime="text/html"
    )
