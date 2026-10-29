"""Streamlit dashboard for the Retail Sales Analytics and Forecasting Pipeline.
Run locally:  streamlit run app.py
"""
import json, os, subprocess, sys
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

st.set_page_config(page_title="Retail Sales Analytics & Forecasting", layout="wide")
st.title("Retail Sales Analytics and Forecasting Pipeline")
st.caption("Data Science & Data Engineering Internship - ETL -> SQLite star schema -> SQL analysis -> forecasting")

def run_pipeline():
    result = subprocess.run([sys.executable, "retail_pipeline.py"], cwd=HERE, capture_output=True, text=True)
    return result

if not os.path.exists(os.path.join(OUT, "report.json")):
    with st.spinner("Running the pipeline (generate data, ETL, load, analyse, forecast)..."):
        res = run_pipeline()
    if res.returncode != 0:
        st.error("Pipeline failed.")
        st.code(res.stderr)
        st.stop()

if st.button("Re-run pipeline"):
    with st.spinner("Re-running pipeline..."):
        res = run_pipeline()
    if res.returncode != 0:
        st.error("Pipeline failed.")
        st.code(res.stderr)
        st.stop()
    st.success("Pipeline completed.")

report = json.load(open(os.path.join(OUT, "report.json")))
m = report["metrics"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw records", f"{report['raw_rows']:,}")
c2.metric("Clean rows loaded", f"{report['fact_rows']:,}")
c3.metric("Total revenue (millions)", report["revenue_total_m"])
c4.metric("Best model", report["best_model"])

st.subheader("1. ETL data-quality summary")
st.table(pd.DataFrame({
    "Check": ["Duplicate rows removed", "Missing units (filled with group median)", "Missing region (set to Unknown)", "Clean rows loaded"],
    "Count": [report["duplicates"], report["missing_units"], report["missing_region"], report["clean_rows"]],
}))

st.subheader("2. SQL analysis")
st.image(os.path.join(OUT, "chart_category_region.png"), use_container_width=True)
st.dataframe(pd.read_csv(os.path.join(OUT, "sql_revenue_by_category.csv")))

st.subheader("3. Monthly revenue trend")
st.image(os.path.join(OUT, "chart_monthly_trend.png"), use_container_width=True)

st.subheader("4. Forecast vs actual")
st.image(os.path.join(OUT, "chart_forecast.png"), use_container_width=True)
st.table(pd.DataFrame({
    "Model": list(m.keys()),
    "RMSE": [v["RMSE"] for v in m.values()],
    "R2": [v["R2"] for v in m.values()],
}))

st.subheader("5. Category x month heatmap")
st.image(os.path.join(OUT, "chart_heatmap.png"), use_container_width=True)
