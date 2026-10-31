"""
RetailIQ - Retail Sales Analytics and Forecasting Pipeline (Streamlit web app)
Run locally:  py -m streamlit run app.py
"""
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DB = os.path.join(OUT, "retail_dw.db")
REPORT = os.path.join(OUT, "report.json")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="RetailIQ | Sales Analytics & Forecasting", page_icon="📈", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1250px;}
[data-testid="stSidebar"] {background: #f4f8f6; border-right: 1px solid #e3ece7;}
.brand {display:flex; align-items:center; gap:10px; font-size:1.35rem; font-weight:800; color:#0f172a; margin-bottom:2px;}
.brand span.logo {background:#10b981; color:white; border-radius:10px; padding:4px 9px; font-size:1.1rem;}
.brand em {color:#10b981; font-style:normal;}
.tag {color:#64748b; font-size:0.78rem; margin-bottom:14px;}
.hero {background: linear-gradient(120deg,#064e3b 0%,#10b981 100%); color:white; border-radius:18px;
       padding:26px 30px; margin-bottom:18px;}
.hero h1 {color:white; font-size:1.8rem; margin:0 0 6px 0; padding:0;}
.hero p {color:#d1fae5; margin:0; font-size:0.98rem;}
.kpi {background:white; border:1px solid #e5e7eb; border-radius:16px; padding:16px 18px;
      box-shadow:0 1px 3px rgba(15,23,42,.06);}
.kpi .lbl {font-size:0.72rem; letter-spacing:.06em; text-transform:uppercase; color:#64748b; font-weight:600;}
.kpi .val {font-size:1.7rem; font-weight:800; color:#0f172a; margin-top:2px;}
.kpi .sub {font-size:0.78rem; color:#10b981; font-weight:600;}
.card {background:white; border:1px solid #e5e7eb; border-radius:16px; padding:18px 20px; margin-bottom:12px;
       box-shadow:0 1px 3px rgba(15,23,42,.06);}
.card h4 {margin:0 0 6px 0; color:#0f172a;}
.card p {margin:0; color:#475569; font-size:0.92rem;}
.stage {border-left:5px solid #10b981;}
.pill {display:inline-block; background:#d1fae5; color:#065f46; border-radius:999px; padding:2px 12px;
       font-size:0.78rem; font-weight:700;}
.section {font-size:1.15rem; font-weight:800; color:#0f172a; margin:14px 0 6px 0;}
</style>
""",
    unsafe_allow_html=True,
)


def kpi(label, value, sub=""):
    return f'<div class="kpi"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def card(title, text, cls=""):
    return f'<div class="card {cls}"><h4>{title}</h4><p>{text}</p></div>'


# ------------------------------------------------------------------ pipeline + data
def run_pipeline():
    return subprocess.run([sys.executable, "retail_pipeline.py"], cwd=HERE, capture_output=True, text=True)


if not (os.path.exists(DB) and os.path.exists(REPORT)):
    with st.spinner("First start: running the pipeline (generate data, ETL, load warehouse, forecast)..."):
        res = run_pipeline()
    if res.returncode != 0:
        st.error("The pipeline failed to run.")
        st.code(res.stderr)
        st.stop()


@st.cache_data(show_spinner=False)
def load_report():
    with open(REPORT) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_facts():
    con = sqlite3.connect(DB)
    df = pd.read_sql(
        """SELECT d.order_date AS date, d.year, d.month, c.category, r.region,
                  f.units, f.unit_price, f.revenue
           FROM fact_sales f
           JOIN dim_date d ON f.date_id = d.date_id
           JOIN dim_category c ON f.category_id = c.category_id
           JOIN dim_region r ON f.region_id = r.region_id""",
        con,
    )
    con.close()
    return df


def sql(query):
    con = sqlite3.connect(DB)
    try:
        return pd.read_sql(query, con)
    finally:
        con.close()


report = load_report()
facts = load_facts()
ALL_CATS = sorted(facts["category"].unique())
ALL_REGIONS = sorted(facts["region"].unique())
ALL_YEARS = sorted(int(y) for y in facts["year"].unique())

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown(
    '<div class="brand"><span class="logo">R</span>Retail<em>IQ</em></div>'
    '<div class="tag">Sales Analytics & Forecasting</div>',
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🔎 Sales Analysis", "🔮 Forecast", "⚙️ Data Pipeline", "🗂️ Data Explorer"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")
sel_years = st.sidebar.multiselect("Year", ALL_YEARS, default=ALL_YEARS)
sel_cats = st.sidebar.multiselect("Category", ALL_CATS, default=ALL_CATS)
sel_regs = st.sidebar.multiselect("Region", ALL_REGIONS, default=ALL_REGIONS)
st.sidebar.markdown("---")
st.sidebar.caption("Teeparthi Rishitha Sai · 23MH1A1257")
st.sidebar.caption("SkillDzire × APSCHE Internship")

view = facts[facts["year"].isin(sel_years) & facts["category"].isin(sel_cats) & facts["region"].isin(sel_regs)]


def monthly_series(df):
    m = df.groupby(["year", "month"], as_index=False)["revenue"].sum().sort_values(["year", "month"])
    m["label"] = m["year"].astype(int).astype(str) + "-" + m["month"].astype(int).astype(str).str.zfill(2)
    return m


def features(t, month):
    month = np.asarray(month)
    return pd.DataFrame(
        {
            "t": np.asarray(t),
            "sin": np.sin(2 * np.pi * month / 12),
            "cos": np.cos(2 * np.pi * month / 12),
            "festive": np.isin(month, [10, 11]).astype(int),
        }
    )


# ------------------------------------------------------------------ pages
if page.startswith("📊"):
    st.markdown(
        '<div class="hero"><h1>Retail Sales Analytics and Forecasting Pipeline</h1>'
        "<p>Raw data → ETL → SQLite star-schema warehouse → SQL analysis → machine-learning forecast. "
        "Use the filters on the left to explore the sales.</p></div>",
        unsafe_allow_html=True,
    )
    if view.empty:
        st.warning("No data for the selected filters. Please select at least one year, category and region.")
        st.stop()
    total_rev = view["revenue"].sum()
    mser = monthly_series(view)
    top_cat = view.groupby("category")["revenue"].sum().idxmax()
    top_reg = view.groupby("region")["revenue"].sum().idxmax()
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Total revenue", f"{total_rev / 1e6:,.2f} M", "selected filters"), unsafe_allow_html=True)
    c2.markdown(kpi("Units sold", f"{int(view['units'].sum()):,}", f"{len(view):,} records"), unsafe_allow_html=True)
    c3.markdown(kpi("Top category", top_cat, f"top region: {top_reg}"), unsafe_allow_html=True)
    c4.markdown(
        kpi("Best model", report["best_model"], f"R² = {report['metrics'][report['best_model']]['R2']}"),
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section">Monthly revenue trend</div>', unsafe_allow_html=True)
    st.line_chart(mser.set_index("label")["revenue"], height=300)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section">Revenue by category</div>', unsafe_allow_html=True)
        st.bar_chart(view.groupby("category")["revenue"].sum().sort_values(ascending=False), height=280)
    with right:
        st.markdown('<div class="section">Revenue by region</div>', unsafe_allow_html=True)
        st.bar_chart(view.groupby("region")["revenue"].sum().sort_values(ascending=False), height=280)

elif page.startswith("🔎"):
    st.markdown('<div class="hero"><h1>Sales Analysis</h1><p>These tables come from live SQL queries on the '
                "star-schema warehouse (retail_dw.db).</p></div>", unsafe_allow_html=True)
    q1 = """SELECT c.category, ROUND(SUM(f.revenue)/1000000.0, 2) AS revenue_millions,
       SUM(f.units) AS units_sold
FROM fact_sales f
JOIN dim_category c ON f.category_id = c.category_id
GROUP BY c.category
ORDER BY revenue_millions DESC"""
    q2 = """SELECT d.year, d.month, ROUND(SUM(f.revenue)/1000000.0, 2) AS revenue_millions
FROM fact_sales f
JOIN dim_date d ON f.date_id = d.date_id
GROUP BY d.year, d.month
ORDER BY revenue_millions DESC
LIMIT 5"""
    q3 = """SELECT r.region, ROUND(SUM(f.revenue)/1000000.0, 2) AS revenue_millions
FROM fact_sales f
JOIN dim_region r ON f.region_id = r.region_id
GROUP BY r.region
ORDER BY revenue_millions DESC"""
    st.markdown('<div class="section">1. Revenue by category</div>', unsafe_allow_html=True)
    st.code(q1, language="sql")
    st.dataframe(sql(q1))
    st.markdown('<div class="section">2. Top 5 revenue months</div>', unsafe_allow_html=True)
    st.code(q2, language="sql")
    st.dataframe(sql(q2))
    st.markdown('<div class="section">3. Revenue by region</div>', unsafe_allow_html=True)
    st.code(q3, language="sql")
    st.dataframe(sql(q3))
    st.markdown('<div class="section">4. Category × month heatmap (selected filters)</div>', unsafe_allow_html=True)
    if view.empty:
        st.warning("No data for the selected filters.")
    else:
        pv = view.pivot_table(index="category", columns="month", values="revenue", aggfunc="sum") / 1e6
        pv.columns = [MONTHS[int(c) - 1] for c in pv.columns]
        st.dataframe(pv.round(2).style.background_gradient(cmap="Greens", axis=None))

elif page.startswith("🔮"):
    st.markdown('<div class="hero"><h1>Revenue Forecast</h1><p>Pick a scope and a horizon. A regression model using trend, '
                "seasonal and festive-month features predicts future monthly revenue.</p></div>", unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    scope_cat = fc1.selectbox("Category", ["All categories"] + ALL_CATS)
    scope_reg = fc2.selectbox("Region", ["All regions"] + ALL_REGIONS)
    horizon = fc3.slider("Months to forecast", 3, 12, 6)
    base = facts
    if scope_cat != "All categories":
        base = base[base["category"] == scope_cat]
    if scope_reg != "All regions":
        base = base[base["region"] == scope_reg]
    ms = monthly_series(base).reset_index(drop=True)
    ms["t"] = np.arange(len(ms))
    X = features(ms["t"].values, ms["month"].values)
    # honest check on the last 6 known months
    n_test = 6
    chk = LinearRegression().fit(X.iloc[:-n_test], ms["revenue"].iloc[:-n_test])
    pred_test = chk.predict(X.iloc[-n_test:])
    r2 = r2_score(ms["revenue"].iloc[-n_test:], pred_test)
    rmse = float(np.sqrt(mean_squared_error(ms["revenue"].iloc[-n_test:], pred_test)))
    # final model on all history
    model = LinearRegression().fit(X, ms["revenue"])
    last_year, last_month = int(ms["year"].iloc[-1]), int(ms["month"].iloc[-1])
    fut = []
    y, m = last_year, last_month
    for i in range(1, horizon + 1):
        m += 1
        if m > 12:
            m, y = 1, y + 1
        fut.append((y, m, len(ms) - 1 + i))
    fdf = pd.DataFrame(fut, columns=["year", "month", "t"])
    fdf["label"] = fdf["year"].astype(str) + "-" + fdf["month"].astype(str).str.zfill(2)
    fdf["forecast"] = model.predict(features(fdf["t"].values, fdf["month"].values)).clip(min=0)
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi("Model", "Linear Regression", "trend + season + festive"), unsafe_allow_html=True)
    k2.markdown(kpi("Check on last 6 months", f"R² = {r2:.3f}", f"RMSE = {rmse:,.0f}"), unsafe_allow_html=True)
    k3.markdown(kpi("Forecast total", f"{fdf['forecast'].sum() / 1e6:,.2f} M", f"next {horizon} months"), unsafe_allow_html=True)
    hist = ms.set_index("label")[["revenue"]].rename(columns={"revenue": "Actual"})
    fore = fdf.set_index("label")[["forecast"]].rename(columns={"forecast": "Forecast"})
    chart = pd.concat([hist, fore], axis=0)
    chart.loc[hist.index[-1], "Forecast"] = hist["Actual"].iloc[-1]  # join the two lines
    st.markdown('<div class="section">Actual vs forecast (monthly revenue)</div>', unsafe_allow_html=True)
    st.line_chart(chart, height=340)
    st.markdown('<div class="section">Forecast table</div>', unsafe_allow_html=True)
    out_tbl = fdf[["label", "forecast"]].rename(columns={"label": "Month", "forecast": "Forecast revenue"})
    out_tbl["Forecast revenue"] = out_tbl["Forecast revenue"].round(0)
    st.dataframe(out_tbl)
    st.download_button("⬇️ Download forecast (CSV)", out_tbl.to_csv(index=False), file_name="forecast.csv", mime="text/csv")

elif page.startswith("⚙️"):
    st.markdown('<div class="hero"><h1>Data Pipeline</h1><p>How raw data becomes trustworthy, analysis-ready data.</p></div>',
                unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(kpi("Raw records", f"{report['raw_rows']:,}", "ingested"), unsafe_allow_html=True)
    p2.markdown(kpi("Duplicates removed", f"{report['duplicates']:,}", "dropped"), unsafe_allow_html=True)
    p3.markdown(kpi("Missing values fixed", f"{report['missing_units'] + report['missing_region']:,}",
                    f"{report['missing_units']} units + {report['missing_region']} regions"), unsafe_allow_html=True)
    p4.markdown(kpi("Clean rows loaded", f"{report['fact_rows']:,}", "into fact_sales"), unsafe_allow_html=True)
    st.markdown('<div class="section">Pipeline stages</div>', unsafe_allow_html=True)
    st.markdown(card("1 · Extract", "Raw sales records (date, category, region, units, unit price) are generated/read into a "
                     "Pandas DataFrame and saved as raw_sales.csv.", "stage"), unsafe_allow_html=True)
    st.markdown(card("2 · Transform", "Duplicates removed, missing units filled with the group median, missing regions set to "
                     "'Unknown', dates converted, revenue = units × price, and data-quality assertions run.", "stage"),
                unsafe_allow_html=True)
    st.markdown(card("3 · Load", "Clean data is loaded into a SQLite star schema: fact_sales with dim_date, dim_category "
                     "and dim_region.", "stage"), unsafe_allow_html=True)
    st.markdown(card("4 · Analyse", "SQL joins and aggregations produce revenue by category, region and month.", "stage"),
                unsafe_allow_html=True)
    st.markdown(card("5 · Forecast", "Linear Regression and Random Forest are compared on a 6-month test period; the better "
                     "model is selected.", "stage"), unsafe_allow_html=True)
    st.markdown('<div class="section">Model comparison (6-month test period)</div>', unsafe_allow_html=True)
    mt = pd.DataFrame(
        {"Model": list(report["metrics"].keys()),
         "RMSE": [v["RMSE"] for v in report["metrics"].values()],
         "R²": [v["R2"] for v in report["metrics"].values()]}
    )
    st.dataframe(mt)
    st.markdown(f'Selected model: <span class="pill">{report["best_model"]}</span>', unsafe_allow_html=True)
    st.markdown('<div class="section">Run the pipeline again</div>', unsafe_allow_html=True)
    if st.button("▶ Re-run pipeline"):
        with st.spinner("Running pipeline..."):
            res = run_pipeline()
        if res.returncode == 0:
            st.cache_data.clear()
            st.success("Pipeline completed successfully. Reload the page to see refreshed numbers.")
        else:
            st.error("Pipeline failed.")
            st.code(res.stderr)

else:
    st.markdown('<div class="hero"><h1>Data Explorer</h1><p>Browse the cleaned warehouse data and download it.</p></div>',
                unsafe_allow_html=True)
    st.caption(f"{len(view):,} rows match the current filters")
    show = view.sort_values("date").reset_index(drop=True)
    st.dataframe(show.head(1000))
    st.download_button("⬇️ Download filtered data (CSV)", show.to_csv(index=False), file_name="retail_sales_filtered.csv",
                       mime="text/csv")
