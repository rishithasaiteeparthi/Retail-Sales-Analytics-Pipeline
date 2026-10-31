# Retail Sales Analytics and Forecasting Pipeline

A Data Science & Data Engineering capstone project built during the **SkillDzire x APSCHE** short-term virtual internship (01-05-2026 to 30-06-2026).

The project takes raw retail sales data, cleans and validates it with an ETL process, stores it in a **star-schema data warehouse (SQLite)**, analyses it with SQL, and **forecasts monthly revenue** with machine learning. Results are shown in an interactive **Streamlit dashboard**.

## Pipeline

```
Raw sales data (CSV)
      |
      v
ETL (Python / Pandas)  -> remove duplicates, fill missing values, fix types, validate
      |
      v
SQLite star schema     -> fact_sales + dim_date, dim_category, dim_region
      |
      v
SQL analysis           -> revenue by category, region and month
      |
      v
Forecasting model      -> Linear Regression vs Random Forest (Scikit-learn)
      |
      v
Charts + Streamlit dashboard
```

## Features

- **Automated ETL:** removes duplicate rows, fills missing units with group medians, replaces missing regions with "Unknown", fixes data types and runs data-quality checks.
- **Data warehouse:** star schema with one fact table (`fact_sales`) and three dimension tables (`dim_date`, `dim_category`, `dim_region`) in SQLite.
- **SQL analysis:** joins and aggregations for revenue by category, region and month.
- **Forecasting:** trend, seasonal and festive-month features; two models compared using RMSE and R2 on a 6-month test period.
- **Web dashboard (RetailIQ):** five pages with a left menu - Dashboard (KPI cards + charts), Sales Analysis (live SQL queries), Forecast (choose category, region and horizon), Data Pipeline (stages and model comparison) and Data Explorer (filter and download data). Year, category and region filters are in the sidebar.

## Results (sample run)

| Item | Value |
|---|---|
| Raw records | 14,740 |
| Duplicates removed | 120 |
| Missing units filled | 301 |
| Missing regions set to "Unknown" | 152 |
| Clean rows loaded to warehouse | 14,620 |
| Top category / top region | Electronics / East |
| Best model | Linear Regression (R2 = 0.944, RMSE = 332,627) |
| Random Forest | R2 = 0.438, RMSE = 1,052,521 |

> The dataset is **simulated** by the script (2024-2025, 5 categories x 4 regions) with deliberate data-quality problems, so no download is needed. Revenue is in simulated currency units.

## Tech stack

Python 3 - Pandas - NumPy - SQLite (SQL) - Matplotlib - Scikit-learn - Streamlit - Git/GitHub

## Project structure

```
.
|-- app.py                # Streamlit dashboard
|-- retail_pipeline.py    # ETL + warehouse + SQL analysis + forecasting + charts
|-- requirements.txt      # Python libraries
`-- README.md
```

Running the pipeline creates an `out/` folder containing the raw data (`raw_sales.csv`), `retail_dw.db` (the SQLite warehouse), `report.json`, and the chart images.

## How to run

**1. Install Python 3.9 or newer** from [python.org](https://www.python.org/downloads/) and tick *"Add python.exe to PATH"*.

**2. Install the libraries**

```bash
py -m pip install -r requirements.txt
```

(On Mac/Linux use `python3 -m pip install -r requirements.txt`.)

**3. Run the pipeline only**

```bash
py retail_pipeline.py
```

**4. Run the dashboard**

```bash
py -m streamlit run app.py
```

The browser opens at `http://localhost:8501`. The first load runs the pipeline automatically; use the **Re-run pipeline** button to run it again.

## Deploy the dashboard (free)

1. Upload this repository to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **New app**, choose this repository, set the main file to `app.py`, and click **Deploy**.

## Author

**Teeparthi Rishitha Sai** (23MH1A1257)
B.Tech Information Technology, Aditya College of Engineering & Technology (A), Surampalem - JNTUK

Internship: Data Science & Data Engineering, SkillDzire in collaboration with the Andhra Pradesh State Council of Higher Education (APSCHE).
