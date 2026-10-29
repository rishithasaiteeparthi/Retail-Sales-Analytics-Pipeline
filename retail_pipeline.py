"""
Retail Sales Analytics and Forecasting Pipeline
Data Science & Data Engineering Internship (SkillDzire / APSCHE)

Stages: 1) generate raw data  2) ETL (clean + load to SQLite star schema)
        3) SQL analysis        4) ML forecasting  5) charts
Run:  python retail_pipeline.py
"""
import sqlite3, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
OUT = "out"
import os
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 1. RAW DATA
def make_raw():
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    cats = {"Electronics": 1800, "Groceries": 350, "Clothing": 900,
            "Home & Kitchen": 700, "Stationery": 200}
    regions = ["North", "South", "East", "West"]
    rows = []
    for d in dates:
        season = 1 + 0.25 * np.sin(2 * np.pi * (d.dayofyear - 80) / 365)
        festive = 1.5 if d.month in (10, 11) else 1.0
        weekend = 1.2 if d.dayofweek >= 5 else 1.0
        trend = 1 + 0.0004 * (d - dates[0]).days
        for c, price in cats.items():
            for r in regions:
                units = max(0, int(rng.normal(12, 3) * season * festive * weekend * trend))
                rows.append([d.strftime("%Y-%m-%d"), c, r, units, round(price * rng.uniform(0.9, 1.1), 2)])
    df = pd.DataFrame(rows, columns=["order_date", "category", "region", "units", "unit_price"])
    # inject typical data-quality problems
    df.loc[rng.choice(df.index, 300, replace=False), "units"] = np.nan
    df.loc[rng.choice(df.index, 150, replace=False), "region"] = None
    df = pd.concat([df, df.sample(120, random_state=1)], ignore_index=True)  # duplicates
    return df

raw = make_raw()
raw.to_csv(f"{OUT}/raw_sales.csv", index=False)

# ---------------------------------------------------------------- 2. ETL
report = {"raw_rows": len(raw), "missing_units": int(raw.units.isna().sum()),
          "missing_region": int(raw.region.isna().sum()),
          "duplicates": int(raw.duplicated().sum())}
df = raw.drop_duplicates().copy()
df["region"] = df["region"].fillna("Unknown")
df["units"] = df.groupby(["category", "region"])["units"].transform(lambda s: s.fillna(s.median()))
df["units"] = df["units"].fillna(0).astype(int)
df["order_date"] = pd.to_datetime(df["order_date"])
df["revenue"] = df["units"] * df["unit_price"]
assert (df.units >= 0).all() and df.revenue.notna().all()        # data-quality checks
report["clean_rows"] = len(df)

con = sqlite3.connect(f"{OUT}/retail_dw.db")
cur = con.cursor()
cur.executescript("""
DROP TABLE IF EXISTS fact_sales; DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_category; DROP TABLE IF EXISTS dim_region;
CREATE TABLE dim_date(date_id INTEGER PRIMARY KEY, order_date TEXT, year INT, month INT, day_of_week INT);
CREATE TABLE dim_category(category_id INTEGER PRIMARY KEY, category TEXT);
CREATE TABLE dim_region(region_id INTEGER PRIMARY KEY, region TEXT);
CREATE TABLE fact_sales(sale_id INTEGER PRIMARY KEY AUTOINCREMENT, date_id INT, category_id INT,
  region_id INT, units INT, unit_price REAL, revenue REAL);
""")
dd = pd.DataFrame({"order_date": sorted(df.order_date.unique())})
dd["date_id"] = range(1, len(dd) + 1)
dd["year"] = dd.order_date.dt.year; dd["month"] = dd.order_date.dt.month
dd["day_of_week"] = dd.order_date.dt.dayofweek
dc = pd.DataFrame({"category": sorted(df.category.unique())}); dc["category_id"] = range(1, len(dc) + 1)
dr = pd.DataFrame({"region": sorted(df.region.unique())}); dr["region_id"] = range(1, len(dr) + 1)
dd2 = dd.copy(); dd2["order_date"] = dd2.order_date.dt.strftime("%Y-%m-%d")
dd2[["date_id", "order_date", "year", "month", "day_of_week"]].to_sql("dim_date", con, if_exists="append", index=False)
dc[["category_id", "category"]].to_sql("dim_category", con, if_exists="append", index=False)
dr[["region_id", "region"]].to_sql("dim_region", con, if_exists="append", index=False)
fact = (df.merge(dd, on="order_date").merge(dc, on="category").merge(dr, on="region")
          [["date_id", "category_id", "region_id", "units", "unit_price", "revenue"]])
fact.to_sql("fact_sales", con, if_exists="append", index=False)
con.commit()
report["fact_rows"] = len(fact)

# ---------------------------------------------------------------- 3. SQL ANALYSIS
q_cat = """SELECT c.category, ROUND(SUM(f.revenue)/1e6,2) AS revenue_m
           FROM fact_sales f JOIN dim_category c USING(category_id)
           GROUP BY c.category ORDER BY revenue_m DESC"""
q_reg = """SELECT r.region, ROUND(SUM(f.revenue)/1e6,2) AS revenue_m
           FROM fact_sales f JOIN dim_region r USING(region_id) GROUP BY r.region ORDER BY revenue_m DESC"""
q_mon = """SELECT d.year, d.month, SUM(f.revenue) AS revenue
           FROM fact_sales f JOIN dim_date d USING(date_id) GROUP BY d.year, d.month ORDER BY d.year, d.month"""
cat_df = pd.read_sql(q_cat, con); reg_df = pd.read_sql(q_reg, con); mon_df = pd.read_sql(q_mon, con)
cat_df.to_csv(f"{OUT}/sql_revenue_by_category.csv", index=False)

# ---------------------------------------------------------------- 4. FORECASTING MODEL
mon_df["t"] = range(len(mon_df))
mon_df["sin"] = np.sin(2 * np.pi * mon_df.month / 12)
mon_df["cos"] = np.cos(2 * np.pi * mon_df.month / 12)
mon_df["festive"] = mon_df.month.isin([10, 11]).astype(int)
feats = ["t", "sin", "cos", "festive"]
train, test = mon_df.iloc[:18], mon_df.iloc[18:]
models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42)}
results = {}
for name, m in models.items():
    m.fit(train[feats], train.revenue)
    p = m.predict(test[feats])
    results[name] = {"RMSE": float(np.sqrt(mean_squared_error(test.revenue, p))),
                     "R2": float(r2_score(test.revenue, p)), "pred": p}
best = max(results, key=lambda k: results[k]["R2"])
report["best_model"] = best
report["metrics"] = {k: {"RMSE": round(v["RMSE"]), "R2": round(v["R2"], 3)} for k, v in results.items()}
report["revenue_total_m"] = round(float(df.revenue.sum()) / 1e6, 2)
report["top_category"] = cat_df.iloc[0].category
report["top_region"] = reg_df.iloc[0].region
json.dump(report, open(f"{OUT}/report.json", "w"), indent=2)
print(json.dumps(report, indent=2))

# ---------------------------------------------------------------- 5. CHARTS
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
G, B = "#1f7a5a", "#2b5fa8"

fig, ax = plt.subplots(1, 2, figsize=(10, 5.6))
ax[0].barh(cat_df.category[::-1], cat_df.revenue_m[::-1], color=G); ax[0].set_title("Revenue by Category (millions)")
ax[1].bar(reg_df.region, reg_df.revenue_m, color=B); ax[1].set_title("Revenue by Region (millions)")
plt.tight_layout(); plt.savefig(f"{OUT}/chart_category_region.png", dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(10, 5.6))
labels = [f"{int(y)}-{int(m):02d}" for y, m in zip(mon_df.year, mon_df.month)]
ax.plot(labels, mon_df.revenue / 1e6, marker="o", color=G)
ax.set_title("Monthly Revenue Trend (millions)"); plt.xticks(rotation=60, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/chart_monthly_trend.png", dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(10, 5.6))
tl = labels[18:]
ax.plot(tl, test.revenue / 1e6, marker="o", color="black", label="Actual")
for name, c in zip(results, [B, "#d9822b"]):
    ax.plot(tl, results[name]["pred"] / 1e6, marker="s", linestyle="--", color=c,
            label=f"{name} (R2={results[name]['R2']:.2f})")
ax.set_title("Forecast vs Actual Revenue (millions) - Test Period"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/chart_forecast.png", dpi=170); plt.close()

pivot = df.assign(month=df.order_date.dt.month).pivot_table(index="category", columns="month",
                                                              values="revenue", aggfunc="sum") / 1e6
fig, ax = plt.subplots(figsize=(10, 5.6))
im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto")
ax.set_xticks(range(12)); ax.set_xticklabels(range(1, 13)); ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index); ax.set_title("Revenue Heatmap: Category x Month (millions)")
plt.colorbar(im); plt.tight_layout(); plt.savefig(f"{OUT}/chart_heatmap.png", dpi=170); plt.close()
con.close()
