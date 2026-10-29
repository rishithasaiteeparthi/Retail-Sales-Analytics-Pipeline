# Retail Sales Analytics and Forecasting Pipeline

Data Science & Data Engineering Internship (SkillDzire / APSCHE)

Raw sales data -> ETL (clean + validate) -> SQLite star-schema warehouse -> SQL analysis -> sales forecasting -> charts.

## Run the pipeline
    pip install -r requirements.txt
    python retail_pipeline.py
Results are saved in the `out` folder (charts, retail_dw.db, report.json).

## Run the dashboard
    streamlit run app.py
Opens in the browser at http://localhost:8501

## Deploy free (live URL)
Upload these files to a public GitHub repo, then go to share.streamlit.io -> New app -> choose the repo -> main file `app.py` -> Deploy.
