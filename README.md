# Zepto Catalog Benchmarking & Data Pipeline Capstone Project

## Module 2 — Analytics Pipeline (`/analytics`)

The analytics module covers the full Titanic data analysis and machine learning pipeline.
See [analytics/README.md](analytics/README.md) for a section-by-section breakdown of all 15 tasks including EDA, modeling, imbalance handling, hyperparameter tuning, regression, and the saved pipeline.

Key files inside `/analytics`:
- `01_eda.ipynb` — data loading, cleaning, univariate, bivariate, and multivariate analysis
- `02_modeling.ipynb` — classification, evaluation, imbalance handling, tuning, regression, and pipeline save
- `titanic.csv` — committed offline fallback loadable via pd.read_csv
- `titanic_survival_pipeline.pkl` — saved end-to-end pipeline for deployment

---

## Module 1 — Data Pipeline (`/data_pipeline`)

This repository contains the complete end-to-end data pipeline for catalog pricing and availability benchmarking.

### Detailed Module Documentation
For a complete task-by-task breakdown, database schema, data cleaning justifications, and grading checklist, see **[data_pipeline/README.md](file:///Users/mac/Documents/masai/capstone-project/data_pipeline/README.md)**.

### Pipeline Highlights
- **Live Web Scraping**: Scrapes >60 products across >3 categories from `books.toscrape.com`.
- **Data Cleaning & Imputation**: Converts ratings (1–5), booleans (`in_stock`), and applies median imputation for numeric price anomalies. Drops incomplete records missing title or category.
- **Fixed Baseline Rate**: Converts `price_gbp` to `price_inr` using the fixed baseline constant **`1 GBP = 105.50 INR`**.
- **SQLite Database**: Schema with `categories` (PK) and `books` (FK).
- **SQL & Pandas Verification**: Executes 6+ SQL queries (including `JOIN`) and validates equivalence against `pd.merge()`.

### Quickstart

```bash
# 1. Activate venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run data pipeline
python data_pipeline/scrape.py
```
