# Zepto Catalog Benchmarking & Data Pipeline Capstone Project

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
