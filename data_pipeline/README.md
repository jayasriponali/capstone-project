# Module 1 — Data Pipeline (`/data_pipeline`)

## 📌 Overview & Objective
This module implements an automated end-to-end data engineering pipeline designed to benchmark catalog pricing and stock availability data for competitive intelligence analytics.

**Pipeline Flow**:
`Web Scraping (requests + BeautifulSoup) → Data Cleaning & Imputation → Currency Conversion → Relational Storage (SQLite) → SQL Queries & Pandas Verification`

---

## 🛠️ Detailed Walkthrough of `scrape.py` Implementation

The pipeline is structured modularly in `data_pipeline/scrape.py` across the following core functions:

### Task 1: Web Scraping (`main()`)
- **Data Source**: `https://books.toscrape.com/` (Public scraping practice website).
- **Libraries Used**: `requests` for HTTP fetching, `BeautifulSoup` (`html.parser`) for DOM parsing.
- **Scraping Scope**:
  - Iterates through category links in `div.side_categories ul li a`.
  - Handles category pagination dynamically by parsing `li.next a` links.
  - Continues until at least **60+ books across ≥ 3 categories** are scraped (e.g., Travel, Mystery, Historical Fiction, etc.).
- **Captured Raw Fields**:
  1. `title`: Product title from `<h3><a title="...">`.
  2. `price`: Raw price string (e.g., `"£45.17"`).
  3. `star_rating`: Text rating extracted from CSS class name (e.g., `"Two"`).
  4. `availability`: Text stock status (e.g., `"In stock"`).
  5. `category`: Category name extracted from category header.
- **Output Artifact**: `books.csv`

---

### Task 2: Data Cleaning & Type Conversion (`clean_data()`)
Reads `books.csv` and cleans raw fields into proper Python data types:

1. **Price (`price_gbp`)**: Strips non-numeric currency symbols (`£`, `Â£`) and converts to `float`.
2. **Star Rating (`rating`)**: Maps textual rating (`"One"`...`"Five"`) into integer values (`1`...`5`) using a dictionary lookup (`rating_map`). Fallback default is set to `3`.
3. **Availability (`in_stock`)**: Evaluates string to boolean (`True` if lowercased availability text contains `"in stock"`, else `False`).

#### Handling Messy Data: Stated Choices & Justifications

* **Dropping Records (Choice & Justification)**:
  - **Rule**: Drop rows where essential identifier fields (`title` or `category`) are missing or empty (`not row.get("title") or not row.get("category")`).
  - **Justification**: The `title` and `category` are essential primary attributes for product identification and foreign key relational mapping. Without these fields, the product data is not meaningful, and keeping them would result in inconsistent orphaned records across CSV files and the SQLite database.

* **Median Price Imputation (Choice & Justification)**:
  - **Rule**: If a price field is corrupted or fails to convert to `float`, impute it using the dataset's pre-calculated **median price**.
  - **Calculation Steps**:
    ```python
    if valid_prices:
        sorted_prices = sorted(valid_prices)
        mid_index = len(sorted_prices) // 2
        median_price = round(sorted_prices[mid_index], 2)
    else:
        median_price = 25.00
    ```
  - **Justification**: Imputing missing prices with the median does not distort the overall price distribution of the dataset and is less sensitive to extreme pricing outliers (as per class notes). It ensures the pipeline processes raw scraped data robustly without crashing.
- **Output Artifact**: `cleaned_books.csv`

---

### Task 3: Baseline Currency Conversion (`convert_GBP_to_INR()`)
Enriches the dataset by converting GBP prices to INR:
- **Required Fixed Conversion Rate**: **`1 GBP = 105.50 INR`** (Fixed project constant with no date reference or external network dependency).
- **Formula**: `price_inr = round(float(price_gbp) * 105.50, 2)`
- **Output Artifact**: `cleaned_books_inr.csv`

---

### Task 4: Normalized Relational SQLite Schema (`insert_into_db()`)
Creates a normalized SQLite database (`masai-books-db.db`) with two tables linked by a Primary Key / Foreign Key relationship:

```sql
-- Categories Table (Primary Table)
CREATE TABLE categories(
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

-- Books Table (Foreign Key Table)
CREATE TABLE books(
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER,
    in_stock BOOLEAN,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories (category_id)
);
```

#### Insertion Workflow:
1. Unique `category_name` entries are inserted into `categories` using `INSERT OR IGNORE`.
2. The generated `category_id` is queried from `categories`.
3. Book records are inserted into `books` with the corresponding `category_id`.

---

### Task 5: Executing SQL Queries (`db_queries()`)
Executes 6 distinct SQL queries against `masai-books-db.db` to demonstrate key SQL clauses:

1. **`SELECT`**: `SELECT * FROM categories;` (Fetches all categories)
2. **`ORDER BY`**: `SELECT * FROM books ORDER BY price_inr DESC;` (Sorts books by price descending)
3. **`LIMIT`**: `SELECT * FROM books LIMIT 10;` (Retrieves top 10 rows)
4. **`DISTINCT`**: `SELECT DISTINCT category_name FROM categories;` (Lists distinct categories)
5. **`BETWEEN` / `WHERE`**: `SELECT * FROM books WHERE price_inr BETWEEN 1000 AND 10000;` (Filters by price range)
6. **`INNER JOIN`**: 
   ```sql
   SELECT categories.category_name, books.title 
   FROM books 
   INNER JOIN categories ON categories.category_id = books.category_id;
   ```

---

### Task 6: Pandas DataFrames & Equivalence Verification (`read_data_from_pandas()`)
Reads SQL data back into Pandas DataFrames and verifies equivalence between relational SQL `JOIN` and Pandas in-memory `pd.merge()`:

1. **SQL JOIN via `pd.read_sql`**:
   ```python
   sql_join_df = pd.read_sql('''
       SELECT b.book_id, b.title, b.price_gbp, b.price_inr, b.rating, b.in_stock, b.category_id, c.category_name
       FROM books b
       JOIN categories c ON b.category_id = c.category_id
       ORDER BY b.book_id
   ''', sqllite_connection)
   ```
2. **In-Memory `pd.merge()`**:
   ```python
   df_books = pd.read_sql("SELECT * FROM books ORDER BY book_id", sqllite_connection)
   df_categories = pd.read_sql("SELECT * FROM categories", sqllite_connection)
   df_merged = pd.merge(df_books, df_categories, on="category_id")
   ```
3. **Equivalence Verification**:
   - `are_equal = sql_join_df["title"].equals(df_merged["title"])`
   - **Result**: `True` (Proves both approaches produce identical outputs).
- **Exported CSV Artifacts**:
  - `categories.csv`
  - `books.csv`
  - `books_between_1000_and_10000.csv`
  - `books_and_categories.csv`


## 🚀 Prerequisites & Execution Steps

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Execute Pipeline Script
```bash
python data_pipeline/scrape.py
```
