# Data Pipeline Module - Books Web Scraping and SQL Database

This module covers a full data engineering pipeline that scrapes book data
from a website, cleans it up, converts the prices to Indian Rupees, and then
stores everything in a proper relational SQLite database. The goal was to
practice the whole flow from raw messy web data all the way to clean SQL
tables that can be queried.

The whole pipeline lives in one file called scrape.py. It runs as one script
from start to finish.

---

## Section 1 - Scraping the Website

The website I scraped from is books.toscrape.com. It is a practice site made
just for scraping so it is completely fine to use it.

I used the requests library to download each page and BeautifulSoup with the
html.parser to read through the HTML and pull out the pieces I needed.

**Approach (multi-page, multi-category crawl):** The scraper does not just
grab one page. It goes into the category list on the left side of the site
(div.side_categories ul li a) and visits each category one at a time. Inside
a category it also follows the "next" button (li.next a) so it keeps going
through every page of that category instead of stopping at page one.

**Requirement (≥60 books across ≥3 categories):** I kept scraping until I
had at least 60 books collected from 3 or more different categories, for
example Travel, Mystery, and Historical Fiction.

For every single book I grabbed 5 pieces of information:

title, which comes from the h3 and a title attribute on the book link.
price, which comes out as a raw string like "£45.17".
star_rating, which is actually stored as a word inside a CSS class name like
"Two" instead of a number.
availability, which is a text string like "In stock".
category, which comes from the category page heading.

All of this raw data gets saved into a file called books.csv.

---

## Section 2 - Cleaning the Data

Raw scraped data is messy so the next step was a cleaning function called
clean_data() that reads books.csv and turns everything into proper types.

Price became price_gbp. I stripped out the currency symbol and any weird
characters like Â£ that sometimes show up from encoding issues, then
converted what was left into a float.

Star rating became rating. Since the website stores this as a word and not a
number, I made a small dictionary that maps "One" through "Five" to the
numbers 1 through 5. If a rating word did not match anything in the
dictionary I fell back to a default of 3 so the pipeline does not crash.

Availability became in_stock. I checked if the lowercased text contained the
words "in stock" and turned that into a simple True or False.

**Design Decision (real boolean, not text):** when this value gets inserted
into the SQLite books table it is stored as a real 0/1 integer rather than
the text "True"/"False", so it behaves as an actual boolean column in the
database and not just boolean-looking text.

### How I decided what to do with bad or missing data

Some rows were missing important information, so I had to make some rules.

**Design Decision (drop vs. fix):** Dropping rows: if a row was missing its
title or its category I dropped it completely. These two fields are the main
identifiers for a book, so a row without them is not really usable and would
only cause problems later when linking to the database tables.

**Design Decision (median imputation over mean):** Fixing bad prices: if a
price could not be converted into a float, instead of dropping that row I
filled it in using the median price of all the other valid prices I had
already collected. I picked the median instead of the mean because it is
less affected by a few very expensive or very cheap books, so it gives a
more realistic fill-in value. If for some reason there were no valid prices
at all to calculate a median from, I used 25.00 as a safe fallback number.

The cleaned data gets saved into a new file called cleaned_books.csv.

---

## Section 3 - Converting GBP to INR

The prices on the site are in British pounds, so I added a step to convert
every price into Indian Rupees as well.

**Requirement (fixed-rate baseline, no API call):** I used a fixed
conversion rate of 1 GBP equals 105.50 INR. It is a constant number built
into the pipeline, not something pulled live from the internet, so the
pipeline still works even without a network connection.

The formula is simple: price_inr = price_gbp multiplied by 105.50, rounded
to 2 decimal places.

This enriched data gets saved into cleaned_books_inr.csv.

---

## Section 4 - Building a Relational SQLite Database

Once the data was clean I moved it into a real SQLite database file called
masai-books-db.db instead of just leaving everything in one flat CSV.

**Design Decision (normalized two-table schema):** I split the data into
two related tables so there is no repeated category text sitting inside
every single book row.

categories table (the primary table):
category_id as the primary key that auto increments.
category_name which must be unique.

books table (linked to categories through a foreign key):
book_id as the primary key that auto increments.
title, price_gbp, price_inr, rating, in_stock.
category_id which points back to the categories table.

To fill these tables I first inserted every unique category name into the
categories table using INSERT OR IGNORE so duplicates are skipped
automatically. Then for each book I looked up its category_id from the
categories table and inserted the book row along with that id. This is what
makes the two tables properly linked instead of just being two separate
piles of data.

---

## Section 5 - Running SQL Queries

To show that the database actually works I wrote and ran 6 different SQL
queries, each one demonstrating a different SQL concept.

1. SELECT - a plain SELECT * FROM categories to get every category.
2. ORDER BY - selecting all books sorted by price_inr from highest to lowest.
3. LIMIT - selecting just the first 10 rows from the books table.
4. DISTINCT - selecting only the unique category names.
5. WHERE with BETWEEN - filtering books with a price_inr between 1000 and
10000.
6. INNER JOIN - joining categories and books together on category_id so I
can see the category name next to each book title instead of just the id
number.

---

## Section 6 - Comparing SQL JOIN and Pandas merge

The last part of the pipeline reads the data back out of the database into
Pandas DataFrames and checks that doing a join in SQL gives the exact same
result as doing a merge in Pandas.

First I ran a JOIN directly in SQL using pd.read_sql, joining books and
categories on category_id and ordering by book_id.

Then I did it a second way. I read the books table and the categories table
separately into two DataFrames and used pd.merge() to join them together in
memory on category_id.

To check both methods actually agree, I compared the title column from both
results using .equals() and it came back True, which proves the SQL JOIN and
the Pandas merge produced identical data.

I also exported a few of these results as CSV files so they can be checked
without needing to open the database:

categories.csv
books_sorted_by_price_inr.csv
books_between_1000_and_10000.csv
books_and_categories.csv

**Design Decision (persisted query log):** on top of the CSV exports, every
single query above (the query text and its full printed output) also gets
written out to sql_query_log.txt so there is a permanent record of what each
query returned, not just something that flashed by in the terminal when the
script ran.

---

## How to Run the Pipeline

1. Activate the virtual environment.
```bash
source venv/bin/activate
```

2. Install the required libraries.
```bash
pip install -r requirements.txt
```

3. Run the script.
```bash
python data_pipeline/scrape.py
```

---

## Files in this Folder

scrape.py is the script that does everything: scraping, cleaning, currency
conversion, building the database, and running the SQL queries.
books.csv is the raw scraped data straight off the website and is never
touched again after Section 1, so it always stays the original raw export.
**Design Decision:** the SQL export step deliberately writes to its own
books_sorted_by_price_inr.csv instead of overwriting books.csv, so the raw
scrape artifact is never lost.
cleaned_books.csv is the data after cleaning and type conversion.
cleaned_books_inr.csv is the cleaned data with the INR price column added.
masai-books-db.db is the SQLite database with the categories and books
tables.
categories.csv, books_sorted_by_price_inr.csv,
books_between_1000_and_10000.csv, and books_and_categories.csv are extra
exports from the SQL queries and the JOIN vs merge comparison.
sql_query_log.txt is the full saved text output of every SQL query that
db_queries() and read_data_from_pandas() ran, so the query results are kept
around after the script finishes and not just printed to the terminal.
README.md is this file explaining every part of the pipeline.
