import sqlite3
from csv import DictWriter
from csv import DictReader
import requests
import sqlite3
import json
import pandas as pd
from bs4 import BeautifulSoup

def get_conn_cursor():
    sqllite_connection = sqlite3.connect("masai-books-db.db")
    #create the cursor used to execute the sql commands.
    cursor = sqllite_connection.cursor()
    return sqllite_connection, cursor

def main():


    #Task 1: request to the website and scraping the data from the books.toscrape.com
    # make an request to the website
    response = requests.get("https://books.toscrape.com")
    #print the response in the text format
    #print(response.text)

    soup = BeautifulSoup(response.text, 'html.parser')
    #print(soup.prettify())

    # Scrape across multiple book categories (at least 3 categories and 60+ books total)
    categories = soup.select("div.side_categories ul li a")[1:]
    books = []

    for category in categories:
        category_name = category.get_text(strip=True)
        print(f"Category Name ==== {category_name}")
        category_url = category.get("href")
        full_url = "https://books.toscrape.com/" + category_url

        while full_url:
            books_response = requests.get(full_url)
            if books_response.status_code != 200:
                break

            books_soup = BeautifulSoup(books_response.text, 'html.parser')
            product_pods = books_soup.select("article.product_pod")

            for product in product_pods:
                book = {}
                title = product.find("h3").find("a")["title"]

                # getting the product price from the product
                price_info = product.find("p", class_="price_color")
                price = price_info.get_text(strip=True).replace("Â£", "").replace("£","").strip() if price_info else ""

                # rating information
                star_rating_class = product.find("p", "star-rating")
                star_rating = star_rating_class.get("class")[1] if star_rating_class else "Three"

                # availability information
                availability_tag = product.find("p", "instock availability")
                availability = availability_tag.get_text(strip=True) if availability_tag else ""

                book["title"] = title
                book["price"] = price
                book["star_rating"] = star_rating
                book["availability"] = availability
                book["category"] = category_name

                books.append(book)

            # Check for next page in category pagination
            next_li = books_soup.select_one("li.next a")
            if next_li:
                next_href = next_li.get("href")
                parent_url = full_url.rsplit('/', 1)[0]
                full_url = f"{parent_url}/{next_href}"
            else:
                full_url = None

        if len(books) >= 60 and len(set(b["category"] for b in books)) >= 3:
            print(f"Successfully scraped {len(books)} books across {len(set(b['category'] for b in books))} categories!")
            break

    print(f"Total Scraped Books Count: {len(books)}")
        
        #convert the books in to csv format

        
        
    with open("books.csv", "w", newline="", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=["title", "price", "star_rating", "availability", "category"])
        writer.writeheader()
        writer.writerows(books)
    clean_data()
    convert_GBP_to_INR()
    insert_into_db()
    db_queries()
    read_data_from_pandas()
           
# read raw scraped data, clean types, and handle messy rows via median imputation
def clean_data():
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    # Step 1: Read all raw rows and compute median price across valid rows
    valid_prices = []
    rows = []
    with open("books.csv", "r", encoding="utf-8") as file:
        reader = DictReader(file)
        for row in reader:
            rows.append(row)
            try:
                valid_prices.append(float(row["price"]))
            except (ValueError, TypeError):
                pass

    # Pre-calculate median price for imputation using multiple explicit steps
    if valid_prices:
        sorted_prices = sorted(valid_prices)
        mid_index = len(sorted_prices) // 2
        median_price = round(sorted_prices[mid_index], 2)
    else:
        median_price = 25.00

    # Step 2: Write cleaned data with median imputation for any corrupt/missing prices
    with open("cleaned_books.csv", "w", newline="", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=["title", "price_gbp", "rating", "in_stock", "category"])
        writer.writeheader()
        
        for row in rows:
            # 1. DROP ROW CHOICE & JUSTIFICATION:
            # we are droppring the records where essential category and title fields are missing
            if not row.get("title") or not row.get("title").strip() or not row.get("category"):
                print(f"Skipping/Dropping row due to missing essential title/category: {row}")
                continue

            # 2. MEDIAN IMPUTATION CHOICE & JUSTIFICATION:
            # we need to impute the price_gbp column if the price is not a float
            # we are using the median price for the imputation
            # justification for using median imputation
            # it is not distorting the distribution of the data 
            # and the median is less sensitive to outliers as per balaji sir's notes.
            try:
                price_gbp = float(row["price"])
            except (ValueError, TypeError):
                price_gbp = median_price

            # Parse star rating with fallback default
            rating = rating_map.get(row["star_rating"], 3)

            # Parse availability boolean
            in_stock = True if "in stock" in row["availability"].lower() else False

            writer.writerow({
                "title": row["title"].strip(),
                "price_gbp": price_gbp,
                "rating": rating,
                "in_stock": in_stock,
                "category": row["category"]
            })
            
#Convert price_gbp to a price_inr column using the project's fixed baseline conversion rate: 1 GBP = 105.50 INR. 
def convert_GBP_to_INR():
    with open("cleaned_books.csv", "r" ,encoding="utf-8") as file:
        reader = DictReader(file)
        with open("cleaned_books_inr.csv", "w", newline="", encoding="utf-8") as f:
            writer = DictWriter(f, fieldnames=["title", "price_gbp", "price_inr", "rating", "in_stock", "category"])
            writer.writeheader()
            #This is an artificial, project-defined constant for this assignment, 
            # not a live or historical market rate, so it never needs a lookup or a date reference. 
            # This fixed-rate conversion is the required, keyless baseline and is what gets graded 
            # for this task — it requires no external API call and no network access; simply state 
            # this exact rate in your README. (Optional, ungraded stretch — must not affect your required 
            # submission: if you want extra practice with the requests library and explicit HTTP 
            # status-code handling, you may additionally look up any free, keyless currency-conversion API 
            # of your own choosing, check its response status code explicitly, and fall back to the fixed rate 
            # above on any failure. 
            # This is entirely optional; your price_inr column must be fully correct using only the required 
            # fixed-rate baseline, since that path alone is what gets graded.)
            for row in reader:
                price_inr = float(row["price_gbp"]) * 105.50
                writer.writerow({
                    "title": row["title"],
                    "price_gbp": row["price_gbp"],
                    "price_inr": round(price_inr, 2),
                    "rating": row["rating"],
                    "in_stock": row["in_stock"],
                    "category": row["category"]
                })


#Task 4: Design a normalized SQLite schema with at least two tables sharing a primary/foreign key relationship, for example:

def insert_into_db():
    sqllite_connection, cursor = get_conn_cursor()
    cursor.execute("DROP TABLE IF EXISTS books")
    cursor.execute("DROP TABLE IF EXISTS categories")
    #1) categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)
    cursor.execute('''
        CREATE TABLE categories(
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE
        )
    ''')
    #2) books(book_id INTEGER PRIMARY KEY, title TEXT NOT NULL, price_gbp REAL NOT NULL, 
    # price_inr REAL NOT NULL, rating INTEGER, in_stock BOOLEAN, category_id INTEGER, 
    # FOREIGN KEY (category_id) REFERENCES categories (category_id))
    cursor.execute('''
        CREATE TABLE books(
            book_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER,
            in_stock BOOLEAN,
            category_id INTEGER,
            FOREIGN KEY (category_id) REFERENCES categories (category_id)
        )
    ''')


#Task 5: Using Python's sqlite3 (or pandas.DataFrame.to_sql), insert your cleaned, converted data into this schema.
def db_queries():
    sqllite_connection, cursor = get_conn_cursor()
    #Using Python's sqlite3 (or pandas.DataFrame.to_sql), insert your cleaned, 
    # converted data into this schema. Then write and execute at least 5 SQL queries against 
    # the database that collectively demonstrate: SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, 
    # and (IN or BETWEEN) — plus at least one JOIN between your two tables
    #  (e.g., "list the 10 highest-rated books per category"). Save each query string and its output.

    #first insert the categories by reading the /Users/mac/Documents/masai/capstone-project/data_pipeline/cleaned_books_inr.csv 
    # categoreis colum get the unique values.
    with open("cleaned_books_inr.csv", "r", encoding="utf-8") as file:
        reader = DictReader(file)
        for row in reader:
            #we need to insert only unique categories not all 
            #use the on conflict do nothing         
            cursor.execute('''INSERT OR IGNORE INTO categories (category_name) 
            VALUES(?) ''', (row["category"],))
            # now insert the books
            #first get the category_id of the current book
            cursor.execute('''SELECT category_id FROM categories WHERE category_name = ?''', (row["category"],))
            category_id = cursor.fetchone()[0]
            print(category_id)  
            #now insert the book
            cursor.execute('''INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id) 
            VALUES(?, ?, ?, ?, ?, ?) ''', (row["title"], row["price_gbp"], row["price_inr"], row["rating"], row["in_stock"], category_id))
            sqllite_connection.commit()
    
    #1) get All categories
    cursor.execute("SELECT * FROM categories")
    results = cursor.fetchall()
    for result in results:
        print(f"#################### all categories {result}")
    #2) get all products uing order by
    cursor.execute("SELECT * FROM books ORDER BY price_inr DESC")
    results = cursor.fetchall()
    for result in results:
        print(f"#################### all products sorted by price inr {result}")
    #3) get products by limit
    cursor.execute("SELECT * FROM books LIMIT 10")
    results = cursor.fetchall()
    for result in results:
        print(f"#################### top 10 expensive products {result}")
    #4) get  distinct products based on the categories
    cursor.execute("SELECT DISTINCT category_name FROM categories")
    results = cursor.fetchall()
    for result in results:
        print(f"#################### distinct category ids from products {result}")
    #5) get products by
    cursor.execute("SELECT * FROM books WHERE price_inr BETWEEN 1000 AND 10000")
    results = cursor.fetchall()
    for result in results:
        print(f"#################### books between 1000 and 10000 {result}" )

    #6 join categories and books
    cursor.execute("SELECT categories.category_name, books.title from books INNER JOIN categories ON categories.category_id = books.category_id")
    results = cursor.fetchall()
    for result in results:
        print(f"#################### join categories and books {result}" )
    
    sqllite_connection.close()


#Task 6: Read back at least two of the above query 
# results into pandas DataFrames using pd.read_sql(...), 
# and separately reproduce the join-query's result using pd.merge(...) 
# directly on your in-memory DataFrames (no SQL) 
# — show that both approaches produce equivalent output.
def read_data_from_pandas():

    sqllite_connection, cursor = get_conn_cursor()
    
    df = pd.read_sql("SELECT * FROM categories", sqllite_connection)
    print(f" dataframe of categories:\n{df}")
    df.to_csv("categories.csv", index=False)

    df = pd.read_sql("SELECT * FROM books ORDER BY price_inr DESC", sqllite_connection)
    print(f" dataframe of books:\n{df}")
    df.to_csv("books.csv", index=False)

    df = pd.read_sql("SELECT * FROM books WHERE price_inr BETWEEN 1000 AND 10000", sqllite_connection)
    print(f" dataframe of books between 1000 and 10000:\n{df}")
    df.to_csv("books_between_1000_and_10000.csv", index=False)

    # 1. Fetch SQL JOIN via pd.read_sql
    sql_join_df = pd.read_sql('''
        SELECT b.book_id, b.title, b.price_gbp, b.price_inr, b.rating, b.in_stock, b.category_id, c.category_name
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.book_id
    ''', sqllite_connection)
    print(f"\n[1. pd.read_sql SQL JOIN Result]:\n{sql_join_df.head()}")

    # 2. Perform in-memory pd.merge on separate DataFrames
    df_books = pd.read_sql("SELECT * FROM books ORDER BY book_id", sqllite_connection)
    df_categories = pd.read_sql("SELECT * FROM categories", sqllite_connection)
    df_merged = pd.merge(df_books, df_categories, on="category_id")
    print(f"\n[2. In-Memory pd.merge Result]:\n{df_merged.head()}")
    df_merged.to_csv("books_and_categories.csv", index=False)

    # 3. Compare outputs side-by-side to show equivalence
    are_equal = sql_join_df["title"].equals(df_merged["title"])
    print(f"\n comparing the pd.read_sql JOIN and pd.merge outputs (True or False) {are_equal}")

    sqllite_connection.close()


if __name__ == "__main__":
    main()

    
