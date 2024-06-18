import json
import sqlite3
import os
import re

def create_database(db_name="cvpr_papers.db"):
    """Creates a SQLite database with the specified name and table structure."""

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER NOT NULL,
            title TEXT NOT NULL,
            authors TEXT,
            abstract TEXT,
            pdf_link TEXT
        )
    """)

    conn.commit()
    conn.close()

def insert_data(data, year, db_name="cvpr_papers.db"):
    """Inserts data into the SQLite database."""

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    for item in data:
        title = item.get("title")
        authors = ", ".join(item.get("authors", []))
        abstract = item.get("abstract")
        pdf_link = item.get("related_material", {}).get("pdf")

        cursor.execute(
            """
            INSERT INTO papers (year, title, authors, abstract, pdf_link)
            VALUES (?, ?, ?, ?, ?)
            """,
            (year, title, authors, abstract, pdf_link),
        )

    conn.commit()
    conn.close()

def process_json_files(directory):
    """Processes all JSON files in the given directory."""

    create_database()

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            year = int(re.findall(r"\d+", filename)[0])

            with open(filepath, "r") as f:
                data = json.load(f)
                insert_data(data, year)

if __name__ == "__main__":
    process_json_files("/Users/moji/Projects/bitter-lesson-cvpr/data/thecvf/cvpr/")
