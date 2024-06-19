import sqlite3
import bibtexparser

def create_database(db_name="dbs/icml_papers.db"):
    """Creates a SQLite database with the specified name and table structure."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT,
            year INTEGER,
            abstract TEXT,
            publisher TEXT,
            booktitle TEXT,
            articleno TEXT,
            numpages TEXT,
            location TEXT,
            series TEXT
        )
    """)

    conn.commit()
    conn.close()

def insert_data(data, db_name="dbs/icml_papers.db"):
    """Inserts data into the SQLite database."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    for item in data.entries:
        cursor.execute(
            """
            INSERT INTO papers (title, author, year, abstract, publisher, booktitle, articleno, numpages, location, series)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.get("title", ""),
                item.get("author", ""),
                int(item.get("year", 0)),  # Convert year to integer
                item.get("abstract", ""),
                item.get("publisher", ""),
                item.get("booktitle", ""),
                item.get("articleno", ""),
                item.get("numpages", ""),
                item.get("location", ""),
                item.get("series", ""),
            ),
        )

    conn.commit()
    conn.close()

def process_bib_file(bib_file_path):
    """Processes the BibTeX file and inserts data into the database."""
    with open(bib_file_path, 'r') as bibtex_file:
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        bib_database = bibtexparser.load(bibtex_file, parser=parser)

    create_database()
    insert_data(bib_database)

if __name__ == "__main__":
    bib_file_path = "data/icml_proceedings_acm.bib"  # Update with the actual path to your BibTeX file
    process_bib_file(bib_file_path)
