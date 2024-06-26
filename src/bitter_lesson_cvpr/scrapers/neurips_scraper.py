import requests
from bs4 import BeautifulSoup
import sqlite3
from tqdm import tqdm

def create_database(db_name="dbs/neurips_papers.db"):
    """Creates a SQLite database with the specified name and table structure."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER NOT NULL,
            title TEXT NOT NULL,
            authors TEXT,
            abstract TEXT
        )
    """)

    conn.commit()
    conn.close()

def scrape_neurips_papers(year, db_name="dbs/neurips_papers.db"):
    """Scrapes Neurips papers for a given year and stores them in the database."""

    url = f"https://proceedings.neurips.cc/paper/{year}"
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    paper_links = soup.find_all('a', href=lambda href: href and "/paper_files/paper/" in href)

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    for link in tqdm(paper_links, desc=f"Scraping year {year}"):
        paper_url = f"https://proceedings.neurips.cc{link['href']}"
        paper_response = requests.get(paper_url)
        paper_response.raise_for_status()

        paper_soup = BeautifulSoup(paper_response.content, 'html.parser')

        title = paper_soup.find('h4').text.strip()

        # Extract authors correctly:
        authors = ", ".join([author['content'] for author in paper_soup.find_all('meta', {'name': 'citation_author'})])

        abstract_element = paper_soup.find('h4', text='Abstract').find_next_sibling('p')
        abstract = abstract_element.text.strip() if abstract_element else "N/A"


        cursor.execute(
            """
            INSERT INTO papers (year, title, authors, abstract)
            VALUES (?, ?, ?, ?)
            """,
            (year, title, authors, abstract),
        )

    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_database()

    for year in range(1987, 2024):  # Scrape from 1987 to 2023
        scrape_neurips_papers(year)
