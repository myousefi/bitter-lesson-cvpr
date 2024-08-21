import requests
from bs4 import BeautifulSoup
import sqlite3
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

S2_API_KEY = os.getenv("S2_API_KEY")

def create_database(db_name="dbs/cvpr_papers.db"):
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

def extract_paper_info(session, paper_entry):
    """Extracts paper information from a given paper entry."""
    # Extract title
    title_tag = paper_entry.find('span', class_='title')
    title = title_tag.text.strip() if title_tag else "N/A"

    # Extract authors
    authors_tags = paper_entry.find_all('span', itemprop='author')
    authors = [author_tag.text.strip() for author_tag in authors_tags]

    # Extract DOI link
    doi_tag = paper_entry.find('a', href=lambda href: href and 'doi.org' in href)
    doi_link = doi_tag['href'] if doi_tag else None

    abstract = None
    if doi_link:
        abstract = get_abstract_from_semanticscholar(session, doi_link)

    return {
        'title': title,
        'authors': authors,
        'abstract': abstract,
        'doi_link': doi_link
    }

def get_abstract_from_semanticscholar(session, doi_link):
    """Gets the abstract from Semantic Scholar API using the DOI link."""
    doi = '/'.join(doi_link.split('/')[-2:])  # Extract DOI from the link
    url = f'https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract'
    headers = {
        'x-api-key': S2_API_KEY
    }
    try:
        response = session.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        abstract = data.get('abstract', 'Abstract not found')
    except requests.exceptions.RequestException as e:
        print(f"Error fetching abstract for DOI {doi_link}: {e}")
        abstract = 'Abstract not found'
    return abstract

def insert_paper_data(paper_info, year, db_name="dbs/cvpr_papers.db"):
    """Inserts paper data into the SQLite database."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    title = paper_info.get("title")
    authors = ", ".join(paper_info.get("authors", []))
    abstract = paper_info.get("abstract")
    pdf_link = paper_info.get("doi_link")

    cursor.execute(
        """
        INSERT INTO papers (year, title, authors, abstract, pdf_link)
        VALUES (?, ?, ?, ?, ?)
        """,
        (year, title, authors, abstract, pdf_link),
    )

    conn.commit()
    conn.close()

def scrape_cvpr_papers(year, base_url="https://dblp.org/db/conf/cvpr/", url=None):
    """Scrapes CVPR papers from DBLP for a given year and stores them in the database."""
    if url is None:
        url = f"{base_url}cvpr{year}.html"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    
    session = requests.Session()
    response = session.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    paper_lists = soup.find_all('ul', class_='publ-list')
    for paper_list in paper_lists:
        paper_entries = paper_list.find_all('li', class_='entry inproceedings')

        for paper_entry in tqdm(paper_entries, desc=f"Scraping CVPR {year} papers"):
            paper_info = extract_paper_info(session, paper_entry)
            insert_paper_data(paper_info, year)
            time.sleep(1)  # Add delay to avoid rate limiting

if __name__ == '__main__':
    create_database()
    # for year in range(2004, 2011):
    for year, url in [(2004, "https://dblp.org/db/conf/cvpr/cvpr2004-1.html"), (2004, "https://dblp.org/db/conf/cvpr/cvpr2004-2.html"), (2005, "https://dblp.org/db/conf/cvpr/cvpr2005-1.html"), (2005, "https://dblp.org/db/conf/cvpr/cvpr2005-2.html"), (2006, "https://dblp.org/db/conf/cvpr/cvpr2006-1.html"), (2006, "https://dblp.org/db/conf/cvpr/cvpr2006-2.html")]:
        print(f"Scraping papers for the year {year}...")
        scrape_cvpr_papers(year, url=url)
