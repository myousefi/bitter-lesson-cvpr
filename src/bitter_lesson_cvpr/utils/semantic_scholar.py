from calendar import c
import requests
from dotenv import load_dotenv
import os
import sqlite3
import json

from tqdm import tqdm
import time

load_dotenv()  # Load environment variables from .env file

S2_API_KEY = os.getenv("S2_API_KEY")

# Define the endpoint URL
url = "https://api.semanticscholar.org/graph/v1/paper/search"

# Database connection
conn = sqlite3.connect('dbs/cvpr_papers.db')
cursor = conn.cursor()

# Create the new table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS semantic_scholar_data (
    paper_id INTEGER PRIMARY KEY,
    paperId TEXT,
    DBLP TEXT,
    MAG TEXT,
    DOI TEXT,
    CorpusId TEXT,
    publicationVenue TEXT,
    title TEXT,
    venue TEXT,
    citationCount INTEGER,
    influentialCitationCount INTEGER,
    publicationDate TEXT,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
)
""")

# Get papers with bitter_lesson_scores
cursor.execute("""
SELECT id, title, year
FROM papers
WHERE Abstract IS NULL
""")
papers = cursor.fetchall()

for paper_id, title, year in tqdm(papers, desc="Fetching Semantic Scholar data"):
    # Define the query parameters
    query_params = {
        'query': title,
        'limit': 1,
        'year': f'{year-2}-',  # Search from the paper's year onwards
        'fields': 'title,externalIds,citationCount,influentialCitationCount,venue,publicationVenue,publicationDate,abstract'
    }

    headers = {
            "X-API-KEY": S2_API_KEY,
    }

    # Make the request with the specified parameters
    response = requests.get(url, params=query_params, headers=headers)

    if response.status_code == 200:
        data = response.json().get('data', [{}])[0]  # Get the first result or an empty dict

        # Extract data, handling potential KeyError exceptions
        paperId = data.get('paperId')
        external_ids = data.get('externalIds', {})
        dblp = external_ids.get('DBLP')
        mag = external_ids.get('MAG')
        doi = external_ids.get('DOI')
        corpus_id = external_ids.get('CorpusId')

        publication_venue = data.get('publicationVenue')
        if isinstance(publication_venue, dict):
            # Assuming you want to store the entire dictionary as JSON
            publication_venue = json.dumps(publication_venue) 

        title = data.get('title')
        venue = data.get('venue')
        citation_count = data.get('citationCount')
        influential_citation_count = data.get('influentialCitationCount')
        publication_date = data.get('publicationDate')
        abstract = data.get('abstract')

        # Insert data into the database
        cursor.execute("""
        INSERT OR IGNORE INTO semantic_scholar_data (
            paper_id, paperId, DBLP, MAG, DOI, CorpusId, publicationVenue, 
            title, venue, citationCount, influentialCitationCount, publicationDate
        ) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (paper_id, paperId, dblp, mag, doi, corpus_id, publication_venue,
              title, venue, citation_count, influential_citation_count, publication_date))

        if abstract:
            cursor.execute("""
            UPDATE papers
            SET abstract = ?
            WHERE id = ?
            """, (abstract, paper_id))
        else:
            cursor.execute("""
            UPDATE papers
            SET abstract = ?
            WHERE id = ?
            """, ("NOT AVAILABLE", paper_id))

        conn.commit()
    else:
        print(f"Error fetching data for paper: {title} (ID: {paper_id})")

        cursor.execute("""
            UPDATE papers
            SET abstract = ?
            WHERE id = ?
            """, ("NOT AVAILABLE", paper_id))

    time.sleep(0.5)  # Add a 0.5-second delay between requests

conn.close()
