import json
import sqlite3

def migrate_json_to_sqlite(json_file, db_file):
    """
    Migrates data from a JSON file to a SQLite database.

    Args:
        json_file (str): Path to the JSON file.
        db_file (str): Path to the SQLite database file.
    """

    with open(json_file, 'r') as f:
        data = json.load(f)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create tables (if they don't exist)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS PublicationVenues (
            publicationVenueId TEXT PRIMARY KEY,
            name TEXT,
            type TEXT,
            issn TEXT,
            url TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Papers (
            paperId TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            year INTEGER,
            publicationVenueId TEXT,
            FOREIGN KEY (publicationVenueId) REFERENCES PublicationVenues(publicationVenueId)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Authors (
            authorId TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS PaperAuthors (
            paperId TEXT,
            authorId TEXT,
            PRIMARY KEY (paperId, authorId),
            FOREIGN KEY (paperId) REFERENCES Papers(paperId),
            FOREIGN KEY (authorId) REFERENCES Authors(authorId)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS AlternateVenueNames (
            venueNameId INTEGER PRIMARY KEY AUTOINCREMENT,
            publicationVenueId TEXT,
            alternateName TEXT,
            FOREIGN KEY (publicationVenueId) REFERENCES PublicationVenues(publicationVenueId)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS AlternateVenueURLs (
            venueURLId INTEGER PRIMARY KEY AUTOINCREMENT,
            publicationVenueId TEXT,
            alternateURL TEXT,
            FOREIGN KEY (publicationVenueId) REFERENCES PublicationVenues(publicationVenueId)
        )
    """)

    # Insert data
    for item in data:
        # Publication Venues
        publication_venue = item.get('publicationVenue', {})
        cursor.execute(
            """
            INSERT OR IGNORE INTO PublicationVenues (publicationVenueId, name, type, issn, url)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                publication_venue.get('id'),
                publication_venue.get('name'),
                publication_venue.get('type'),
                publication_venue.get('issn'),
                publication_venue.get('url')
            )
        )

        # Alternate Venue Names
        for alt_name in publication_venue.get('alternate_names', []):
            cursor.execute(
                """
                INSERT OR IGNORE INTO AlternateVenueNames (publicationVenueId, alternateName)
                VALUES (?, ?)
                """,
                (publication_venue.get('id'), alt_name)
            )

        # Alternate Venue URLs
        for alt_url in publication_venue.get('alternate_urls', []):
            cursor.execute(
                """
                INSERT OR IGNORE INTO AlternateVenueURLs (publicationVenueId, alternateURL)
                VALUES (?, ?)
                """,
                (publication_venue.get('id'), alt_url)
            )

        # Papers
        cursor.execute(
            """
            INSERT OR IGNORE INTO Papers (paperId, title, abstract, year, publicationVenueId)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                item.get('paperId'),
                item.get('title'),
                item.get('abstract'),
                item.get('year'),
                publication_venue.get('id')
            )
        )

        # Authors
        for author in item.get('authors', []):
            cursor.execute(
                """
                INSERT OR IGNORE INTO Authors (authorId, name)
                VALUES (?, ?)
                """,
                (author.get('authorId'), author.get('name'))
            )

            # PaperAuthors
            cursor.execute(
                """
                INSERT OR IGNORE INTO PaperAuthors (paperId, authorId)
                VALUES (?, ?)
                """,
                (item.get('paperId'), author.get('authorId'))
            )

    conn.commit()
    conn.close()

if __name__ == "__main__":
    json_file = "/Users/moji/Projects/bitter-lesson-cvpr/data/cvpr_2024_abstract.json"  # Replace with your JSON file path
    db_file = "/Users/moji/Projects/bitter-lesson-cvpr/data/database.db"  # Replace with your desired database file name
    migrate_json_to_sqlite(json_file, db_file)
    print(f"Data migrated from '{json_file}' to '{db_file}' successfully!")
