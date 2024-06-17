import asyncio
import csv
import json
import logging
import os
from datetime import datetime

import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()  # Load environment variables from .env file

S2_API_KEY = os.getenv("S2_API_KEY")

# Configure logging
logging.basicConfig(filename="error.log", level=logging.ERROR)


async def fetch_paper_data(session, title):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "fields": "title,authors,abstract,year,publicationVenue,paperId",
    }
    headers = {
        "X-API-KEY": S2_API_KEY,
    }
    try:
        async with session.get(url, params=params, headers=headers) as response:
            data = await response.json()
            return data["data"]
    except KeyError as e:
        logging.error(f"Error fetching data for title: {title}")
        logging.error(str(e))
        return None


async def main():
    with open("data/cvpr_2024_papers.json", "r") as file:
        papers = json.load(file)

    paper_titles = [paper["title"] for paper in papers]

    # Clear the abstract JSON file
    with open("data/cvpr_2024_abstract.json", "w") as file:
        json.dump([], file)

    # Open the CSV file for logging
    with open("log.csv", "w", newline="") as csvfile:
        fieldnames = ["timestamp", "title", "status"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        async with aiohttp.ClientSession() as session:
            for title in tqdm(paper_titles, desc="Fetching paper data"):
                paper_data = await fetch_paper_data(session, title)
                if paper_data is not None:
                    # Append the paper data to the abstract JSON file
                    with open("data/cvpr_2024_abstract.json", "a") as file:
                        json.dump(paper_data, file)
                        file.write("\n")
                    writer.writerow(
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "title": title,
                            "status": "Success",
                        }
                    )
                else:
                    writer.writerow(
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "title": title,
                            "status": "Error",
                        }
                    )
                await asyncio.sleep(1)  # Delay of 1 second between each request


asyncio.run(main())
