import requests
from bs4 import BeautifulSoup
import json
import asyncio
import aiohttp
from tqdm.asyncio import tqdm
import time
from aiohttp.client_exceptions import ClientOSError, ServerDisconnectedError

async def extract_paper_info(session, paper_url, retries=3, delay=1):
    """Extracts paper information from a given paper URL with retries."""
    for attempt in range(retries):
        try:
            async with session.get(paper_url) as response:
                response.raise_for_status()
                soup = BeautifulSoup(await response.text(), 'html.parser')

                # Extract title
                title_div = soup.find('div', id='papertitle')
                title = title_div.text.strip() if title_div else "N/A"

                # Extract authors
                authors_div = soup.find('div', id='authors')
                authors_text = authors_div.text.strip() if authors_div else "N/A"
                authors = [author.strip() for author in authors_text.split(';')[0].split(',')]

                # Extract abstract
                abstract_div = soup.find('div', id='abstract')
                abstract = abstract_div.text.strip() if abstract_div else "N/A"

                # Extract related material (PDF, supplementary, bibtex)
                related_material = {}
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href:
                        if href.endswith('.pdf'):
                            if 'paper.pdf' in href:
                                related_material['pdf'] = href
                            else:
                                related_material['supp'] = href
                        elif 'arxiv.org' in href:
                            related_material['arxiv'] = href

                return {
                    'title': title,
                    'authors': authors,
                    'abstract': abstract,
                    'related_material': related_material
                }

        except (ClientOSError, ServerDisconnectedError, aiohttp.ClientError) as e:
            if attempt < retries - 1:
                print(f"Error: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                print(f"Failed to fetch {paper_url} after {retries} attempts.")
                return None  # Return None to indicate failure

async def write_to_json(queue, year):
    """Writes paper data from the queue to the JSON file."""
    with open(f'data/thecvf/cvpr/{year}_papers.json', 'a') as f:
        while True:
            paper_info = await queue.get()
            if paper_info is None:  # Sentinel value to stop the writer
                break
            json.dump(paper_info, f, indent=4)
            f.write('\n')
            queue.task_done()

async def scrape_cvpr_papers(base_url, year, day="all",batch_size=10, delay=1):
    """Scrapes CVPR papers with batching and rate limiting."""
    all_papers = []
    paper_queue = asyncio.Queue()

    async with aiohttp.ClientSession() as session:
        response = await session.get(f"{base_url}/CVPR{year}?day={day}")
        response.raise_for_status()

        soup = BeautifulSoup(await response.text(), 'html.parser')
        paper_links = soup.find_all('dt', class_='ptitle')

        # Start the writer task
        writer_task = asyncio.create_task(write_to_json(paper_queue, year))

        # Process links in batches with rate limiting
        for i in range(0, len(paper_links), batch_size):
            batch_links = paper_links[i:i + batch_size]
            tasks = [extract_paper_info(session, f"{base_url}{link.find('a')['href']}") 
             for link in batch_links]
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                paper_info = await coro
                if paper_info:  # Only add successful results
                    all_papers.append(paper_info)
                    await paper_queue.put(paper_info)
            await asyncio.sleep(delay)  # Wait between batches

        # Signal the writer to stop after processing all items
        await paper_queue.put(None)
        await writer_task

    return all_papers

if __name__ == '__main__':
    for year, day in [
        (2018,"2018-06-19"),
        (2018,"2018-06-20"),
        (2018,"2018-06-21"),
        (2019,"2019-06-18"),
        (2019,"2019-06-19"),
        (2019,"2019-06-20"),
        (2020,"2020-06-16"),
        (2020,"2020-06-17"),
        (2020,"2020-06-18"),
    ]:
        base_url = f"https://openaccess.thecvf.com/"
        asyncio.run(scrape_cvpr_papers(base_url, year, day))
