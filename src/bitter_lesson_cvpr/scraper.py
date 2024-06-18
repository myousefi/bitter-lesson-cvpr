import requests
from bs4 import BeautifulSoup
import json

url = "https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

papers = []

table = soup.find("table")
rows = table.find_all("tr")

for row in rows[1:]:  # Skip the header row
    cells = row.find_all("td")
    if len(cells) == 3:
        title_info = cells[0].text.strip().split('\n')
        title = title_info[0].strip()
        session = title_info[1].strip() if len(title_info) > 1 else ""
        authors_str = title_info[-1].strip() if len(title_info) > 2 else ""
        authors = [author.strip() for author in authors_str.split('\u00b7')]
        
        paper = {
            "title": title,
            "session": session,
            "authors": authors,
            "location": cells[2].text.strip()
        }
        papers.append(paper)

with open("data/cvpr_2024_papers.json", "w") as file:
    json.dump(papers, file, indent=2)

print("CVPR 2024 paper information extracted and saved to cvpr_2024_papers.json")
