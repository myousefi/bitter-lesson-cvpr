from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.service import Service # Import Service
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

driver.get("https://www.paperdigest.org/topic/?topic=cvpr&year=2024")

# Wait for the page to load
WebDriverWait(driver, 10).until(EC.presence_of_element_located(locator=(By.ID, "example")))

# Get all the abstracts
abstracts_table = driver.find_element(By.ID, "abstracts-table")
abstracts_html = abstracts_table.get_attribute("innerHTML")

# Parse the HTML content
soup = BeautifulSoup(abstracts_html, "html.parser")

# Find all the abstracts
all_abstracts = soup.find_all("div", class_="abstract")

# Loop through each abstract
for abstract in all_abstracts:
    # Click the abstract to expand it
    abstract.find("button").click()

    # Wait for the content to expand
    WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CLASS_NAME, "abstract-content")))

    # Get the expanded abstract content
    expanded_content = driver.find_element(By.CLASS_NAME, "abstract-content").text

    # Process the abstract and expanded content here
    print(f"Abstract: {abstract.text}")
    print(f"Expanded Content: {expanded_content}\n")

# Close the browser
driver.quit()