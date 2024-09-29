from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import os
import time


# Function to download a file from a URL
def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    filename = os.path.join(dest_folder, url.split('/')[-1])
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download {url}")


# Set up Selenium WebDriver (make sure you have ChromeDriver installed and in your PATH)
driver = webdriver.Chrome()
actions = ActionChains(driver)

# URL of the website
url = "https://www.ema.gov.sg/resources/statistics/half-hourly-system-demand-data"
driver.get(url)

# Wait for the page to load and the Vue.js component to render
wait = WebDriverWait(driver, 10)
dropdown = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'custom-dropdown.year-dropdown')))
dropdown.click()
time.sleep(1)


# Function to get all .xlsx links from the current page
def get_xls_links():
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    links = soup.find_all('a', href=True)
    return [link['href'] for link in links if link['href'].endswith('.xls')]


def download_xls_links():
    # Get all .xlsx links from the current page
    xlsx_links = get_xls_links()

    # Download each .xlsx file
    for link in xlsx_links:
        full_url = f"https://www.ema.gov.sg{link}"  # Ensure full URL
        print(f"Downloading: {full_url}")
        download_file(full_url, 'C:/Users/stucws/Documents/astar/data/dataset/EMA dataset')


# Iterate through each option in the year dropdown
year_options = dropdown.find_elements(By.TAG_NAME, 'li')
print(year_options)

for i, year in enumerate(year_options):
    if i > 0:
        driver.execute_script("arguments[0].click();", dropdown)
        time.sleep(1)

    driver.execute_script("arguments[0].click();", year)
    time.sleep(1)

    download_xls_links()

    page_numbers = driver.find_elements(By.CLASS_NAME, 'cmp-media__tag')
    last_page = page_numbers[-1].text

    # Find and click the 'Next' button
    for pages in range(int(last_page) - 1):
        next_button = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, 'page__btn.page-item.next-pagination')))
        next_button.click()
        time.sleep(1)

        download_xls_links()

driver.quit()
print("All files downloaded.")
