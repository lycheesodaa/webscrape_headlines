from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import NoSuchElementException, TimeoutException
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
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
        print("Response:", response.status_code)
        print(f"Failed to download {url}")


# Set up Selenium WebDriver (make sure you have ChromeDriver installed and in your PATH)
driver = webdriver.Chrome()
actions = ActionChains(driver)

for year in range(2014, 2025):
    for month in range(1, 13):
        # URL of the website
        url = f"https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{year}{month:02d}_NSW1.csv"
        driver.get(url)

exit()
#
# print(driver.page_source)
#
# def find_select_elements():
#     # Method 1: Wait and find by class name
#     try:
#         selects = WebDriverWait(driver,  5).until(
#             EC.presence_of_all_elements_located((By.CLASS_NAME, "visualisation-select"))
#         )
#         return selects[-2:]  # Return the last two elements
#     except TimeoutException:
#         print("Timeout waiting for elements by class name")
#
#     # Method 4: Use JavaScript to find elements
#     script = """
#         return Array.from(document.querySelectorAll('select.visualisation-select'))
#             .filter(el => el.id === 'year-select' || el.id === 'month-select');
#         """
#     selects = driver.execute_script(script)
#     if selects:
#         return selects
#     else:
#         print("Failed to find select elements w JS")
#
#     # method2: look for iframes
#     iframes = driver.find_elements(By.TAG_NAME, "iframe")
#     for iframe in iframes:
#         driver.switch_to.frame(iframe)
#         try:
#             selects = driver.find_elements(By.CLASS_NAME, "visualisation-select")
#             if len(selects) >= 2:
#                 return selects[-2:]
#         except NoSuchElementException:
#             pass
#         finally:
#             driver.switch_to.default_content()
#
#     return []  # If no elements found
#
# # Wait for the page to load and the Vue.js component to render
# selects = find_select_elements()
# year_dropdown = selects[0]
# month_dropdown = selects[1]
# time.sleep(1)
#
# try:
#     # Create a Select object
#     year_select = Select(year_dropdown)
#
#     # Iterate through all options
#     for year_option in year_select.options:
#         if int(year_option.text) < 2014:
#             continue
#
#         print(f"Value: {year_option.get_attribute('value')}, Text: {year_option.text}")
#
#         # Optionally, select each option
#         year_select.select_by_value(year_option.get_attribute('value'))
#
#         month_select = Select(month_dropdown)
#
#         for month_option in month_select.options:
#             download_link = driver.find_elements(by=By.CLASS_NAME, value='visualisation-button')
#             assert len(download_link) == 2
#
#             download_file(download_link[1].get_attribute('href'), "C:\\Users\\stucws\\Documents\\astar\\sentiment-analysis\\webscrape_headlines\\external_data\\aus_demand_data")
#
# except Exception as e:
#     print(f"An error occurred: {e}")
#
# driver.quit()
# print("All files downloaded.")
