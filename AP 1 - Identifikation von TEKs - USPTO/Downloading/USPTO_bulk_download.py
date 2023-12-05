import time
import requests
import re
import zipfile
import io
import os
from bs4 import BeautifulSoup
from tqdm import tqdm

urls = [
    ''
    # Add more urls as needed
]

# Set the number of requests per minute
requests_per_minute = 10

# Calculate the delay between requests
delay = 60 / requests_per_minute

# Set a list of links to skip
links_to_skip = [
    # Add more links to skip as needed
]

# Set the paths where the downloads should be saved
download_path = [
    "/Users/jurgenthiesen/Documents/Full Text Bulk Data USPTO 02_23/"
    # Add more paths as needed
]

# Keep track of the time of the last request
last_request = 0

# Loop through all the urls
for url in urls:
    # Make sure that we don't exceed the number of requests per minute
    current_time = time.time()
    time_since_last_request = current_time - last_request
    if time_since_last_request < delay:
        time.sleep(delay - time_since_last_request)
    last_request = time.time()

    # Make a request to the website
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, "html.parser")

        # Find all the links on the page
        links = soup.find_all("a")

        # Loop through all the links
        for link in links:
            # Get the href attribute of the link
            href = link.get("href")

            # Check if the link ends with ".zip"
            if href and href.endswith(".zip") and href not in links_to_skip:

                # Download the file
                print(href)
                zip_file = requests.get(url + href)
                # Set file_path to correct entry in download_path list, based on the url
                file_path = download_path[urls.index(url)] + href.split("/")[-1]
                open(file_path, "wb").write(zip_file.content)

                # Unpack the zip file
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(download_path[urls.index(url)])
                # Delete the zip file in download_path
                os.remove(file_path)
                # Delete files ending with "lst.txt" and "rpt.txt"
                for file in os.listdir(download_path[urls.index(url)]):
                    if file.endswith(".sgm"):
                        os.remove(download_path[urls.index(url)] + file)
    elif response.status_code == 400:
        print("Bad request: " + url)
    elif response.status_code == 401:
        print("Unauthorized: " + url)
    elif response.status_code == 403:
        print("Access denied: " + url)
    elif response.status_code == 404:
        print("The page does not exist: " + url)
    elif response.status_code == 500:
        print("Internal server error: " + url)
    elif response.status_code == 503:
        print("Service unavailable: " + url)
    elif response.status_code == 504:
        print("Gateway timeout: " + url)
    else:
        print("Failed to retrieve the webpage: " + url)