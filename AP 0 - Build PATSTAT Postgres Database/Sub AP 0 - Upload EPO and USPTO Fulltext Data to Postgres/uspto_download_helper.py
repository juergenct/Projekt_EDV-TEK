import os
import requests
from multiprocessing import Pool
from tqdm import tqdm

# Directory to save the downloaded files
download_directory = "/mnt/hdd01/patentsview/Fulltext Data/Raw Brief Summary"
os.makedirs(download_directory, exist_ok=True)

def download_file(url):
    """ Function to download a single file """
    local_filename = os.path.join(download_directory, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # This will raise an exception if the request failed
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded {local_filename}")

def download_files_concurrently(urls):
    """ Function to download files using multiple processes """
    with Pool(processes=18) as pool:
        list(tqdm(pool.imap_unordered(download_file, urls), total=len(urls), desc="Downloading files"))

if __name__ == "__main__":
    urls = [
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2024.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2023.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2022.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2021.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2020.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2019.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2018.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2017.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2016.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2015.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2014.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2013.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2012.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2011.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2010.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2009.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2008.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2007.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2006.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2005.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2004.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2003.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2002.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2001.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2000.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1999.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1998.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1997.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1996.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1995.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1994.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1993.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1992.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1991.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1990.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1989.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1988.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1987.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1986.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1985.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1984.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1983.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1982.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1981.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1980.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1979.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1978.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1977.tsv.zip',
        'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1976.tsv.zip'
    ]

    download_files_concurrently(urls)
