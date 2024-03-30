import requests
import zipfile
import io
import os

def download_and_extract_zip(url):
    response = requests.get(url)
    if response.status_code == 200:
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        zip_file.extractall()
        zip_file.close()
        print("Zip file downloaded and extracted successfully!")
    else:
        print("Failed to download the zip file.")

# Usage example
repo_url = 'https://github.com/MWiechmann/enron_spam_data/enron_spam_data.zip'
download_and_extract_zip(repo_url)
