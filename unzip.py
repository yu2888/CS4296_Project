import zipfile
import os

def unzip_file(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_path))
    print("Zip file extracted successfully!")

# Usage example
zip_file_path = 'enron_spam_data.zip'  # Replace with the actual path to your zip file
unzip_file(zip_file_path)