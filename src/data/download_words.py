import io
import os
import requests
import shutil
import zipfile


URL_WORDS = "https://gist.github.com/MarvinJWendt/2f4f4154b8ae218600eb091a5706b5f4/archive/36b70dd6be330aa61cd4d4cdfda6234dcb0b8784.zip"


if __name__ == "__main__":
    response = requests.get(URL_WORDS)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extract_path = "/tmp/german_words"
    z.extractall(extract_path)
    filepath = os.path.join(extract_path, os.listdir(extract_path)[0], "wordlist-german.txt")
    shutil.copy(filepath, os.path.join(os.path.dirname(__file__), "words.txt"))


