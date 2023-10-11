import io
import os
import requests
import zipfile


URL_TEXTS = "https://figshare.com/ndownloader/files/7320866"


if __name__ == "__main__":
    response = requests.get(URL_TEXTS)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extract_path = os.path.join(script_dir, "download")
    z.extractall(extract_path)
