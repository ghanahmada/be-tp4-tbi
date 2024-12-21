import os
import gdown
import zipfile
import logging
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_dir(dir_name: str) -> bool:
    return os.path.isdir(dir_name)

def download_data(url, filename, dir_name: str = "experiments") -> None:
    if not check_dir(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)

    logging.info("Downloading data....")
    gdown.download(url, quiet=False)
    logging.info(f"Downloading to: {os.getcwd()}")

    logging.info("Extracting zip file....")
    with zipfile.ZipFile(f"{filename}.zip", 'r') as zip_ref:
        zip_ref.extractall(filename)
    logging.info(f"Extracted files to: {filename}")

    os.remove(f"{filename}.zip")
    logging.info("Removed zip file after extraction.")

    os.chdir("..")
