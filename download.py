import GEOparse
from config import *
import os


for s in SERIES_NAMES:
    file_path = f"{DATA_FOLDER}/{s}_family.soft.gz"
    if not os.path.exists(file_path):
        gse = GEOparse.get_GEO(geo=s, destdir=DATA_FOLDER)
    else:
        print(f"File {file_path} already exists. Skipping download.")
