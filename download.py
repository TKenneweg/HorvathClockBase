import GEOparse
from config import *

# series = ["GSE15745"]
# series = ["GSE41037","GSE15745"]

for s in SERIES_NAMES:
    gse = GEOparse.get_GEO(geo=s, destdir="./data")
    # gse = GEOparse.get_GEO(filepath="./data/GSE15745_family.soft.gz")

# print("done")
# print(gse.gpls)