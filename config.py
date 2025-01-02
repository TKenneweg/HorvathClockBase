SERIES_NAMES = ["GSE41037", "GSE15745", "GSE27317", "GSE27097", "GSE34035"]
# SERIES_NAMES = ["GSE41037", "GSE15745", "GSE27317",  "GSE34035"]
# SERIES_NAMES = [ "GSE27097"]
# SERIES_NAMES = [ "GSE41037"]
# GSE27097

DATA_FOLDER = "./data"

BATCH_SIZE = 32
LR = 2e-4
NUM_EPOCHS = 200
TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42
NUM_PROBES = 27578
#we take datasets:

# GSE41037 with n = 715, median = 2.8 
# GSE15745 with uncertain n = 506 mediam somewhere arond 3-5
# GSE27317 with n= 216 median = 0.1
# GSE27097 with n = 386 median = 1.7
# GSE34035 with n= 131 median = 2.7


