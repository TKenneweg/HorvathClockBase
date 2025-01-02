import pickle
import os
import math

def main():
    # pkl_path = "./data/GSE27097/GSM666344.pkl"
    pkl_path = "./data/GSE41037/GSM1007129.pkl"

    if not os.path.isfile(pkl_path):
        print(f"Pickle file not found at {pkl_path}")
        return

    # Load the dictionary
    with open(pkl_path, "rb") as f:
        methylation_dict = pickle.load(f)

    # Print the number of sites
    print(f"Loaded methylation dict from {pkl_path}")
    print(f"Number of methylation sites: {len(methylation_dict)}")

    # Show just a handful of entries (e.g., 5)
    print("Example entries:")
    nan_count = 0
    for key in methylation_dict.keys():
        if isinstance(methylation_dict[key], float) and math.isnan(methylation_dict[key]):
            nan_count += 1
    for i, (site, value) in enumerate(methylation_dict.items()):
        print(f"  {site}: {value}")
        if i >= 10:
            break

    print(f"Number of NaN values: {nan_count}")
    print(methylation_dict["age"])

if __name__ == "__main__":
    main()
