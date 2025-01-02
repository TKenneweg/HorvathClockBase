import os
import logging
import pickle
import re

import GEOparse
import sys
from config import *
import numpy as np

# Uncomment to suppress verbose DEBUG messages from GEOparse:
# logging.getLogger("GEOparse").setLevel(logging.ERROR)

def parse_age(gsm,divisor=1):
    """
    Attempt to extract an age value from the  sample metadata.
    This example looks at characteristics_ch1 lines for something like 'age: 57'.
    Adjust to match your exact metadata format.
    """
    characteristics = gsm.metadata.get("characteristics_ch1", [])
    if not isinstance(characteristics, list):
        characteristics = [characteristics]
    
    for entry in characteristics:
        # We look for a pattern 'age: 57' or 'age=57' or 'Age: 57', etc.
        match = re.search(r"age\D+(\d+)", entry, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))/divisor
    
    # If no pattern found, return None
    return None


def main():
    for series_id in SERIES_NAMES:
        # Construct the path to the .soft.gz file
        filepath = DATA_FOLDER + "/" + f"{series_id}_family.soft.gz"
        
        # 1. Load GEO data using GEOparse
        print(f"Loading {series_id} from file: {filepath}")
        gse = GEOparse.get_GEO(filepath=filepath)
        
        # Create a subfolder for this series (if it doesn't exist)
        series_subfolder = DATA_FOLDER + "/" + series_id
        os.makedirs(series_subfolder, exist_ok=True)
        
        # 2. For each GSM, check platform, extract methylation data, and add age
        for gsm_name, gsm in gse.gsms.items():
            # Check platform ID
            # gsm.metadata["platform_id"] is typically a list; we want the first item
            platform_list = gsm.metadata.get("platform_id", [])
            if not platform_list or platform_list[0] != "GPL8490":
                # Skip this sample if it's not from platform GPL8490
                continue
            
            # Extract methylation table
            df = gsm.table

            # Decide which column to use for the methylation value
            if "VALUE" in df.columns:
                col = "VALUE"
            elif "AVG_Beta" in df.columns:
                col = "AVG_Beta"
            else:
                # fallback: pick the last column if the common ones are missing
                col = df.columns[-1]
            
            if "ID_REF" not in df.columns:
                print(f"Warning: 'ID_REF' not found in {gsm_name} table. Skipping.")
                continue
            
            # Set 'ID_REF' as the index for easier conversion to dict
            df = df.set_index("ID_REF")
            methylation_dict = df[col].to_dict()
            for key, value in methylation_dict.items():
                methylation_dict[key] = float(value) if value is not None and np.isfinite(value) and value >= 0 and value <=1 else 0.5
            
            # Parse and add age
            age_val = parse_age(gsm) if series_id != "GSE27097" else parse_age(gsm,divisor=12)
            if age_val is None or np.isnan(age_val):
                print(f"Warning: No age found for {gsm_name}. Skipping.")
                continue

            methylation_dict["age"] = age_val
            
            # 3. Save the dictionary as a pickle file
            pkl_filename = os.path.join(series_subfolder, f"{gsm_name}.pkl")
            with open(pkl_filename, "wb") as f:
                pickle.dump(methylation_dict, f)
            
            print(f"Saved {gsm_name} data -> {pkl_filename} | AGE={age_val}")


if __name__ == "__main__":
    main()
