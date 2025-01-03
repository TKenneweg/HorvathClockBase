import os
import logging
import pickle
import re

import GEOparse
import sys
from config import *
import numpy as np

def get_age(gsm,divisor=1):
    characteristics = gsm.metadata.get("characteristics_ch1", [])
    if not isinstance(characteristics, list):
        characteristics = [characteristics]
    
    for entry in characteristics:
        # We look for a pattern 'age: 57' or 'age=57' or 'Age: 57', etc.
        match = re.search(r"age\D+(\d+)", entry, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))/divisor
    return None


def main():
    for series_id in SERIES_NAMES:
        filepath = DATA_FOLDER + "/" + f"{series_id}_family.soft.gz"
        print(f"Loading {series_id} from file: {filepath}")
        gse = GEOparse.get_GEO(filepath=filepath)
        
        series_subfolder = DATA_FOLDER + "/" + series_id
        os.makedirs(series_subfolder, exist_ok=True)
        
        for gsm_name, gsm in gse.gsms.items():
            #check platform
            platform_list = gsm.metadata.get("platform_id", [])
            if not platform_list or platform_list[0] != "GPL8490":
                continue
            
            df = gsm.table
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
            age_val = get_age(gsm) if series_id != "GSE27097" else get_age(gsm,divisor=12)
            if age_val is None or np.isnan(age_val):
                print(f"Warning: No age found for {gsm_name}. Skipping.")
                continue

            methylation_dict["age"] = age_val
            
            # 3. Save the dictionary as a pickle file
            pkl_filename = series_subfolder + "/" + f"{gsm_name}.pkl"
            with open(pkl_filename, "wb") as f:
                pickle.dump(methylation_dict, f)
            
            print(f"Saved {gsm_name} data -> {pkl_filename} | AGE={age_val}")


if __name__ == "__main__":
    main()
