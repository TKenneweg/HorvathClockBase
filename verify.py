import os
import pickle

# List the series you want to check
seriesnames = [
    "GSE41037",
    "GSE15745",
    # Add more as needed
]

data_folder = "./data"

def main():
    for series_id in seriesnames:
        series_subfolder = os.path.join(data_folder, series_id)
        
        if not os.path.isdir(series_subfolder):
            print(f"Folder not found for series '{series_id}' at {series_subfolder}")
            continue
        
        # Gather all .pkl files in this series subfolder
        pkl_files = sorted([f for f in os.listdir(series_subfolder) if f.endswith(".pkl")])
        if not pkl_files:
            print(f"No .pkl files found in {series_subfolder}")
            continue
        
        # Load the first dict as the "reference"
        ref_path = os.path.join(series_subfolder, pkl_files[0])
        with open(ref_path, "rb") as f:
            ref_dict = pickle.load(f)
        
        # We'll compare other dictionaries' keys to these
        ref_keys = list(ref_dict.keys())
        
        mismatch_found = False
        
        # Compare each subsequent dict to the reference
        for pkl_file in pkl_files[1:]:
            curr_path = os.path.join(series_subfolder, pkl_file)
            with open(curr_path, "rb") as f:
                curr_dict = pickle.load(f)
            
            curr_keys = list(curr_dict.keys())
            
            # Check if keys match in length and order
            if curr_keys != ref_keys:
                print(f"\nMismatch in {series_id}:")
                print(f"  Reference file: {pkl_files[0]}")
                print(f"  Current file:   {pkl_file}")
                print("  The sequence of keys does not match.")
                mismatch_found = True
                break
        
        # If no mismatch was found for this series, confirm consistency
        if not mismatch_found:
            print(f"In series '{series_id}', all dictionaries have the same keys in the same order.")

if __name__ == "__main__":
    main()
