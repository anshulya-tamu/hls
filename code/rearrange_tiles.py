import os
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define band lists
bands = {
    "S30": ['B01', 'B05', 'B09', 'B8A', 'B02', 'B06', 'B10', 'Fmask', 'B03', 'B07', 'B11', 'SAA', 'B04', 'B08', 'B12', 'SZA'],
    "L30": ['B01', 'B04', 'B07', 'Fmask', 'B02', 'B05', 'B10', 'SAA', 'B03', 'B06', 'B11', 'SZA']
}

# Base input and output directories
base_input_dir = '/scratch/user/anshulya/hls/data'
base_output_dir = '/scratch/user/anshulya/hls/data/cluster'

# Function to copy and remove a file
def copy_and_remove_file(source, destination):
    try:
        shutil.copy(source, destination)
        os.remove(source)
        return True
    except Exception as e:
        print(f"Error processing {source}: {e}")
        return False

# Function to process a single folder
def process_folder(args):
    product, band_list, folders_path, folder = args
    tile = folder.split('.')[2][1:]

    
    if not tile in ['16RFV', '16RGV', '17SMS', '15SXT', '16SEC', '16SDB', '16SFA', '16SGA', '16SCE', '16SDE', '16SED', '16SDC', '16SCG', '16SDG', '16SCF', '16SDF']:
        return
    
    day = folder.split('.')[3].split('T')[0]
    files_path = os.path.join(folders_path, folder)
    files = os.listdir(files_path)
    
    # Destination folder
    destination_folder = f"{base_output_dir}/{tile}/unmerged_rasters/{day}/{product}"
    os.makedirs(destination_folder, exist_ok=True)
    
    tasks = []
    for fi in files:
        if fi.split('.')[-2] in band_list:  # Check if the band matches
            source = os.path.join(files_path, fi)
            destination = os.path.join(destination_folder, fi)
            tasks.append((source, destination))
    
    # Process file copying/removal in parallel
    with ProcessPoolExecutor(max_workers=40) as executor:
        futures = [executor.submit(copy_and_remove_file, src, dst) for src, dst in tasks]
        for future in as_completed(futures):
            future.result()  # Handle exceptions if any

# Main function to process S30 and L30
def process_files(product, band_list):
    try:
        for y in os.listdir(f'{base_input_dir}/{product}'):
            for tn in os.listdir(f'{base_input_dir}/{product}/{y}'):
                for t1 in os.listdir(f'{base_input_dir}/{product}/{y}/{tn}'):
                    for t2 in os.listdir(f'{base_input_dir}/{product}/{y}/{tn}/{t1}'):
                        for t3 in os.listdir(f'{base_input_dir}/{product}/{y}/{tn}/{t1}/{t2}'):
                            folders_path = f'{base_input_dir}/{product}/{y}/{tn}/{t1}/{t2}/{t3}'
                            folders = os.listdir(folders_path)
                            print(f"Processing {product} - {y} {tn} {t1} {t2} {t3}")
                            
                            # Parallelize folder processing
                            with ProcessPoolExecutor(max_workers=40) as executor:
                                args = [(product, band_list, folders_path, folder) for folder in folders]
                                list(tqdm(executor.map(process_folder, args), total=len(folders), desc="Processing Folders"))
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected! Shutting down gracefully...")
        exit(1)  # Graceful exit

# Entry point
if __name__ == "__main__":
    try:
        # Process S30 and L30 products
        print("Starting to process S30 product...")
        process_files("S30", bands["S30"])
        
        print("Starting to process L30 product...")
        process_files("L30", bands["L30"])

        print("Processing completed successfully!")
    except KeyboardInterrupt:
        print("\nInterrupted! Exiting the script.")
