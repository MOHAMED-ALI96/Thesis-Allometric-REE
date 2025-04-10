import os
import shutil
import pandas as pd
from termcolor import colored
import glob
import config_paths as cng
def copy_folders_from_csv(csv_path, source_folder, output_folder):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        filename = row['File Name']  # Adjusted here
        subject_id = row['Subject ID']
        study_uid = str(row['Study UID']).zfill(5)[-5:]  # Convert to string and extract the last 5 digits

        # Construct the source and destination paths
        source_folder_path = os.path.join(source_folder, f"PETCT_{subject_id}", f"*-{study_uid.zfill(5)}")
        destination_folder = f"PETCT_{subject_id}"

        # Use glob to find matching folders based on the pattern
        matching_folders = glob.glob(source_folder_path)

        if matching_folders:
            source_folder_path = matching_folders[0]

            # Extract the subfolder name from the source path
            subfolder_name = os.path.basename(source_folder_path)

            # Construct the destination path
            destination_path = os.path.join(output_folder, destination_folder, subfolder_name)

            # Ensure the destination directory exists
            os.makedirs(destination_path, exist_ok=True)

            # Copy the contents of the source folder to the destination folder
            for item in os.listdir(source_folder_path):
                s = os.path.join(source_folder_path, item)
                d = os.path.join(destination_path, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, symlinks=True, ignore=None)
                else:
                    shutil.copy2(s, d)

            print(f"Copied {filename} to {destination_path}")
        else:
            print(colored(f"No matching folder found for {filename}", 'red', attrs=['bold']))

# Define paths
csv_path = r"E:\01 Project\Project\Atlas_Negative\AtlasNegativeNames_1.csv"
source_folder =cng.nifti_path
output_folder =cng.Nifti_negative_path
# Call the function
copy_folders_from_csv(csv_path, source_folder, output_folder)
