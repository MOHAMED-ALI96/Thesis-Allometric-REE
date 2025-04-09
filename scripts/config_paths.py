from pathlib import Path

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Core folders
DATA_DIR = BASE_DIR / "data"
PRIVATE_DATA_DIR = DATA_DIR / "private"
PUBLIC_DATA_DIR = DATA_DIR / "public"

SCRIPTS_DIR = BASE_DIR / "scripts"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
THESIS_DIR = BASE_DIR / "thesis"

# Imaging data (PRIVATE - not pushed to GitHub)
DICOM_DIR = PRIVATE_DATA_DIR / "DICOM" / "FDG-PET-CT-Lesions"
NIFTI_DIR = PRIVATE_DATA_DIR / "NIFTI"
ATLAS_DIR = PRIVATE_DATA_DIR / "Atlas_final_dataset_V1_533"
ATLAS_NEGATIVE_DIR = PRIVATE_DATA_DIR / "Atlas_Negative"
NIFTI_NEGATIVE_DIR = PRIVATE_DATA_DIR / "NIFTI_Negative"
NIFTI_NEGATIVE_SUV_DIR = PRIVATE_DATA_DIR / "NIFTI_Negative_SUV"
NIFTI_NEGATIVE_CTRES_DIR = PRIVATE_DATA_DIR / "NIFTI_Negative_CTres"
ATLAS_NEGATIVE_RES_DIR = PRIVATE_DATA_DIR / "Atlas_Negative_res"

# Script reference
DICOM_TO_NIFTI_SCRIPT = SCRIPTS_DIR / "tcia_dicom_to_nifti.py"

# CSV and clinical metadata (PUBLIC - can be pushed)
CLINICAL_METADATA_CSV = PUBLIC_DATA_DIR / "clinical_metadata.csv"
COLLECTED_METRICS_CSV = PUBLIC_DATA_DIR / "collected_metrics.csv"  # Adjust as needed
ATLAS_NEGATIVE_CSV = PUBLIC_DATA_DIR / "AtlasNegativeNames_1.csv"

# Example usage
if __name__ == "__main__":
    print("Public clinical data path:", CLINICAL_METADATA_CSV)
    print("DICOM folder path (private):", DICOM_DIR)