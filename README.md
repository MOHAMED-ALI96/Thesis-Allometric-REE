# Thesis-Allometric-REE

This repository contains the full workflow, scripts, notebooks, and documentation for my Master's thesis:

**"Allometric Modeling for Estimating Resting Energy Expenditure (REE) from PET-CT Data"**

The goal is to analyze the relationship between PET-derived organ volumes and standardized uptake values (SUV) using allometric equations. The result is a reproducible modeling pipeline for estimating REE based on PET-CT data.

---

## 📁 Repository Structure

This project is structured for clarity and reproducibility:

| Folder           | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `scripts/`       | Python modules and utilities (e.g. config paths, model building, tools)     |
| `notebooks/`     | Jupyter notebooks for visualizations, EDA, modeling, and results            |
| `bash/`          | Bash scripts to automate folder setup and preprocessing                     |
| `data/public/`   | Publicly shareable CSVs (e.g. clinical metadata)                            |
| `data/private/`  | Raw DICOM, NIFTI, or sensitive data (🚫 ignored by Git)                     |
| `results/`       | Output files from scripts or notebooks (tables, model results, etc.)        |
| `plots/`         | Generated figures and plots                                                 |
| `thesis/`        | Drafts, references, and presentation materials   

---

## 🔐 Data Usage & Privacy

This project uses imaging and metadata from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/):

- 📂 `data/private/`  
  - Contains DICOM, NIFTI, and segmentation files  
  - **Licensed as "TCIA Restricted" – cannot be shared**  
  - ❌ Ignored by Git using `.gitignore`

- 📂 `data/public/`  
  - Contains CSV metadata (age, sex, diagnosis, etc.)  
  - **Licensed under CC BY 4.0 – can be redistributed**  
  - ✅ Included in this repository

If you are cloning this repo and want to use the imaging data, you must download it directly from TCIA and store it inside `data/private/`.

---

## ⚙️ Setup Instructions

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/MOHAMED-ALI96/Thesis-Allometric-REE.git
cd Thesis-Allometric-REE

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

jupyter notebook