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

To run this project locally, follow these steps to set up a clean Python environment using `requirements.txt`.

---

### 1. Clone the Repository

```bash
git clone https://github.com/MOHAMED-ALI96/Thesis-Allometric-REE.git
cd Thesis-Allometric-REE
2. Create a Virtual Environment
You only need to do this once per machine.

Windows:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate
macOS/Linux:

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
3. Install Required Packages
Install the minimal required Python libraries for this project using:

bash
Copy
Edit
pip install -r requirements.txt
4. Launch Jupyter
To explore the notebooks:

bash
Copy
Edit
jupyter notebook
Or, open this folder in VS Code, and run notebooks with the Jupyter extension.

🛠 Optional Notes
If you update the environment, you can regenerate the file with:

bash
Copy
Edit
pip freeze > requirements.txt
Never include the venv/ folder in your repo — it's already excluded in .gitignore.

yaml
Copy
Edit

---

## ✅ What To Do Next

Now you can:
- ✅ Paste this into your existing `README.md` file
- ✅ Commit it:

```bash
git add README.md
git commit -m "Update README with clean requirements.txt setup instructions"
git push