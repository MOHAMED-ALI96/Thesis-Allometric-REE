# Thesis-Allometric-REE

This repository contains the full workflow, scripts, notebooks, and documentation for my Master's thesis:

**"Allometric Modeling for Estimating Resting Energy Expenditure (REE) from PET-CT Data"**

The project aims to explore allometric relationships between PET-CT-derived organ volumes and standardized uptake values (SUV) across key organs. The ultimate goal is to model REE using these imaging-based variables.

---

## 📁 Repository Structure

```
Thesis-Allometric-REE/
├── scripts/              # Python scripts (models, config, helpers)
├── notebooks/            # Jupyter notebooks for visualizations and analysis
├── bash/                 # Bash scripts for preprocessing
├── data/
│   ├── public/           # Public CSV files (shareable)
│   └── private/          # Imaging data (not tracked by Git)
├── results/              # Outputs (tables, CSVs)
├── plots/                # Figures and charts
├── thesis/               # Thesis drafts and presentations
├── requirements.txt      # Python dependencies
├── .gitignore            # Files/folders to ignore
├── README.md             # This file
```

---

## 🔐 Data Usage & Privacy

This project uses PET-CT data from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/).

### 🔸 `data/private/` (NOT pushed to GitHub)

- Contains all imaging data (DICOM, NIFTI, SEG)
- Licensed under **TCIA Restricted**
- Ignored via `.gitignore`

### 🔸 `data/public/` (INCLUDED in GitHub)

- Contains CC BY 4.0 licensed clinical CSVs
- Includes metadata like age, sex, diagnosis, SUV, volume
- Safe to share and track

---

## ⚙️ Setup Instructions

To run this project locally, follow these steps to set up a clean Python environment using `requirements.txt`.

### 1. Clone the Repository

```bash
git clone https://github.com/MOHAMED-ALI96/Thesis-Allometric-REE.git
cd Thesis-Allometric-REE
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

This installs only the clean dependencies used in the refined project (e.g., numpy, pandas, jupyter, matplotlib).

### 4. Launch Jupyter Notebooks

```bash
jupyter notebook
```

Or open the project folder in **VS Code** and use the Jupyter extension.

---

## 🚀 How to Use

### 📓 Jupyter Notebooks

Located in `notebooks/`:
- Use `scripts/config_paths.py` to access paths
- Each notebook covers a clean topic (EDA, modeling, metrics)
- Example: `01_explore_clinical_data.ipynb`

### 🛠 Python Scripts

Located in `scripts/`:
- `config_paths.py`: manages all file/folder paths
- `create_repo_structure.py`: creates folders
- `add_gitkeep.py`: ensures Git tracks empty folders

### 📜 Bash Scripts

Located in `bash/`:
- Example: `preprocess_data.sh`
- Used for file organization or data conversion steps

---

## 🧪 Example Data

You’ll find a sample clinical metadata file in:

```
data/public/clinical_metadata.csv
```

Fields include:
- `SubjectID`, `Age`, `Sex`, `Diagnosis`

Use this to test notebooks without downloading raw image data.

---

## 🧾 Requirements

Installed from:

```
requirements.txt
```

To regenerate it after changes:

```bash
pip freeze > requirements.txt
```

> Do **not** copy or commit the `venv/` folder — it’s already ignored.

---

## 📄 License

- Code: MIT License
- Public data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Imaging data: TCIA Restricted License

---

## 📌 Roadmap

- [x] Clean repo structure
- [x] Add `.gitignore` and `.gitkeep` handling
- [x] Separate public/private data
- [ ] Add `DATA_ACCESS.md` for external data sources
- [ ] Complete modeling notebooks
- [ ] Add plots and evaluation summaries

---

## 🙌 Acknowledgements

- [TCIA](https://www.cancerimagingarchive.net/) for providing data
- Python open-source tools: NumPy, Pandas, Matplotlib, Scikit-learn
- Jupyter & VS Code for the development environment

---

*Project still in progress — more updates soon!*
