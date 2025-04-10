from pathlib import Path

# ✅ Project root directory (Thesis-Allometric-REE/)
ROOT_DIR = Path(__file__).resolve().parents[3]

# 📁 Data folders
DATA_DIR = ROOT_DIR / "data"
PUBLIC_DATA_DIR = DATA_DIR / "public"
PRIVATE_DATA_DIR = DATA_DIR / "private"

# 📁 Public data files (commonly used)
CSV_325 = PUBLIC_DATA_DIR / "targeted_organs_325.csv"
CSV_51 = PUBLIC_DATA_DIR / "targeted_organs_full_brain_51.csv"

# 📁 Plot output folders
PLOTS_DIR = ROOT_DIR / "plots"

PLOTS_DESCRIPTION = PLOTS_DIR / "01_data_description"
PLOTS_CONTRIBUTION = PLOTS_DIR / "02_organ_contribution"
PLOTS_SCATTER = PLOTS_DIR / "03_scatter_exploration"
PLOTS_MODELING = PLOTS_DIR / "04_modeling"
PLOTS_LINEAR = PLOTS_DIR / "05_linear_model_analysis"
PLOTS_SIGNIFICANCE = PLOTS_DIR / "06_significance_tests"
PLOTS_CV_LINEAR = PLOTS_DIR / "07_cross_validation_linear"
PLOTS_POLY = PLOTS_DIR / "08_polynomial_model"
PLOTS_CV_POLY = PLOTS_DIR / "09_cross_validation_poly"
PLOTS_COMPARISON = PLOTS_DIR / "10_model_comparison"
PLOTS_MIXED = PLOTS_DIR / "11_mixed_effects"
PLOTS_AGE = PLOTS_DIR / "12_age_analysis"
PLOTS_VOLUME = PLOTS_DIR / "13_volume_change"

# 📁 Results output folders
RESULTS_DIR = ROOT_DIR / "results"

RESULTS_ORGAN_STATS = RESULTS_DIR / "organwise_stats"
RESULTS_MODELS = RESULTS_DIR / "model_results"
RESULTS_SIGNIFICANCE = RESULTS_DIR / "significance_tests"
RESULTS_CV = RESULTS_DIR / "cv_results"
RESULTS_POLY = RESULTS_DIR / "polynomial_vs_linear"
RESULTS_MIXED = RESULTS_DIR / "mixed_effects"
RESULTS_AGE = RESULTS_DIR / "age_effects"
 
# 📁 Notebooks & Thesis
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
THESIS_DIR = ROOT_DIR / "thesis"
