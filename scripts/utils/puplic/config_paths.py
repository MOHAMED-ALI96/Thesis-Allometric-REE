from pathlib import Path

# Define root paths
ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data"
PLOTS_DIR = ROOT_DIR / "plots"
RESULTS_DIR = ROOT_DIR / "results"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
SCRIPTS_DIR = ROOT_DIR / "scripts"
BASH_DIR = ROOT_DIR / "bash"
THESIS_DIR = ROOT_DIR / "thesis"

# Data subdirectories
PUBLIC_DATA_DIR = DATA_DIR / "public"
PRIVATE_DATA_DIR = DATA_DIR / "private"

# CSV files
CSV_325 = PUBLIC_DATA_DIR / "targeted_organs_325.csv"
CSV_51 = PUBLIC_DATA_DIR / "targeted_organs_full_brain_51.csv"
CSV_ALL_325 = PUBLIC_DATA_DIR / "all_organs_325.csv"
CSV_ALL_51 = PUBLIC_DATA_DIR / "all_organs_full_brain_51.csv"

# Plot paths
PLOTS_DESCRIPTION = PLOTS_DIR / "00_data_description"
PLOTS_CONTRIBUTION = PLOTS_DIR / "01_organ_contribution"
PLOTS_MODEL_SELECTION = PLOTS_DIR / "02_model_exploration_in_organs_selection"
PLOTS_GENERAL_MODEL = PLOTS_DIR / "03_primary_general_allometric_model"
PLOTS_SUBJECT_LINEAR = PLOTS_DIR / "04_linear_model_subject_level"
PLOTS_POLYNOMIAL = PLOTS_DIR / "05_polynomial_model"
PLOTS_MIXED_EFFECT = PLOTS_DIR / "06_mixed_effect_model"
PLOTS_VALIDATION = PLOTS_DIR / "07_model_validation"
PLOTS_SIGNIFICANCE = PLOTS_DIR / "08_significance_testing"

# Results paths
RESULTS_CONTRIBUTION = RESULTS_DIR / "01_organ_contribution"
RESULTS_MODEL_SELECTION = RESULTS_DIR / "02_model_exploration_in_organs_selection"
RESULTS_GENERAL_MODEL = RESULTS_DIR / "03_primary_general_allometric_model"
RESULTS_SUBJECT_LINEAR = RESULTS_DIR / "04_linear_model_subject_level"
RESULTS_POLYNOMIAL = RESULTS_DIR / "05_polynomial_model"
RESULTS_MIXED_EFFECT = RESULTS_DIR / "06_mixed_effect_model"
RESULTS_VALIDATION = RESULTS_DIR / "07_model_validation"
RESULTS_SIGNIFICANCE = RESULTS_DIR / "08_significance_testing"

# Scripts subfolders
SCRIPTS_PREPROCESSING = SCRIPTS_DIR / "00_preprocessing"
SCRIPTS_CONTRIBUTION = SCRIPTS_DIR / "01_organ_contribution"
SCRIPTS_MODEL_SELECTION = SCRIPTS_DIR / "02_model_exploration_in_organs_selection"
SCRIPTS_GENERAL_MODEL = SCRIPTS_DIR / "03_primary_general_allometric_model"
SCRIPTS_SUBJECT_LINEAR = SCRIPTS_DIR / "04_linear_model_subject_level"
SCRIPTS_POLYNOMIAL = SCRIPTS_DIR / "05_polynomial_model"
SCRIPTS_COHORT_LINEAR = SCRIPTS_DIR / "06_linear_model_full_cohort"
SCRIPTS_VALIDATION = SCRIPTS_DIR / "07_model_validation"
SCRIPTS_SIGNIFICANCE = SCRIPTS_DIR / "08_significance_testing"
SCRIPTS_UTILS = SCRIPTS_DIR / "utils"
SCRIPTS_UTILS_PUBLIC = SCRIPTS_UTILS / "public"
SCRIPTS_UTILS_PRIVATE = SCRIPTS_UTILS / "private"       