from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

plots_folders = [
    "01_data_description",
    "02_organ_contribution",
    "03_scatter_exploration",
    "04_modeling/04a_ols_models",
    "04_modeling/04b_final_models",
    "05_linear_model_analysis",
    "06_significance_tests",
    "07_cross_validation_linear",
    "08_polynomial_model",
    "09_cross_validation_poly",
    "10_model_comparison",
    "11_mixed_effects",
    "12_age_analysis",
    "13_volume_change"
]

results_folders = [
    "organwise_stats",
    "model_results",
    "significance_tests",
    "cv_results",
    "polynomial_vs_linear",
    "mixed_effects",
    "age_effects"
]

# Create plots
for f in plots_folders:
    folder = BASE_DIR / "plots" / f
    folder.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created plot folder: {folder.relative_to(BASE_DIR)}")

# Create results
for f in results_folders:
    folder = BASE_DIR / "results" / f
    folder.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created result folder: {folder.relative_to(BASE_DIR)}")
