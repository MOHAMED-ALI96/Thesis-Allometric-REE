from pathlib import Path

# 📁 Root project directory (automatically resolved)
ROOT = Path(__file__).resolve().parents[2]

# 🧱 Define base structure
folders_to_create = {
    "scripts": [
        "00_preprocessing",
        "01_organ_contribution",
        "02_general_allometric_model",
        "03_model_exploration",
        "04_linear_model_subject_level",
        "05_polynomial_model",
        "06_linear_model_full_cohort",
        "07_model_validation",
        "08_significance_testing"
    ],
    "plots": [
        "01_organ_contribution/suv_shares",
        "01_organ_contribution/volume_shares",
        "01_organ_contribution/log_scale_scatter",

        "02_general_allometric_model/scatter_colored",
        "02_general_allometric_model/regression_lines",
        "02_general_allometric_model/residual_diagnostics",

        "03_model_exploration/best_model",
        "03_model_exploration/model_grid_summary",

        "04_linear_model_subject_level/subject_regression_lines",
        "04_linear_model_subject_level/boxplots_r2_slope_51",

        "05_polynomial_model/poly_vs_linear_curves",
        "05_polynomial_model/cross_validation_plots",

        "06_linear_model_full_cohort/subject_regression_lines",
        "06_linear_model_full_cohort/boxplots_r2_slope_325",

        "07_model_validation/cross_validation",
        "07_model_validation/qq_plots",
        "07_model_validation/mixed_effects_model",

        "08_significance_testing/age/slope_vs_age",
        "08_significance_testing/age/r2_vs_age",
        "08_significance_testing/age/organwise_suv_vs_age",

        "08_significance_testing/sex/ttest_slopes_r2",
        "08_significance_testing/sex/organwise_suv_vs_sex",

        "08_significance_testing/bmi/slope_vs_bmi",
        "08_significance_testing/bmi/r2_vs_bmi",
        "08_significance_testing/bmi/organwise_volume_suv_vs_bmi"
    ],
    "results": [
        "01_organ_contribution",
        "02_general_allometric_model",
        "03_model_exploration",
        "04_linear_model_subject_level",
        "05_polynomial_model",
        "06_linear_model_full_cohort",
        "07_model_validation",
        "08_significance_testing"
    ]
}

def create_folders(base_dir, subdirs):
    for sub in subdirs:
        path = base_dir / sub
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {path.relative_to(ROOT)}")

def main():
    for category, subdirs in folders_to_create.items():
        base = ROOT / category
        create_folders(base, subdirs)

if __name__ == "__main__":
    print("🚀 Creating thesis directory structure...")
    main()
    print("✅ All folders ready.")
