import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.patches as mpatches

# ------------------------------------------------------------------------------
# Add project root to sys.path if needed (comment out if not used in your setup).
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ------------------------------------------------------------------------------
# Import configured paths (adjust or remove if you have your own path setup)
# ------------------------------------------------------------------------------
from scripts.utils.puplic.config_paths import (
    CSV_325,
    PLOTS_MIXED_EFFECT,
    RESULTS_MIXED_EFFECT
)

# Ensure directories exist
os.makedirs(PLOTS_MIXED_EFFECT, exist_ok=True)
os.makedirs(RESULTS_MIXED_EFFECT, exist_ok=True)

# ------------------------------------------------------------------------------
# Local File & Output Setup
# ------------------------------------------------------------------------------
CSV_FILE = CSV_325
OUTPUT_DIR = Path("plots_bmi")  # local directory if you'd like to store plots here too
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Load the Dataset
# ------------------------------------------------------------------------------
df = pd.read_csv(CSV_FILE)

# Define organs of interest
organs_of_interest = ["Muscles", "Liver", "Kidneys", "Fat", "Bone", "Heart"]

# Define your exact color mapping for each organ
organ_colors = {
    "Muscles": "blue",
    "Liver": "green",
    "Kidneys": "red",
    "Fat": "orange",
    "Bone": "brown",
    "Heart": "pink",
}

# ------------------------------------------------------------------------------
# Build a Combined DataFrame for Mixed-Effects (Using Log_V_bmi)
# ------------------------------------------------------------------------------
combined_data = pd.concat([
    df[['Subject ID', f'Log_V_bmi({organ})', f'Log_SUV_A({organ})']]
    .dropna()
    .rename(columns={
        f'Log_V_bmi({organ})': 'Log_V_bmi',
        f'Log_SUV_A({organ})': 'Log_SUV_A'
    })
    .assign(Organ=organ)
    for organ in organs_of_interest
], ignore_index=True)

# ------------------------------------------------------------------------------
# Create a Custom Palette for Seaborn
# ------------------------------------------------------------------------------
custom_palette = {org: organ_colors[org] for org in organs_of_interest if org in organ_colors}

# ------------------------------------------------------------------------------
# Prepare the Scatter Plot
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x="Log_V_bmi",
    y="Log_SUV_A",
    hue="Organ",
    data=combined_data,
    palette=custom_palette,
    alpha=0.7
)

# For storing model equation lines in the title
title_lines = []
model_summaries = []

# ------------------------------------------------------------------------------
# Mixed Effects Models (Two Lines)
# ------------------------------------------------------------------------------
mixed_models = [
    {
        "formula": "Log_SUV_A ~ Log_V_bmi",
        "group_col": "Subject ID",
        "re_formula": None,
        "description": "Subject-level Random Intercept Model",
        "color": "red",
        "linestyle": "--"
    },
    {
        "formula": "Log_SUV_A ~ Log_V_bmi + C(Organ)",
        "group_col": "Subject ID",
        "re_formula": "~Organ",
        "description": "Subject-level Random Slope on Organ Model",
        "color": "blue",
        "linestyle": "--"
    }
]

for conf in mixed_models:
    try:
        # Fit the mixed effects model
        model = smf.mixedlm(
            conf["formula"],
            combined_data,
            groups=combined_data[conf["group_col"]],
            re_formula=conf["re_formula"]
        )
        fit = model.fit(method="bfgs", maxiter=1000)

        # Extract fixed effects
        intercept = fit.params.get("Intercept", 0)
        slope = fit.params.get("Log_V_bmi", 0)

        # Compute predictions (fixed effects) and metrics
        preds = intercept + slope * combined_data["Log_V_bmi"]
        ss_total = np.sum((combined_data["Log_SUV_A"] - combined_data["Log_SUV_A"].mean()) ** 2)
        ss_resid = np.sum((combined_data["Log_SUV_A"] - preds) ** 2)
        r2 = 1 - (ss_resid / ss_total)
        mse = np.mean((combined_data["Log_SUV_A"] - preds) ** 2)
        ase = np.mean(np.abs(combined_data["Log_SUV_A"] - preds))

        # Build an equation line (title) with no metrics
        equation_text = (
            f"{conf['description']}: "
            f"Log_SUV_A = {intercept:.2f} + {slope:.2f} * Log_V_bmi"
        )
        title_lines.append(equation_text)

        # Plot line with metrics in the legend
        legend_label = (
            f"{conf['description']} "
            f"(R²={r2:.2f}, MSE={mse:.3f}, ASE={ase:.3f})"
        )
        sns.lineplot(
            x=combined_data["Log_V_bmi"],
            y=preds,
            color=conf["color"],
            linestyle=conf["linestyle"],
            linewidth=2,
            label=legend_label
        )

        # Save summary
        model_summaries.append({
            "Model": conf["description"],
            "Intercept": intercept,
            "Slope": slope,
            "R²": r2,
            "MSE": mse,
            "ASE": ase,
            "Log-Likelihood": fit.llf
        })

    except Exception as e:
        title_lines.append(f"{conf['description']} failed: {str(e)}")
        model_summaries.append({"Model": conf["description"], "Error": str(e)})

# ------------------------------------------------------------------------------
# OLS Model (Black Solid Line)
# ------------------------------------------------------------------------------
x_all_ols, y_all_ols = [], []

for organ in organs_of_interest:
    x_col = f"Log_V_bmi({organ})"
    y_col = f"Log_SUV_A({organ})"
    if x_col in df.columns and y_col in df.columns:
        # dropna() for each organ to align x & y
        x_vals = df[x_col].dropna().values
        y_vals = df[y_col].dropna().values
        x_all_ols.extend(x_vals)
        y_all_ols.extend(y_vals)

x_all_ols = np.array(x_all_ols)
y_all_ols = np.array(y_all_ols)

if len(x_all_ols) > 0 and len(y_all_ols) > 0:
    X_ols = sm.add_constant(x_all_ols)
    ols_fit = sm.OLS(y_all_ols, X_ols).fit()

    intercept_ols = ols_fit.params[0]
    slope_ols = ols_fit.params[1]

    # Predictions & metrics
    x_range_ols = np.linspace(x_all_ols.min(), x_all_ols.max(), 200)
    y_pred_ols = intercept_ols + slope_ols * x_range_ols

    # Evaluate
    r2_ols = ols_fit.rsquared
    mse_ols = np.mean((ols_fit.predict(X_ols) - y_all_ols) ** 2)
    ase_ols = np.mean(np.abs(ols_fit.predict(X_ols) - y_all_ols))

    # Title line for OLS (no stats, just the equation)
    ols_eqn_line = (
        f"OLS: Log(SUV) = {intercept_ols:.2f} + {slope_ols:.2f} * Log_V_bmi"
    )
    title_lines.append(ols_eqn_line)

    # Legend label (with stats)
    ols_legend_label = (
        f"OLS Fit (R²={r2_ols:.2f}, MSE={mse_ols:.3f}, ASE={ase_ols:.3f})"
    )

    plt.plot(
        x_range_ols,
        y_pred_ols,
        color="black",
        linewidth=2,
        alpha=0.7,
        label=ols_legend_label
    )

    model_summaries.append({
        "Model": "OLS Fit",
        "Intercept": intercept_ols,
        "Slope": slope_ols,
        "R²": r2_ols,
        "MSE": mse_ols,
        "ASE": ase_ols,
        "Log-Likelihood": ols_fit.llf
    })

# ------------------------------------------------------------------------------
# Finalize Plot (Title, Legend, Axes)
# ------------------------------------------------------------------------------
# Build a multi-line title showing each model's equation
plt.title("\n".join(title_lines), fontsize=11)
plt.xlabel("Log(V/BMI)")
plt.ylabel("Log(SUV)")
plt.grid(True)
plt.tight_layout()

# Place legend inside the plot area (lower left corner, or as you prefer)
plt.legend(title="Models")

# ------------------------------------------------------------------------------
# Save and Display
# ------------------------------------------------------------------------------
plot_filepath = OUTPUT_DIR / "Comparison_Mixed_Models_Log_V_bmi.png"
plt.savefig(plot_filepath, dpi=120, bbox_inches="tight")
plt.show()

print(f"✅ Plot saved to: {plot_filepath}")

# ------------------------------------------------------------------------------
# Save Model Summaries
# ------------------------------------------------------------------------------
model_summaries_df = pd.DataFrame(model_summaries)
summaries_file_path = OUTPUT_DIR / "Mixed_Model_Summaries_Log_V_bmi.csv"
model_summaries_df.to_csv(summaries_file_path, index=False)

print(f"✅ Model summaries saved to: {summaries_file_path}")
