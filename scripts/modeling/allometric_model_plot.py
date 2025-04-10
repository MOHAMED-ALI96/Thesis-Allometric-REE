from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm

# 🔧 Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ✅ Import configured paths
from scripts.utils.puplic.config_paths import (
    CSV_51,CSV_325,
    PLOTS_MODELING
)

# 📦 Load dataset
df = pd.read_csv(CSV_325)

# 🧠 Define organs and colors
organs = ['Muscles', 'Liver', 'Kidneys', 'Fat', 'Bone', 'Heart'
         # ,'Brain'
          ]
organ_colors = {
    'Muscles': 'blue',
    'Liver': 'green',
    'Kidneys': 'red',
    'Fat': 'orange',
    'Bone': 'brown',
    'Heart': 'pink',
   # 'Brain': 'purple'
}

# 📁 Define output folder
output_dir = PLOTS_MODELING / "04b_final_models"
output_dir.mkdir(parents=True, exist_ok=True)


def plot_allometric_regression(filtered_organs, filename_suffix="model", return_stats=False):
    """
    Create and save an allometric regression plot for the given list of organs.
    Optionally returns a dictionary of regression statistics.
    """
    x_all, y_all, colors = [], [], []

    for organ in filtered_organs:
        x_col = f"Log_V_bmi({organ})"
        y_col = f"Log_SUV_A({organ})"
        if x_col in df.columns and y_col in df.columns:
            x = df[x_col].dropna().values
            y = df[y_col].dropna().values

            x_all.extend(x)
            y_all.extend(y)
            colors.extend([organ_colors.get(organ, 'gray')] * len(x))

    x_all = np.array(x_all)
    y_all = np.array(y_all)

    # Linear regression
    X = sm.add_constant(x_all)
    model = sm.OLS(y_all, X).fit()

    # Regression line
    x_range = np.linspace(x_all.min(), x_all.max(), 100)
    y_pred = model.params[0] + model.params[1] * x_range

    # Evaluation metrics
    r2 = model.rsquared
    mse = np.mean((model.predict(X) - y_all) ** 2)
    ase = np.mean(np.abs(model.predict(X) - y_all))
    equation = f"Model Equation: Log(SUV) = {model.params[0]:.2f}{model.params[1]:.2f} * Log(V/BMI),"
    stats_line = f"R²={r2:.3f}, MSE={mse:.3f}, ASE={ase:.3f}"

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(x_all, y_all, c=colors, alpha=0.6, label="True values")
    plt.plot(x_range, y_pred, color="black", linewidth=3, label="OLS Fit")

    legend_patches = [mpatches.Patch(color=organ_colors[o], label=o)
                      for o in filtered_organs if o in organ_colors]
    plt.legend(handles=legend_patches + [mpatches.Patch(color='black', label="OLS Fit")])
    plt.xlabel("Log(V/BMI)")
    plt.ylabel("Log(SUV)")
    plt.title(
        f"Allometric Model for: {', '.join(filtered_organs)}\n{equation} ({stats_line})",
        fontsize=11
    )
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    save_path = output_dir / f"allometric_model_{filename_suffix}.png"
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved plot: {save_path.name}")

    if return_stats:
        return {
            "organs": filtered_organs,
            "intercept": model.params[0],
            "slope": model.params[1],
            "r2": r2,
            "mse": mse,
            "ase": ase
        }

# 🟢 Call the fucntion
if __name__ == "__main__":
    print("🔍 Running allometric plot test...")
    result = plot_allometric_regression(
        ['Muscles', 'Liver', 'Kidneys', 'Fat', 'Bone', 'Heart'
        # ,'Brain'
         ],
        filename_suffix="325_no_brain_test",
        return_stats=True
    )
    print("📊 Model stats:", result)