import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.patches as mpatches

# Description:
# This script fits a simple linear regression model (log-log scale) across all organs combined.
# It plots Y = Log(Average SUV) vs. X = Log(V/BMI) with a single regression line.
# The regression equation, R², Mean Squared Error (MSE), and Absolute Squared Error (ASE) are displayed in the plot legend.

# Load dataset
df = pd.read_csv("E:\Project\CSV\data all\Subset_corrected\collected_metrics_All_Full_Brain.csv")

# Define organs and corresponding columns
organs = ['Muscles', 'Liver', 'Kidneys', 'Brain', 'Fat', 'Bone', 'Heart']
x_columns = [f"Log_V_bmi({organ})" for organ in organs]
y_columns = [f"Log_SUV_A({organ})" for organ in organs]

# Define organ colors
organ_colors = {
    'Muscles': 'blue',
    'Liver': 'green',
    'Kidneys': 'red',
    'Brain': 'purple',
    'Fat': 'orange',
    'Bone': 'brown',
    'Heart': 'pink'
}

# Output directory
output_directory = r"E:\1Plots\plots_feb_24\New_Simple_allometric_plot_colors"
os.makedirs(output_directory, exist_ok=True)

# Collect all data points across organs
all_x_values = []
all_y_values = []
all_colors = []

for organ, x_col, y_col in zip(organs, x_columns, y_columns):
    x_values = df[x_col].dropna()
    y_values = df.loc[x_values.index, y_col].dropna()
    x_values = x_values.loc[y_values.index]  # Ensure matching indices

    all_x_values.extend(x_values)
    all_y_values.extend(y_values)
    all_colors.extend([organ_colors[organ]] * len(x_values))

# Convert lists to numpy arrays
all_x_values = np.array(all_x_values)
all_y_values = np.array(all_y_values)
all_colors = np.array(all_colors)

# Fit a single linear regression model
X = sm.add_constant(all_x_values)
model = sm.OLS(all_y_values, X).fit()

# Get regression line
x_range = np.linspace(all_x_values.min(), all_x_values.max(), 100)
y_pred = model.params[0] + model.params[1] * x_range

# Compute statistics
r2 = model.rsquared
mse = np.mean((model.predict(X) - all_y_values) ** 2)
ase = np.mean(np.abs(model.predict(X) - all_y_values))
equation = f"Log SUV = {model.params[0]:.2f} + {model.params[1]:.2f} * Log(V/BMI)"

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(all_x_values, all_y_values, c=all_colors, alpha=0.6, label="True values")
plt.plot(x_range, y_pred, color='black', linestyle='-', linewidth=2,
         label=f"{equation} (R²={r2:.3f}, MSE={mse:.3f}, ASE={ase:.3f})")

# Create custom legend entries for organ colors
legend_patches = [mpatches.Patch(color=color, label=organ) for organ, color in organ_colors.items()]
plt.legend(handles=legend_patches + [
    mpatches.Patch(color='black', label=f"{equation} (R²={r2:.3f}, MSE={mse:.3f}, ASE={ase:.3f})")])

# Plot customization
plt.xlabel("Log(V/BMI)")
plt.ylabel("Log(Average SUV)")
plt.title("Allometric Model: Simple Linear Regression Across All Organs")
plt.grid(True)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(output_directory, "allometric_linear_regression_combined.png")
plt.savefig(plot_path)
plt.close()

print(f"Plot saved to: {plot_path}")