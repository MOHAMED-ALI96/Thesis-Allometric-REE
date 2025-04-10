import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.patches as mpatches

# Description:
# This script fits a simple linear regression model (log-log scale) for a specific subject and study ID.
# It saves relevant data in an Excel sheet and plots Y = Log(Average SUV) vs. X = Log(V/BMI) with a single regression line.
# The regression equation, R², Mean Squared Error (MSE), and Absolute Squared Error (ASE) are displayed in the plot legend.
# Additionally, it includes White Matter (WM) and Gray Matter (GM) points.

# Load dataset
df = pd.read_csv(r"E:\Project\CSV\data all_corrected\collected_metrics_All_325_sum_t.csv")

# Define target subject and study ID
target_subject = "f0a1a38a3b"
target_study = 38725

# Filter dataset for the specific subject and study
df_subject = df[(df['Subject ID'] == target_subject) & (df['Study ID'] == target_study)]

# Define organs and corresponding colors
organs = ['Muscles', 'Liver', 'Kidneys', 'Brain', 'Fat', 'Bone', 'Heart']
organ_colors = {
    'Muscles': 'blue',
    'Liver': 'green',
    'Kidneys': 'red',
    'Brain': 'purple',
    'Fat': 'orange',
    'Bone': 'brown',
    'Heart': 'pink',
    'WM': 'yellow',
    'GM': 'gray'
}

# Define column names
x_columns = [f"Log_V_bmi({organ})" for organ in organs]
y_columns = [f"Log_SUV_A({organ})" for organ in organs]

# Define output directory
output_directory = r"E:\1Plots\plots_feb_3"
os.makedirs(output_directory, exist_ok=True)

# Collect data points for the specific subject
data = []
all_x_values = []
all_y_values = []
all_colors = []

for organ, x_col, y_col in zip(organs, x_columns, y_columns):
    x_values = df_subject[x_col].dropna()
    y_values = df_subject.loc[x_values.index, y_col].dropna()
    v_values = df_subject[f"V({organ})"].dropna()
    suv_values = df_subject[f"SUV_A({organ})"].dropna()

    if not x_values.empty and not y_values.empty:
        data.append([organ, df_subject['BMI'].values[0], df_subject['Age'].values[0], df_subject['Sex'].values[0],
                     suv_values.values[0], v_values.values[0], x_values.values[0], y_values.values[0]])
        all_x_values.extend(x_values)
        all_y_values.extend(y_values)
        all_colors.extend([organ_colors[organ]] * len(x_values))

# Correct WM and GM volume calculations
brain_volume = df_subject["V(Brain)"].values[0]
bmi = df_subject["BMI"].values[0]
wm_volume = 0.45 * brain_volume  # 45% of total brain volume
gm_volume = 0.55 * brain_volume  # 55% of total brain volume
wm_log_v_bmi = np.log(wm_volume / bmi)
gm_log_v_bmi = np.log(gm_volume / bmi)

wm_suv = 3.98
gm_suv = 7.82
wm_log_suv = np.log(wm_suv)
gm_log_suv = np.log(gm_suv)

data.append(
    ["WM", bmi, df_subject['Age'].values[0], df_subject['Sex'].values[0], wm_suv, wm_volume, wm_log_v_bmi, wm_log_suv])
data.append(
    ["GM", bmi, df_subject['Age'].values[0], df_subject['Sex'].values[0], gm_suv, gm_volume, gm_log_v_bmi, gm_log_suv])
all_x_values.extend([wm_log_v_bmi, gm_log_v_bmi])
all_y_values.extend([wm_log_suv, gm_log_suv])
all_colors.extend([organ_colors['WM'], organ_colors['GM']])

# Convert data into a DataFrame and save as Excel
columns = ["Organ", "BMI", "Age", "Sex", "SUV_A", "V", "Log_V_bmi", "Log_SUV_A"]
df_output = pd.DataFrame(data, columns=columns)
output_file = os.path.join(output_directory, f"subject_data_{target_subject}_study_{target_study}.xlsx")
df_output.to_excel(output_file, index=False)

print(f"Data saved to: {output_file}")

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

# Plot All Data
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
plt.title(f"Allometric Model: Subject {target_subject}, Study {target_study}")
plt.grid(True)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(output_directory, f"allometric_regression_subject_{target_subject}_study_{target_study}.png")
plt.savefig(plot_path)
plt.close()

print(f"Plot saved to: {plot_path}")
