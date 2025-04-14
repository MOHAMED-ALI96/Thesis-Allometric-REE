import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm  # For significance test

# 🔧 Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ✅ Import configured paths
from scripts.utils.puplic.config_paths import (
    CSV_51, 
    PLOTS_POLYNOMIAL, 
    RESULTS_POLYNOMIAL
)

# -----------------------------
# 1. Data Loading & Preprocessing
# -----------------------------
print("Loading CSV file...")
df = pd.read_csv(CSV_51)

# Define organs and column names.
# (We interpret the independent variable as Log(V/BMI) even if the column is named Log_V(...))
organs = ['Muscles', 'Liver', 'Kidneys', 'Brain', 'Fat', 'Bone', 'Heart']
x_columns = [f"Log_V_bmi({organ})" for organ in organs]      # To be interpreted as Log(V/BMI)
y_columns = [f"Log_SUV_A({organ})" for organ in organs]    # Dependent variable (Log(SUV))

print("Filling NaN values with column medians...")
df[x_columns] = df[x_columns].apply(lambda col: col.fillna(col.median()), axis=0)
df[y_columns] = df[y_columns].apply(lambda col: col.fillna(col.median()), axis=0)

print("Dropping rows with NaN values in Age and Sex...")
df = df.dropna(subset=['Age', 'Sex'])

# Create directories for saving plots and results if they don't exist.
os.makedirs(PLOTS_POLYNOMIAL, exist_ok=True)
os.makedirs(RESULTS_POLYNOMIAL, exist_ok=True)
print(f"Output directory for plots: {PLOTS_POLYNOMIAL}")

# -----------------------------
# 2. Define Helper Functions
# -----------------------------
def fit_polynomial_model(X_train, y_train, degree=2):
    coefs = np.polyfit(X_train, y_train, degree)
    print(f"Fitted polynomial model of degree {degree}")
    return coefs

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def calculate_ase(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def polynomial_predict(coefs, X):
    return np.polyval(coefs, X)

def get_organ_color_map():
    return {
        'Muscles': 'blue',
        'Liver': 'green',
        'Kidneys': 'red',
        'Brain': 'purple',
        'Fat': 'orange',
        'Bone': 'brown',
        'Heart': 'pink'
    }

def save_plot(plot_path):
    try:
        plt.tight_layout()
    except Exception as e:
        print("tight_layout error:", e)
    plt.savefig(plot_path)
    print(f"Plot saved: {plot_path}")
    plt.close()

def format_equation(coefs):
    """
    Given a coefficient array from np.polyfit for a quadratic model ([a, b, c]),
    return a string representing the equation:
      Log(SUV)= a Log(V/BMI)^2 ± b Log(V/BMI) ± c
    using the computed coefficients.
    """
    a, b, c = coefs  # np.polyfit returns [a, b, c] for a quadratic model.
    eq = f"$Log(SUV)= {a:.2f} Log(V/BMI)^2"
    if b >= 0:
        eq += f" + {b:.2f} Log(V/BMI)"
    else:
        eq += f" - {abs(b):.2f} Log(V/BMI)"
    if c >= 0:
        eq += f" + {c:.2f}$"
    else:
        eq += f" - {abs(c):.2f}$"
    return eq

def format_linear_equation(coefs):
    """
    Given a coefficient array from np.polyfit for a linear model ([m, c]),
    return a string representing the equation:
      Log(SUV)= m Log(V/BMI) ± c
    using the computed coefficients.
    """
    m, c = coefs
    eq = f"$Log(SUV)= {m:.2f} Log(V/BMI)"
    if c >= 0:
        eq += f" + {c:.2f}$"
    else:
        eq += f" - {abs(c):.2f}$"
    return eq

# -----------------------------
# 3. K-Fold Cross-Validation & Best Fold Selection (Quadratic)
# -----------------------------
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Pool all data from all organs (flattening arrays)
X_all = df[x_columns].values.flatten()
y_all = df[y_columns].values.flatten()

best_fold_r2, best_fold_mse, best_fold_ase, best_fold = None, None, None, None
fold_results = []  # Will store dictionaries with fold metrics and coefficients.

for fold, (train_index, test_index) in enumerate(kf.split(X_all)):
    X_train, X_test = X_all[train_index], X_all[test_index]
    y_train, y_test = y_all[train_index], y_all[test_index]

    coefs = fit_polynomial_model(X_train, y_train, degree=2)
    y_pred_test = polynomial_predict(coefs, X_test)
    
    r2_val = calculate_r2(y_test, y_pred_test)
    mse_val = calculate_mse(y_test, y_pred_test)
    ase_val = calculate_ase(y_test, y_pred_test)
    
    fold_results.append({
        "Fold": fold + 1,
        "R2": r2_val,
        "MSE": mse_val,
        "ASE": ase_val,
        "Coefficients": coefs.tolist()
    })
    if best_fold_r2 is None or r2_val > best_fold_r2:
        best_fold_r2 = r2_val
        best_fold_mse = mse_val
        best_fold_ase = ase_val
        best_fold = fold + 1

print(f"Best fold for quadratic model: Fold {best_fold} - R²: {best_fold_r2:.3f}, MSE: {best_fold_mse:.3f}, ASE: {best_fold_ase:.3f}")

# Retrieve best fold's coefficients and generate corresponding equation text.
best_fold_dict = next(item for item in fold_results if item["Fold"] == best_fold)
best_coefs = np.array(best_fold_dict["Coefficients"])
equation_text = format_equation(best_coefs)

# Predict on the whole dataset using the best fold's quadratic model.
y_pred_all = polynomial_predict(best_coefs, X_all)
r2_whole = calculate_r2(y_all, y_pred_all)
mse_whole = calculate_mse(y_all, y_pred_all)
ase_whole = calculate_ase(y_all, y_pred_all)

# For smooth plotting of the fitted quadratic curve.
sorted_indices = np.argsort(X_all)
X_all_sorted = X_all[sorted_indices]
y_pred_all_sorted = y_pred_all[sorted_indices]

# -----------------------------
# 4. Plot 1: Fitted Curve and Scatter Plot (Quadratic)
# -----------------------------
plt.figure(figsize=(8, 6))
color_map = get_organ_color_map()
for organ in organs:
    plt.scatter(df[f'Log_V_bmi({organ})'], df[f'Log_SUV_A({organ})'],
                label=organ, color=color_map[organ], alpha=0.5)

plt.plot(X_all_sorted, y_pred_all_sorted,
         label=f"Quadratic Fit (R²: {r2_whole:.3f}, MSE: {mse_whole:.3f}, ASE: {ase_whole:.3f})",
         color='green', alpha=0.6)

plt.xlabel('Log(V/BMI)', fontsize=11)
plt.ylabel('Log(SUV)', fontsize=11)
plt.title(f'Fitted Curve and Scatter Plot\n{equation_text}', fontsize=12)
plt.legend(loc='best', fontsize=9)
plot_name_curve = os.path.join(PLOTS_POLYNOMIAL, "fitted_curve_and_scatter_fixed.png")
save_plot(plot_name_curve)

# -----------------------------
# 5. Plot 2: Prediction vs. Actual Plot (Quadratic)
# -----------------------------
plt.figure(figsize=(8, 6))
for organ in organs:
    x_vals = df[f'Log_V_bmi({organ})']
    actual_y = df[f'Log_SUV_A({organ})']
    predicted_y = polynomial_predict(best_coefs, x_vals)
    plt.scatter(actual_y, predicted_y,
                label=organ, color=color_map[organ], alpha=0.5)

# Plot the ideal diagonal (y = x)
min_val_pa = min(y_all.min(), y_pred_all.min())
max_val_pa = max(y_all.max(), y_pred_all.max())
plt.plot([min_val_pa, max_val_pa], [min_val_pa, max_val_pa],
         color='black', linestyle='--', label='Ideal (y = x)')
plt.xlim([min_val_pa, max_val_pa])
plt.ylim([min_val_pa, max_val_pa])
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel('Actual Log(SUV)', fontsize=11)
plt.ylabel('Predicted Log(SUV)', fontsize=11)
plt.title(f'Prediction vs Actual\n{equation_text}', fontsize=12)
plt.legend(loc='best', fontsize=9)
plot_name_pred_vs_actual = os.path.join(PLOTS_POLYNOMIAL, "prediction_vs_actual_fixed.png")
save_plot(plot_name_pred_vs_actual)

# -----------------------------
# 6. Function to Compare Linear vs. Quadratic Models (F-test)
# -----------------------------
def compare_linear_vs_polynomial(X, y):
    """
    Compare a linear regression model (degree=1) to a quadratic model (degree=2)
    using an F-test for nested models. Returns the F statistic, p-value, and degrees of freedom difference.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Linear model: y = beta0 + beta1*X
    X_lin = sm.add_constant(X)
    model_lin = sm.OLS(y, X_lin).fit()
    
    # Quadratic model: y = beta0 + beta1*X + beta2*X^2
    X_poly = np.column_stack((X, X**2))
    X_poly = sm.add_constant(X_poly)
    model_poly = sm.OLS(y, X_poly).fit()
    
    f_value, p_value, df_diff = model_poly.compare_f_test(model_lin)
    return f_value, p_value, df_diff

f_val, p_val, df_diff = compare_linear_vs_polynomial(X_all, y_all)
print("\nComparison between linear and quadratic models (F-test):")
print(f"F statistic: {f_val:.3f}")
print(f"p-value: {p_val:.3e}")
print(f"Degrees of freedom difference: {df_diff}")

# -----------------------------
# 7. Write Results to Excel
# -----------------------------
def write_results_to_excel(fold_results, best_fold, overall_stats, significance_stats, filepath):
    """
    Write fold results, overall stats, and significance test stats to an Excel workbook.
    """
    df_folds = pd.DataFrame(fold_results)
    df_overall = pd.DataFrame([overall_stats])
    df_significance = pd.DataFrame([significance_stats])
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df_folds.to_excel(writer, sheet_name='Fold Results', index=False)
        df_overall.to_excel(writer, sheet_name='Overall Stats', index=False)
        df_significance.to_excel(writer, sheet_name='Significance Test', index=False)
    print(f"Results written to Excel at: {filepath}")

overall_stats = {
    "Best Fold": best_fold,
    "Best Fold R2": best_fold_r2,
    "Best Fold MSE": best_fold_mse,
    "Best Fold ASE": best_fold_ase,
    "Overall R2": r2_whole,
    "Overall MSE": mse_whole,
    "Overall ASE": ase_whole,
    "Fitted Equation": equation_text
}

significance_stats = {
    "F Statistic": f_val,
    "p-value": p_val,
    "Degrees of Freedom Diff": df_diff
}

excel_filepath = os.path.join(RESULTS_POLYNOMIAL, "polynomial_results.xlsx")
write_results_to_excel(fold_results, best_fold, overall_stats, significance_stats, excel_filepath)

# -----------------------------
# 8. New Plot: Linear vs Quadratic Regression Comparison
# -----------------------------
# Compute linear and quadratic models on the entire dataset.
linear_coefs = np.polyfit(X_all, y_all, deg=1)
poly_coefs_full = np.polyfit(X_all, y_all, deg=2)

# Compute predictions on sorted X for smooth curves.
sorted_idx = np.argsort(X_all)
X_sorted = X_all[sorted_idx]
linear_pred_sorted = np.polyval(linear_coefs, X_sorted)
poly_pred_sorted = np.polyval(poly_coefs_full, X_sorted)

# Evaluation metrics for the full dataset.
linear_r2 = calculate_r2(y_all, np.polyval(linear_coefs, X_all))
linear_mse = calculate_mse(y_all, np.polyval(linear_coefs, X_all))
linear_ase = calculate_ase(y_all, np.polyval(linear_coefs, X_all))

poly_r2 = calculate_r2(y_all, np.polyval(poly_coefs_full, X_all))
poly_mse = calculate_mse(y_all, np.polyval(poly_coefs_full, X_all))
poly_ase = calculate_ase(y_all, np.polyval(poly_coefs_full, X_all))

# Get formatted equation strings.
linear_eq = format_linear_equation(linear_coefs)
poly_eq = format_equation(poly_coefs_full)

# Perform significance test on the full dataset.
f_val_full, p_val_full, df_diff_full = compare_linear_vs_polynomial(X_all, y_all)

# Create the comparison plot.
plt.figure(figsize=(10, 7))
# Plot the raw data.
plt.scatter(X_all, y_all, color='gray', alpha=0.5, label='Data')

# Plot the linear regression line.
plt.plot(X_sorted, linear_pred_sorted, label=f"Linear: {linear_eq}\nR²={linear_r2:.3f}, MSE={linear_mse:.3f}, ASE={linear_ase:.3f}", color='blue', linewidth=2)

# Plot the quadratic regression line.
plt.plot(X_sorted, poly_pred_sorted, label=f"Quadratic: {poly_eq}\nR²={poly_r2:.3f}, MSE={poly_mse:.3f}, ASE={poly_ase:.3f}", color='red', linewidth=2)

# Add a dummy plot to include the significance test p-value in the legend.
plt.plot([], [], ' ', label=f"Significance Test (Linear vs Poly): p-value = {p_val_full:.3e}")

plt.xlabel('Log(V/BMI)', fontsize=12)
plt.ylabel('Log(SUV)', fontsize=12)
plt.title(f"Comparison: Linear vs Quadratic Regression", fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True)
comparison_plot_path = os.path.join(PLOTS_POLYNOMIAL, "linear_vs_polynomial_comparison.png")
save_plot(comparison_plot_path)