from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import ttest_ind, pearsonr

# 🔧 Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ✅ Import configured paths
from scripts.utils.puplic.config_paths import (
    CSV_51, CSV_325,
    PLOTS_SUBJECT_LINEAR, PLOTS_SIGNIFICANCE, RESULTS_SIGNIFICANCE, RESULTS_SUBJECT_LINEAR
)

# ===============================
# Global Variables
# ===============================
organs = ['Muscles', 'Liver', 'Kidneys', 'Brain', 'Fat', 'Bone', 'Heart']
x_columns = [f"Log_V_bmi({organ})" for organ in organs]
y_columns = [f"Log_SUV_A({organ})" for organ in organs]

# ===============================
# Core Analysis Functions
# ===============================
def fit_ols_model(x_values, y_values):
    """
    Fit an ordinary least squares (OLS) regression model.
    Returns the fitted model or None if not enough data.
    """
    if len(x_values) < 2:
        return None
    X = sm.add_constant(x_values)
    model = sm.OLS(y_values, X).fit()
    return model


def calculate_average_model_parameters(df, x_cols, y_cols, exclude_brain=False):
    """
    For each subject (row) in the dataframe, fit an OLS model using the provided columns.
    Optionally exclude the brain data (first element) from the x and y arrays.
    Returns average intercept, slope, R² (and its std), as well as lists of individual intercepts, slopes and R² values.
    """
    intercepts, slopes, r_squared_values = [], [], []
    for _, row in df.iterrows():
        x_values = row[x_cols].astype(float).values
        y_values = row[y_cols].astype(float).values
        if exclude_brain:
            x_values = x_values[1:]
            y_values = y_values[1:]
        model = fit_ols_model(x_values, y_values)
        if model is not None:
            intercepts.append(model.params[0])
            slopes.append(model.params[1])
            r_squared_values.append(model.rsquared)
    avg_intercept = np.mean(intercepts)
    avg_slope = np.mean(slopes)
    avg_r2 = np.mean(r_squared_values)
    std_r2 = np.std(r_squared_values)
    return avg_intercept, avg_slope, avg_r2, std_r2, intercepts, slopes, r_squared_values


def plot_regression_lines(df, x_cols, y_cols, avg_intercept, avg_slope, intercepts, slopes,
                          r_squared_values, std_dev_r2, extended_x_range,
                          title, filename, output_dir, exclude_brain=False):
    """
    Plot all individual OLS regression lines for subjects along with the average regression line.
    Saves the resulting plot to the specified output directory.
    """
    plt.figure(figsize=(18, 12))
    slope_values, r2_values, all_x_values, all_y_values = [], [], [], []
    print("Processing individual subject models:")
    for index, row in df.iterrows():
        x_vals = row[x_cols].astype(float).values
        y_vals = row[y_cols].astype(float).values
        if exclude_brain:
            x_vals = x_vals[1:]
            y_vals = y_vals[1:]
        all_x_values.extend(x_vals)
        all_y_values.extend(y_vals)
        model = fit_ols_model(x_vals, y_vals)
        if model is not None:
            y_pred = model.params[0] + model.params[1] * extended_x_range
            plt.plot(extended_x_range, y_pred, color='gray', linewidth=1, alpha=0.5)
            slope_values.append(model.params[1])
            r2_values.append(model.rsquared)
    print(f"Average R²: {np.mean(r2_values):.3f}; Average Slope: {np.mean(slope_values):.3f}")

    # Plot average regression line
    line_style = '--' if exclude_brain else '-'
    y_extended_pred = avg_intercept + avg_slope * extended_x_range
    avg_line, = plt.plot(extended_x_range, y_extended_pred, color='red', linestyle=line_style, linewidth=3)
    intercept_point, = plt.plot(0, avg_intercept, 'bo', markersize=10)
    
    x_min, x_max = min(all_x_values), max(all_x_values)
    y_min, y_max = min(all_y_values), max(all_y_values)
    y_min = min(y_min, avg_intercept - 1)
    y_max = max(y_max, avg_intercept + 1)
    plt.xlim(0, x_max + 1)
    plt.ylim(y_min, y_max)
    
    r2_mean, r2_sd = np.mean(r2_values), np.std(r2_values)
    label_suffix = " (Excluding Brain)" if exclude_brain else " (Including Brain)"
    equation_text = (f'Log SUV = {avg_intercept:.3f} (± {np.std(intercepts):.3f}) + '
                     f'{avg_slope:.3f} (± {np.std(slopes):.3f}) * Log Volume{label_suffix}')
    r2_text = f'R²: {r2_mean:.3f} (± {r2_sd:.3f})'
    intercept_text = f'Avg. Intercept = {avg_intercept:.3f}'
    
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color='gray', lw=2),
            avg_line,
            intercept_point,
            plt.Line2D([0], [0], color='white', lw=0)
        ],
        labels=[
            f'Individual subject models, N={len(r2_values)}',
            equation_text,
            intercept_text,
            r2_text
        ],
        loc='best', fontsize=12
    )
    plt.title(title)
    plt.xlabel('Logarithm of Volume of an Organ')
    plt.ylabel('Logarithm of the Average SUV within an Organ')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_box_plots(slopes_list, r2_list, labels, filename, output_dir):
    """
    Create side-by-side box plots for allometric exponent (slope) and R² values,
    then save the plots to the provided output directory.
    """
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    axs[0].boxplot(slopes_list, labels=labels)
    axs[0].set_title('Box Plot of Allometric Exponent')
    axs[0].set_ylabel('Slope')
    for i, group in enumerate(slopes_list):
        axs[0].plot([], [], ' ', label=f'{labels[i]}: Avg = {np.mean(group):.3f} ± {np.std(group):.3f}')
    axs[0].legend(loc='upper right', fontsize=12)
    
    axs[1].boxplot(r2_list, labels=labels)
    axs[1].set_title('Box Plot of R-squared')
    axs[1].set_ylabel('R²')
    for i, group in enumerate(r2_list):
        axs[1].plot([], [], ' ', label=f'{labels[i]}: Avg = {np.mean(group):.3f} ± {np.std(group):.3f}')
    axs[1].legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_sex_comparison_box_plots(df, x_cols, y_cols, exclude_brain, filename, output_dir):
    """
    Create and save box plots comparing the allometric exponent and R² values between males and females.
    """
    df_male = df[df['Sex'] == 'M']
    df_female = df[df['Sex'] == 'F']
    
    # Calculate parameters for both groups (using only the chosen modeling approach)
    _, _, _, _, _, slopes_m, r2_m = calculate_average_model_parameters(df_male, x_cols, y_cols, exclude_brain=exclude_brain)
    _, _, _, _, _, slopes_f, r2_f = calculate_average_model_parameters(df_female, x_cols, y_cols, exclude_brain=exclude_brain)
    
    # Box plots for Allometric Exponent
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot([slopes_m, slopes_f], labels=['M', 'F'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'pink']):
        patch.set_facecolor(color)
    ax.set_ylabel('Allometric Exponent')
    ax.set_title('Allometric Exponent by Sex')
    for i, group in enumerate([slopes_m, slopes_f], start=1):
        ax.plot([i], [np.mean(group)], 'o', color='red', markersize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_slopes.png"))
    plt.close()
    
    # Box plots for R²
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot([r2_m, r2_f], labels=['M', 'F'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'pink']):
        patch.set_facecolor(color)
    ax.set_ylabel('R²')
    ax.set_title('R² by Sex')
    for i, group in enumerate([r2_m, r2_f], start=1):
        ax.plot([i], [np.mean(group)], 'o', color='red', markersize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_r2.png"))
    plt.close()


def plot_t_test_boxplot(slopes_f, slopes_m, r2_f, r2_m, exclude_brain, filename, output_dir):
    """
    Create and save box plots for t-test comparisons between females and males.
    """
    labels = ['F', 'M']
    title_suffix = ' (Excluding Brain)' if exclude_brain else ' (Including Brain)'
    # For slopes
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([slopes_f, slopes_m], labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['pink', 'lightblue']):
        patch.set_facecolor(color)
    t_stat_slope, p_value_slope = ttest_ind(slopes_f, slopes_m, equal_var=False)
    ax.set_ylabel('Allometric Exponent')
    ax.set_xlabel('Sex')
    ax.set_title(f'Slopes: t = {t_stat_slope:.3f}, p = {p_value_slope:.2e}{title_suffix}')
    ax.plot([1], [np.mean(slopes_f)], 'o', color='red', markersize=8)
    ax.plot([2], [np.mean(slopes_m)], 'o', color='red', markersize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_slopes.png"))
    plt.close()
    
    # For R²
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([r2_f, r2_m], labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['pink', 'lightblue']):
        patch.set_facecolor(color)
    t_stat_r2, p_value_r2 = ttest_ind(r2_f, r2_m, equal_var=False)
    ax.set_ylabel('R²')
    ax.set_xlabel('Sex')
    ax.set_title(f'R²: t = {t_stat_r2:.3f}, p = {p_value_r2:.2e}{title_suffix}')
    ax.plot([1], [np.mean(r2_f)], 'o', color='red', markersize=8)
    ax.plot([2], [np.mean(r2_m)], 'o', color='red', markersize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_r2.png"))
    plt.close()


def format_p_value(p_value):
    """
    Format the p-value using scientific notation if less than 0.001.
    """
    return f"{p_value:.3e}" if p_value < 0.001 else f"{p_value:.3f}"


def plot_age_correlation_scatter(age, slopes, r2_vals, sex, exclude_brain, filename, output_dir):
    """
    Plot scatter plots for age correlation with allometric exponent and R², for all subjects and by sex.
    """
    # Overall correlation for slopes
    corr_slope, p_slope = pearsonr(age, slopes)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(age, slopes, color='gray', label='All Data Points')
    coeff, intercept = np.polyfit(age, slopes, 1)
    ax.plot(age, coeff * age + intercept,
            color='black', label=f'Fit (r={corr_slope:.3f}, p={format_p_value(p_slope)})')
    title_suffix = ' (Excluding Brain)' if exclude_brain else ' (Including Brain)'
    ax.set_title('Age vs Allometric Exponent' + title_suffix)
    ax.set_xlabel('Age')
    ax.set_ylabel('Allometric Exponent')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_slopes_all_data.png"))
    plt.close()

    # By sex for slopes
    age_male = age[sex == 'M']
    slopes_male = np.array(slopes)[sex == 'M']
    age_female = age[sex == 'F']
    slopes_female = np.array(slopes)[sex == 'F']
    corr_male, p_male = pearsonr(age_male, slopes_male)
    corr_female, p_female = pearsonr(age_female, slopes_female)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(age_male, slopes_male, color='blue', label=f'Male (r={corr_male:.3f}, p={format_p_value(p_male)})')
    ax.scatter(age_female, slopes_female, color='red', label=f'Female (r={corr_female:.3f}, p={format_p_value(p_female)})')
    coeff_m, inter_m = np.polyfit(age_male, slopes_male, 1)
    ax.plot(age_male, coeff_m * age_male + inter_m, color='blue', label='Fit (Male)')
    coeff_f, inter_f = np.polyfit(age_female, slopes_female, 1)
    ax.plot(age_female, coeff_f * age_female + inter_f, color='red', label='Fit (Female)')
    ax.set_title('Age vs Allometric Exponent by Sex' + title_suffix)
    ax.set_xlabel('Age')
    ax.set_ylabel('Allometric Exponent')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_slopes_by_sex.png"))
    plt.close()

    # Overall correlation for R²
    corr_r2, p_r2 = pearsonr(age, r2_vals)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(age, r2_vals, color='gray', label='All Data Points')
    coeff, inter = np.polyfit(age, r2_vals, 1)
    ax.plot(age, coeff * age + inter,
            color='black', label=f'Fit (r={corr_r2:.3f}, p={format_p_value(p_r2)})')
    ax.set_title('Age vs R²' + title_suffix)
    ax.set_xlabel('Age')
    ax.set_ylabel('R²')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_r2_all_data.png"))
    plt.close()

    # By sex for R²
    r2_male = np.array(r2_vals)[sex == 'M']
    r2_female = np.array(r2_vals)[sex == 'F']
    corr_r2_m, p_r2_m = pearsonr(age_male, r2_male)
    corr_r2_f, p_r2_f = pearsonr(age_female, r2_female)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(age_male, r2_male, color='blue', label=f'Male (r={corr_r2_m:.3f}, p={format_p_value(p_r2_m)})')
    ax.scatter(age_female, r2_female, color='red', label=f'Female (r={corr_r2_f:.3f}, p={format_p_value(p_r2_f)})')
    coeff_m2, inter_m2 = np.polyfit(age_male, r2_male, 1)
    ax.plot(age_male, coeff_m2 * age_male + inter_m2, color='blue', label='Fit (Male)')
    coeff_f2, inter_f2 = np.polyfit(age_female, r2_female, 1)
    ax.plot(age_female, coeff_f2 * age_female + inter_f2, color='red', label='Fit (Female)')
    ax.set_title('Age vs R² by Sex' + title_suffix)
    ax.set_xlabel('Age')
    ax.set_ylabel('R²')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_r2_by_sex.png"))
    plt.close()


# ===============================
# New Functions to Save Excel Reports
# ===============================
def save_subject_results(df, x_cols, y_cols, exclude_brain, output_file):
    """
    Iterates over subjects, fits an OLS model for each, and calculates:
        - R², Slope, MSE, ASE and the Model Equation.
    Saves a summary Excel file with columns:
        Subject_ID, Study_ID, R², Allometric Exponent (Slope), MSE, ASE, Model Equation.
    """
    results = []
    for idx, row in df.iterrows():
        subject_id = idx
        # Use Study_ID from df if available; otherwise default to subject_id.
        study_id = row["Study_ID"] if "Study_ID" in df.columns else subject_id
        x_vals = row[x_cols].astype(float).values
        y_vals = row[y_cols].astype(float).values
        if exclude_brain:
            x_vals = x_vals[1:]
            y_vals = y_vals[1:]
        model = fit_ols_model(x_vals, y_vals)
        if model is None:
            r2_val = np.nan
            slope_val = np.nan
            mse = np.nan
            ase = np.nan
            model_eq = ""
        else:
            intercept = model.params[0]
            slope_val = model.params[1]
            predictions = intercept + slope_val * x_vals
            mse = np.mean((y_vals - predictions) ** 2)
            ase = model.bse[1]
            r2_val = model.rsquared
            model_eq = f"y = {intercept:.3f} + {slope_val:.3f} * x"
        results.append({
            "Subject_ID": subject_id,
            "Study_ID": study_id,
            "R²": r2_val,
            "Allometric Exponent (Slope)": slope_val,
            "MSE": mse,
            "ASE": ase,
            "Model Equation": model_eq
        })
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_file, index=False)
    return results_df


def save_sex_significance_summary(df, x_cols, y_cols, exclude_brain, output_file):
    """
    Computes and saves an Excel file summarizing the significance of sex differences on the allometric exponent and R².
    The file includes:
        Metric, Mean_F, Std_F, Mean_M, Std_M, t_stat, p_value.
    """
    df_m = df[df['Sex'] == 'M']
    df_f = df[df['Sex'] == 'F']
    _, _, _, _, _, slopes_m, r2_m = calculate_average_model_parameters(df_m, x_cols, y_cols, exclude_brain=exclude_brain)
    _, _, _, _, _, slopes_f, r2_f = calculate_average_model_parameters(df_f, x_cols, y_cols, exclude_brain=exclude_brain)
    t_stat_slope, p_value_slope = ttest_ind(slopes_m, slopes_f, equal_var=False)
    t_stat_r2, p_value_r2 = ttest_ind(r2_m, r2_f, equal_var=False)
    summary = [
        {
            "Metric": "Allometric Exponent (Slope)",
            "Mean_F": np.mean(slopes_f),
            "Std_F": np.std(slopes_f),
            "Mean_M": np.mean(slopes_m),
            "Std_M": np.std(slopes_m),
            "t_stat": t_stat_slope,
            "p_value": p_value_slope
        },
        {
            "Metric": "R²",
            "Mean_F": np.mean(r2_f),
            "Std_F": np.std(r2_f),
            "Mean_M": np.mean(r2_m),
            "Std_M": np.std(r2_m),
            "t_stat": t_stat_r2,
            "p_value": p_value_r2
        }
    ]
    summary_df = pd.DataFrame(summary)
    summary_df.to_excel(output_file, index=False)
    return summary_df


def save_age_significance_summary(df, x_cols, y_cols, exclude_brain, output_file):
    """
    Computes and saves an Excel summary of age effects on the allometric exponent and R².
    For each metric, it provides Pearson correlations (and p-values) for:
        overall, male, and female subjects.
    """
    # Overall
    _, _, _, _, _, slopes_all, r2_all = calculate_average_model_parameters(df, x_cols, y_cols, exclude_brain=exclude_brain)
    age_all = df['Age'].astype(float).values
    corr_slope_all, p_slope_all = pearsonr(age_all, slopes_all)
    corr_r2_all, p_r2_all = pearsonr(age_all, r2_all)
    # Males
    df_m = df[df['Sex'] == 'M']
    _, _, _, _, _, slopes_m, r2_m = calculate_average_model_parameters(df_m, x_cols, y_cols, exclude_brain=exclude_brain)
    age_m = df_m['Age'].astype(float).values
    corr_slope_m, p_slope_m = pearsonr(age_m, slopes_m)
    corr_r2_m, p_r2_m = pearsonr(age_m, r2_m)
    # Females
    df_f = df[df['Sex'] == 'F']
    _, _, _, _, _, slopes_f, r2_f = calculate_average_model_parameters(df_f, x_cols, y_cols, exclude_brain=exclude_brain)
    age_f = df_f['Age'].astype(float).values
    corr_slope_f, p_slope_f = pearsonr(age_f, slopes_f)
    corr_r2_f, p_r2_f = pearsonr(age_f, r2_f)
    summary = [
        {
            "Metric": "Allometric Exponent (Slope)",
            "Corr_All": corr_slope_all,
            "p_value_All": p_slope_all,
            "Corr_Male": corr_slope_m,
            "p_value_Male": p_slope_m,
            "Corr_Female": corr_slope_f,
            "p_value_Female": p_slope_f,
        },
        {
            "Metric": "R²",
            "Corr_All": corr_r2_all,
            "p_value_All": p_r2_all,
            "Corr_Male": corr_r2_m,
            "p_value_Male": p_r2_m,
            "Corr_Female": corr_r2_f,
            "p_value_Female": p_r2_f,
        }
    ]
    summary_df = pd.DataFrame(summary)
    summary_df.to_excel(output_file, index=False)
    return summary_df


# ===============================
# Main Execution Flow
# ===============================
def main():
    # --- Data source selection based on user input ---
    # If the user enters 'y': use CSV_325 and exclude brain data.
    # Otherwise: use CSV_51 and include brain data.
    user_choice = input("Do you want to exclude brain data? [y/n]: ").strip().lower()
    if user_choice == 'y':
        csv_file = CSV_325
        exclude_brain = True
    else:
        csv_file = CSV_51
        exclude_brain = False

    # Load the chosen CSV file.
    df = pd.read_csv(csv_file)

    # Create output directories for plots and significance results.
    output_plots = PLOTS_SUBJECT_LINEAR
    os.makedirs(output_plots, exist_ok=True)
    os.makedirs(PLOTS_SIGNIFICANCE, exist_ok=True)
    os.makedirs(RESULTS_SIGNIFICANCE, exist_ok=True)
    
    # Define an extended x-range for regression line plotting.
    extended_x_range = np.linspace(0, 10, 100)
    
    # ----- Analysis and Plotting -----
    if exclude_brain:
        # Analysis using CSV_325 with brain excluded.
        (avg_intercept, avg_slope, avg_r2, std_r2,
         intercepts, slopes, r2_vals) = calculate_average_model_parameters(df, x_columns, y_columns, exclude_brain=True)
        plot_regression_lines(df, x_columns, y_columns, avg_intercept, avg_slope,
                              intercepts, slopes, r2_vals, std_r2, extended_x_range,
                              title='OLS Regression Lines (Excluding Brain Data)',
                              filename="regression_lines_excluding_brain.png",
                              output_dir=output_plots,
                              exclude_brain=True)
        plot_box_plots([slopes], [r2_vals], ['Excluding Brain'],
                       filename="box_plots_excluding_brain.png",
                       output_dir=output_plots)
        plot_sex_comparison_box_plots(df, x_columns, y_columns, exclude_brain=True,
                                      filename="sex_comparison_excluding_brain", output_dir=output_plots)
        plot_t_test_boxplot(*calculate_average_model_parameters(df[df['Sex'] == 'F'], x_columns, y_columns, exclude_brain=True)[-2:],
                            *calculate_average_model_parameters(df[df['Sex'] == 'M'], x_columns, y_columns, exclude_brain=True)[-2:],
                            exclude_brain, filename="t_test_sex_effect_excluding_brain", output_dir=output_plots)
        plot_age_correlation_scatter(df['Age'].astype(float).values, slopes, r2_vals, df['Sex'].values,
                                     True, filename="age_correlation_excluding_brain", output_dir=output_plots)
    else:
        # Analysis using CSV_51 with brain included.
        (avg_intercept, avg_slope, avg_r2, std_r2,
         intercepts, slopes, r2_vals) = calculate_average_model_parameters(df, x_columns, y_columns, exclude_brain=False)
        plot_regression_lines(df, x_columns, y_columns, avg_intercept, avg_slope,
                              intercepts, slopes, r2_vals, std_r2, extended_x_range,
                              title='OLS Regression Lines (Including Brain Data)',
                              filename="regression_lines_including_brain.png",
                              output_dir=output_plots,
                              exclude_brain=False)
        plot_box_plots([slopes], [r2_vals], ['Including Brain'],
                       filename="box_plots_including_brain.png",
                       output_dir=output_plots)
        plot_sex_comparison_box_plots(df, x_columns, y_columns, exclude_brain=False,
                                      filename="sex_comparison_including_brain", output_dir=output_plots)
        plot_t_test_boxplot(*calculate_average_model_parameters(df[df['Sex'] == 'F'], x_columns, y_columns, exclude_brain=False)[-2:],
                            *calculate_average_model_parameters(df[df['Sex'] == 'M'], x_columns, y_columns, exclude_brain=False)[-2:],
                            exclude_brain, filename="t_test_sex_effect_including_brain", output_dir=output_plots)
        plot_age_correlation_scatter(df['Age'].astype(float).values, slopes, r2_vals, df['Sex'].values,
                                     False, filename="age_correlation_including_brain", output_dir=output_plots)
    
    # ----- Save Excel Reports -----
    excel_results_file = os.path.join(RESULTS_SIGNIFICANCE, "subject_results.xlsx")
    save_subject_results(df, x_columns, y_columns, exclude_brain, excel_results_file)
    
    excel_sex_file = os.path.join(RESULTS_SIGNIFICANCE, "sex_significance.xlsx")
    save_sex_significance_summary(df, x_columns, y_columns, exclude_brain, excel_sex_file)
    
    excel_age_file = os.path.join(RESULTS_SIGNIFICANCE, "age_significance.xlsx")
    save_age_significance_summary(df, x_columns, y_columns, exclude_brain, excel_age_file)
    
    print("All analysis plots and Excel summary files have been generated.")


if __name__ == '__main__':
    main()
