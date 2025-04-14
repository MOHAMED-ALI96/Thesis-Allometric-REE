#!/usr/bin/env python
import sys
from pathlib import Path

# 🔧 Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ✅ Import configured paths
from scripts.utils.puplic.config_paths import (
    CSV_51, CSV_325,
    PLOTS_VALIDATION, RESULTS_VALIDATION
)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data from a file."""
    df = pd.read_csv(csv_path)
    return df


def preprocess_data(df: pd.DataFrame, organs: list):
    """
    Preprocess the DataFrame by:
      - Generating predictor and outcome columns for each organ.
      - Filling NaNs in these columns with the median.
      - Dropping rows with missing 'Age' or 'Sex'.
    Returns:
      - The processed DataFrame.
      - x_columns (predictors) and y_columns (outcomes) lists.
    """
    x_columns = [f"Log_V_bmi({organ})" for organ in organs]
    y_columns = [f"Log_SUV_A({organ})" for organ in organs]
    
    # Fill NaNs in each predictor/outcome column with the respective column median
    df[x_columns] = df[x_columns].apply(lambda col: col.fillna(col.median()), axis=0)
    df[y_columns] = df[y_columns].apply(lambda col: col.fillna(col.median()), axis=0)
    
    # Drop rows without Age or Sex information
    df = df.dropna(subset=['Age', 'Sex'])
    
    return df, x_columns, y_columns


def fit_ols_model(x_values: np.ndarray, y_values: np.ndarray):
    """Fit an Ordinary Least Squares model and return the slope and intercept."""
    X = sm.add_constant(x_values)
    model = sm.OLS(y_values, X).fit()
    # model.params[0]: intercept, model.params[1]: slope
    return model.params[1], model.params[0]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate R², Mean Squared Error (MSE), and Average Squared Error (ASE)."""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    ase = np.mean(np.abs(y_true - y_pred))
    return r2, mse, ase


def plot_residuals(fold, y_true, y_pred, r2, mse, ase, output_directory, is_whole_data=False):
    """
    Plot the residuals scatter plot.
      - If is_whole_data=True, the plot title and filename reflect the full dataset.
      - Otherwise, use the fold number.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color='blue')
    plt.axhline(0, color='red', linestyle='--')
    
    if is_whole_data:
        plt.title(f'Whole Data Residuals - R²: {r2:.3f}, MSE: {mse:.3f}, ASE: {ase:.3f}')
        plot_name = 'whole_data_residuals.png'
    else:
        plt.title(f'Fold {fold} Residuals - R²: {r2:.3f}, MSE: {mse:.3f}, ASE: {ase:.3f}')
        plot_name = f'residuals_fold_{fold}.png'
    
    plt.xlabel('Predicted Log(Average SUV)')
    plt.ylabel('Residuals (True - Predicted)')
    plt.legend([f'Residuals', 'Zero Residual Line'], loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, plot_name))
    plt.close()


def plot_qq(residuals, slope, intercept, output_directory, fold=None, is_whole_data=False):
    """Plot a Q-Q plot of residuals to assess normality."""
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    equation_text = f"Log(Average SUV) = {intercept:.3f} + {slope:.3f} * Log(V/BMI)"
    
    if is_whole_data:
        plt.title('Normal Q-Q Plot for Whole Data')
        plot_name = 'qq_plot_whole_data.png'
        plt.legend([f'{equation_text}', 'Assess normality of residuals'], loc='best')
    else:
        plt.title(f'Normal Q-Q Plot for Fold {fold}')
        plot_name = f'qq_plot_fold_{fold}.png'
        plt.legend([f'{equation_text}', 'Assess normality of residuals'], loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, plot_name))
    plt.close()


def plot_residuals_vs_leverage(X, residuals, slope, intercept, output_directory, fold=None, is_whole_data=False):
    """Plot standardized residuals against leverage to detect influential data points."""
    model = sm.OLS(residuals, sm.add_constant(X)).fit()
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    standardized_residuals = influence.resid_studentized_internal
    
    plt.figure(figsize=(8, 6))
    plt.scatter(leverage, standardized_residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Leverage')
    plt.ylabel('Standardized Residuals')
    
    equation_text = f"Log(Average SUV) = {intercept:.3f} + {slope:.3f} * Log(V/BMI)"
    if is_whole_data:
        plt.title('Residuals vs. Leverage for Whole Data')
        plot_name = 'residuals_vs_leverage_whole_data.png'
        plt.legend([f'{equation_text}', 'Detect influential data points'], loc='best')
    else:
        plt.title(f'Residuals vs. Leverage for Fold {fold}')
        plot_name = f'residuals_vs_leverage_fold_{fold}.png'
        plt.legend([f'{equation_text}', 'Detect influential data points'], loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, plot_name))
    plt.close()


def plot_error_distribution(residuals, slope, intercept, output_directory, fold=None, is_whole_data=False):
    """Plot a histogram of the residuals (error distribution)."""
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    
    equation_text = f"Log(Average SUV) = {intercept:.3f} + {slope:.3f} * Log(V/BMI)"
    if is_whole_data:
        plt.title('Error Distribution for Whole Data')
        plot_name = 'error_distribution_whole_data.png'
    else:
        plt.title(f'Error Distribution for Fold {fold}')
        plot_name = f'error_distribution_fold_{fold}.png'
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, plot_name))
    plt.close()


def get_median_slope_fold(fold_results):
    """Return the fold number whose slope is closest to the median slope across folds."""
    slopes = [result[4] for result in fold_results]
    median_slope = np.median(slopes)
    median_fold = min(fold_results, key=lambda x: abs(x[4] - median_slope))[0]
    return median_fold


def get_average_slope_intercept(fold_results):
    """Return the average slope and intercept computed over all folds."""
    slopes = [result[4] for result in fold_results]
    intercepts = [result[5] for result in fold_results]
    return np.mean(slopes), np.mean(intercepts)


def main():
    # Use CSV_325 from configured paths; if not defined, fall back to a hard-coded path.
    csv_file_path = CSV_325 if CSV_325 else r"E:\Thesis-Allometric-REE\data\public\targeted_organs_325.csv"
    
    # Load the dataset
    df = load_data(csv_file_path)
    
    # List of organs (adjust as necessary)
    organs = ['Muscles', 'Liver', 'Kidneys', 'Fat', 'Bone', 'Heart']
    
    # Preprocess the data
    df, x_columns, y_columns = preprocess_data(df, organs)
    
    # Create the output directory for saving plots
    output_directory = os.path.join(PLOTS_VALIDATION, "model_13_4_25_CV_n")
    os.makedirs(output_directory, exist_ok=True)
    
    # Prepare data for regression by flattening:
    # (This pools all organ-specific values into one long vector; verify that this is the intended behavior.)
    X_all = df[x_columns].values.flatten()
    y_all = df[y_columns].values.flatten()
    
    # Set up 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    fold_results = []
    r2_scores, mse_scores, ase_scores = [], [], []
    best_r2, best_mse, best_ase = -float('inf'), float('inf'), float('inf')
    best_fold_r2, best_fold_mse, best_fold_ase = None, None, None
    
    # Iterate over each fold
    for fold, (train_index, test_index) in enumerate(kf.split(X_all), start=1):
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]
    
        # Fit OLS regression model
        slope, intercept = fit_ols_model(X_train, y_train)
        y_pred_test = intercept + slope * X_test
    
        # Compute performance metrics
        r2, mse, ase = calculate_metrics(y_test, y_pred_test)
    
        fold_results.append((fold, r2, mse, ase, slope, intercept))
        r2_scores.append(r2)
        mse_scores.append(mse)
        ase_scores.append(ase)
    
        # Track best fold based on metrics
        if r2 > best_r2:
            best_r2, best_fold_r2 = r2, fold
        if mse < best_mse:
            best_mse, best_fold_mse = mse, fold
        if ase < best_ase:
            best_ase, best_fold_ase = ase, fold
    
        # Plot predicted vs actual for this fold, including the regression equation in the title
        equation_text = f"Log(Average SUV) = {intercept:.3f} + {slope:.3f} * Log(V/BMI)"
        title = f'Fold {fold} - R²: {r2:.3f}, MSE: {mse:.3f}, ASE: {ase:.3f}\nEquation: {equation_text}'
    
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test, y_test, label='True values', alpha=0.5)
        plt.plot(X_test, y_pred_test, label=f'Fold {fold} Predicted', color='red', alpha=0.6)
        plt.title(title)
        plt.xlabel('Log(V/BMI)')
        plt.ylabel('Log(Average SUV)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, f'fold_{fold}.png'))
        plt.close()
    
        # Plot residuals for this fold
        plot_residuals(fold, y_test, y_pred_test, r2, mse, ase, output_directory)
    
    # Use the best fold (based on R²) for the whole dataset regression line
    best_slope, best_intercept = fold_results[best_fold_r2 - 1][4], fold_results[best_fold_r2 - 1][5]
    y_pred_all = best_intercept + best_slope * X_all
    r2_whole, mse_whole, ase_whole = calculate_metrics(y_all, y_pred_all)
    
    # Plot whole dataset with the best fold highlighted
    plt.figure(figsize=(8, 6))
    plt.scatter(X_all, y_all, label='True values (All Data)', alpha=0.5, color='blue')
    
    # Extract best fold test indices
    best_fold_test_indices = list(kf.split(X_all))[best_fold_r2 - 1][1]
    X_test_best = X_all[best_fold_test_indices]
    y_test_best = y_all[best_fold_test_indices]
    
    plt.scatter(X_test_best, y_test_best, label=f'Best Fold {best_fold_r2} Data Points', color='red', alpha=0.8)
    plt.plot(X_all, y_pred_all,
             label=f'Predicted (R²: {r2_whole:.3f}, MSE: {mse_whole:.3f}, ASE: {ase_whole:.3f})',
             color='green', alpha=0.6)
    equation_text = f"Equation: Log(Average SUV) = {best_intercept:.3f} + {best_slope:.3f} * Log(V/BMI)"
    plt.title(f'Best Fold Regression Line on Full Data\n{equation_text}')
    plt.xlabel('Log(V/BMI)')
    plt.ylabel('Log(Average SUV)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'prediction_vs_actual_best_fold_highlighted.png'))
    plt.close()
    
    # Residual analysis for whole dataset
    residuals_all = y_all - y_pred_all
    plot_residuals(fold=None, y_true=y_all, y_pred=y_pred_all, r2=r2_whole, mse=mse_whole, ase=ase_whole,
                   output_directory=output_directory, is_whole_data=True)
    plot_qq(residuals_all, best_slope, best_intercept, output_directory, is_whole_data=True)
    plot_residuals_vs_leverage(X_all, residuals_all, best_slope, best_intercept, output_directory, is_whole_data=True)
    plot_error_distribution(residuals_all, best_slope, best_intercept, output_directory, is_whole_data=True)
    
    # Plot Actual vs. Predicted SUV with best fold data highlighted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_all, y_pred_all, color='blue', alpha=0.5, label='Predicted (All Data)')
    plt.scatter(y_test_best, y_pred_all[best_fold_test_indices], color='red',
                label=f'Best Fold {best_fold_r2} Points', alpha=0.8)
    slope_all, intercept_all = np.polyfit(y_all, y_pred_all, 1)
    regression_line = slope_all * np.array(y_all) + intercept_all
    plt.plot(y_all, regression_line, color='green',
             label=f'Regression Line: y = {slope_all:.3f}x + {intercept_all:.3f}\nR² = {r2_whole:.3f}, MSE = {mse_whole:.3f}')
    plt.xlabel('Actual Log(Average SUV)')
    plt.ylabel('Predicted Log(Average SUV)')
    plt.title(f'Actual vs Predicted SUV - Best Fold {best_fold_r2} Highlighted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'actual_vs_predicted_with_regression_best_fold.png'))
    plt.close()
    
    # Print overall and per-fold performance metrics
    print(f"Overall R² on whole dataset: {r2_whole:.3f}")
    print(f"Overall MSE on whole dataset: {mse_whole:.3f}")
    print(f"Overall ASE on whole dataset: {ase_whole:.3f}")
    print(f"Best fold for R²: {best_fold_r2} with R²: {best_r2:.3f}")
    print(f"Best fold for MSE: {best_fold_mse} with MSE: {best_mse:.3f}")
    print(f"Best fold for ASE: {best_fold_ase} with ASE: {best_ase:.3f}")
    
    for fold, r2, mse, ase, slope, intercept in fold_results:
        print(f"Fold {fold}:")
        print(f"  R²: {r2:.3f}, MSE: {mse:.3f}, ASE: {ase:.3f}")
        print(f"  Equation: Log(Average SUV) = {intercept:.3f} + {slope:.3f} * Log(V/BMI)")
    
    # Calculate and plot median and average regression lines
    median_fold = get_median_slope_fold(fold_results)
    median_slope = fold_results[median_fold - 1][4]
    median_intercept = fold_results[median_fold - 1][5]
    y_pred_median = median_intercept + median_slope * X_all
    r2_median, mse_median, ase_median = calculate_metrics(y_all, y_pred_median)
    
    avg_slope, avg_intercept = get_average_slope_intercept(fold_results)
    y_pred_avg = avg_intercept + avg_slope * X_all
    r2_avg, mse_avg, ase_avg = calculate_metrics(y_all, y_pred_avg)
    
    # Plot for Median Slope Fold
    plt.figure(figsize=(8, 6))
    plt.scatter(X_all, y_all, label='True values (All Data)', alpha=0.5, color='blue')
    plt.scatter(X_test_best, y_test_best, color='red', label=f'Best Fold {best_fold_r2}', alpha=0.8)
    median_test_indices = list(kf.split(X_all))[median_fold - 1][1]
    X_test_median = X_all[median_test_indices]
    y_test_median = y_all[median_test_indices]
    plt.scatter(X_test_median, y_test_median, color='orange', label=f'Median Slope Fold {median_fold}', alpha=0.8)
    plt.plot(X_all, y_pred_median, color='green',
             label=f'Median Fold: R² = {r2_median:.3f}, MSE = {mse_median:.3f}, ASE = {ase_median:.3f}')
    plt.title(f'Median Slope Fold Regression\nEquation: Log(Average SUV) = {median_intercept:.3f} + {median_slope:.3f} * Log(V/BMI)')
    plt.xlabel('Log(V/BMI)')
    plt.ylabel('Log(Average SUV)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'median_slope_fold_regression.png'))
    plt.close()
    
    # Plot for Average Regression Across All Folds
    plt.figure(figsize=(8, 6))
    plt.scatter(X_all, y_all, label='True values (All Data)', alpha=0.5, color='blue')
    plt.scatter(X_test_best, y_test_best, color='red', label=f'Best Fold {best_fold_r2}', alpha=0.8)
    plt.scatter(X_test_median, y_test_median, color='orange', label=f'Median Slope Fold {median_fold}', alpha=0.8)
    plt.plot(X_all, y_pred_avg, color='purple',
             label=f'Average: R² = {r2_avg:.3f}, MSE = {mse_avg:.3f}, ASE = {ase_avg:.3f}')
    plt.title(f'Average Regression Line Across All Folds\nEquation: Log(Average SUV) = {avg_intercept:.3f} + {avg_slope:.3f} * Log(V/BMI)')
    plt.xlabel('Log(V/BMI)')
    plt.ylabel('Log(Average SUV)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'average_folds_regression.png'))
    plt.close()


if __name__ == '__main__':
    main()
