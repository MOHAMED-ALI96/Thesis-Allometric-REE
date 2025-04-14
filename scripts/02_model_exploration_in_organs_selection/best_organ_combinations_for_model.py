"""
Module for generating model combinations, computing allometric exponents,
and visualizing the best models. This version handles the cases where the
brain is included and excluded separately by loading different input CSV files.
"""

import os
import sys
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Import config paths ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from scripts.utils.puplic.config_paths import (
    CSV_ALL_325, CSV_ALL_51,
    PLOTS_MODEL_SELECTION, RESULTS_MODEL_SELECTION
)

# Fixed base organs and unique organs
BASE_ORGANS = ['Brain', 'Heart', 'Liver', 'Kidneys']
UNIQUE_ORGANS = [
    'Muscles', 'Fat', 'Gastrointestinal_Tract', 'Pancreas', 'Spinal_cord',
    'Spleen', 'Endocrine', 'Skin', 'Bone', 'Blood_Vessels', 'Respiratory'
]

# Abbreviations for organs
ORGAN_ABBREVIATIONS = {
    'Brain': 'Br', 'Heart': 'Ht', 'Liver': 'Lv', 'Kidneys': 'Kd',
    'Muscles': 'Ms', 'Fat': 'Ft', 'Gastrointestinal_Tract': 'GI', 'Pancreas': 'Pc',
    'Spinal_cord': 'Sc', 'Spleen': 'Sp', 'Endocrine': 'Ed', 'Skin': 'Sk',
    'Bone': 'Bn', 'Blood_Vessels': 'BV', 'Respiratory': 'Rs'
}

# Output directories using pathlib
OUTPUT_PLOT_DIR = Path(PLOTS_MODEL_SELECTION) / 'model_selection'
OUTPUT_EXCEL_DIR = Path(RESULTS_MODEL_SELECTION) / 'model_selection'
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_EXCEL_DIR.mkdir(parents=True, exist_ok=True)


def generate_model_combinations(base_organs: list, unique_organs: list) -> list:
    """
    Generate all model combinations, starting with the base organs and then
    adding every combination of the unique organs.
    """
    models = [{'organs_names': base_organs}]
    for r in tqdm(range(1, len(unique_organs) + 1), desc="Generating Model Combinations"):
        for combo in itertools.combinations(unique_organs, r):
            model_organs = base_organs + list(combo)
            models.append({'organs_names': model_organs})
    return models


def generate_model_combinations_exclude(base_organs: list, unique_organs: list, exclude_organ: str) -> list:
    """
    Generate model combinations excluding a specified organ.
    """
    modified_base = [organ for organ in base_organs if organ != exclude_organ]
    modified_unique = [organ for organ in unique_organs if organ != exclude_organ]
    return generate_model_combinations(modified_base, modified_unique)


def fit_linear_model(x_values: pd.Series, y_values: pd.Series):
    """
    Fit a linear model using OLS. Returns the fitted model or None if fitting is not feasible.
    """
    if len(x_values) < 3:
        return None
    X = sm.add_constant(x_values)
    try:
        model = sm.OLS(y_values, X).fit()
        return model
    except np.linalg.LinAlgError as e:
        logging.warning(f"Linear algebra error during model fitting: {e}")
        return None


def get_organ_columns(organs_names: list) -> tuple:
    """
    Return the x and y column names corresponding to the given organs.
    """
    x_columns = [f"Log_V_bmi({organ})" for organ in organs_names]
    y_columns = [f"Log_SUV_A({organ})" for organ in organs_names]
    return x_columns, y_columns


def calculate_allometric_exponents(df: pd.DataFrame, organs_names: list) -> tuple:
    """
    Calculate allometric exponents for the specified organs from the DataFrame.
    For each subject (row), a linear regression is performed using the columns
    corresponding to the organs. Both slope (allometric exponent) and intercept are
    recorded, along with the model’s R².
    
    Returns a tuple containing:
        - slopes: List of slope values.
        - avg_slope: Average slope.
        - std_slope: Standard deviation of slopes.
        - intercepts: List of intercept values.
        - avg_intercept: Average intercept.
        - std_intercept: Standard deviation of intercepts.
        - avg_r2: Average R².
        - std_r2: Standard deviation of R².
        - n_valid: Number of processed studies (rows with a valid regression).
    """
    x_cols, y_cols = get_organ_columns(organs_names)
    df_x = df[x_cols]
    df_y = df[y_cols]

    slopes = []
    intercepts = []
    r_squared_values = []
    for i, row in df_x.iterrows():
        x_values = pd.to_numeric(row, errors='coerce').dropna().reset_index(drop=True)
        y_values = pd.to_numeric(df_y.iloc[i], errors='coerce').dropna().reset_index(drop=True)
        if len(x_values) < 3 or len(y_values) < 3:
            continue

        model = fit_linear_model(x_values, y_values)
        if model is not None:
            slope = round(model.params.iloc[1], 2)
            intercept = round(model.params.iloc[0], 2)
            slopes.append(slope)
            intercepts.append(intercept)
            r_squared_values.append(model.rsquared)

    avg_slope = np.mean(slopes) if slopes else None
    std_slope = np.std(slopes) if slopes else None
    avg_intercept = np.mean(intercepts) if intercepts else None
    std_intercept = np.std(intercepts) if intercepts else None
    avg_r2 = np.mean(r_squared_values) if r_squared_values else None
    std_r2 = np.std(r_squared_values) if r_squared_values else None
    n_valid = len(slopes)

    return slopes, avg_slope, std_slope, intercepts, avg_intercept, std_intercept, avg_r2, std_r2, n_valid



def process_single_model(model: dict, df: pd.DataFrame):
    """
    Process a single model configuration by calculating its allometric exponents.
    """
    organs_names = model['organs_names']
    return calculate_allometric_exponents(df, organs_names)


def process_models_in_parallel(df: pd.DataFrame, models: list) -> list:
    """
    Process multiple models in parallel and collect results.
    Each result now contains 9 regression metrics and the model label.
    """
    models_results = []
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_single_model, models, [df] * len(models)),
                total=len(models),
                desc="Processing Models in Parallel"
            )
        )
    for i, result in enumerate(results):
        # Append a label for later use in Excel and plotting.
        models_results.append(result + (f'Model {i+1}',))
    return models_results

def save_models_to_excel(models_results: list, models: list, filename: str):
    """
    Save the model results to an Excel file including the model equation and the number
    of processed studies (N).
    The columns include: Model Number, Organs Included, N, Avg Intercept, SD Intercept,
    Avg Slope, SD Slope, Avg R², SD R², and the full Equation.
    """
    data = []
    for i, result in enumerate(models_results):
        # Unpack the tuple returned from calculate_allometric_exponents
        (slopes, avg_slope, std_slope,
         intercepts, avg_intercept, std_intercept,
         avg_r2, std_r2, n_valid, label) = result  # note: we've appended the label later
        
        organs = models[i]['organs_names']
        organs_str = ', '.join(organs)
        # Build a full model equation string (using average intercept and slope)
        equation = f"Log_SUV_A = {avg_intercept:.2f} + {avg_slope:.2f} * Log_V_bmi"
        data.append([i + 1, organs_str, n_valid, avg_intercept, std_intercept,
                     avg_slope, std_slope, avg_r2, std_r2, equation])

    df_results = pd.DataFrame(data, columns=[
        'Model Number', 'Organs Included', 'N',
        'Avg Intercept', 'SD Intercept',
        'Avg Slope', 'SD Slope',
        'Avg R²', 'SD R²',
        'Equation'
    ])
    df_results.to_excel(filename, index=False)
    logging.info(f"Results saved to Excel at {filename}")


def plot_best_20_models(models_results: list, model_colors: list, models: list,
                        num_subjects: int, title_suffix: str = "", filename: str = None):
    """
    Create and save a boxplot for the 20 best models based on R² along with annotations.
    Here, models_results is expected to be a list of tuples with 10 elements:
    (slopes, avg_slope, std_slope, intercepts, avg_intercept, std_intercept, avg_r2, std_r2, n_valid, label)
    """
    # Select the 20 best models based on avg R² (position 6 of the tuple)
    selected = sorted(models_results, key=lambda x: x[6] if x[6] is not None else -1, reverse=True)[:20]
    plt.figure(figsize=(16, 10))
    
    combined_data = []
    for slopes, avg_slope, std_slope, intercepts, avg_intercept, std_intercept, avg_r2, std_r2, n_valid, label in selected:
        df_model = pd.DataFrame({'Allometric Exponent': slopes, 'Model': label})
        combined_data.append(df_model)
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    sns.boxplot(x='Model', y='Allometric Exponent', data=combined_df, palette=model_colors[:20])
    plt.xticks(rotation=90, fontsize=8)
    plt.title(f'Best 20 Models {title_suffix}\nN={num_subjects}')
    plt.ylabel('Allometric Exponent')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Annotate each box with the average R² and its standard deviation
    for i, (slopes, avg_slope, std_slope, intercepts, avg_intercept, std_intercept, avg_r2, std_r2, n_valid, label) in enumerate(selected):
        text_label = f'R²: {avg_r2:.2f} ± {std_r2:.2f}\nN: {n_valid}'
        median_value = combined_df[combined_df['Model'] == label]['Allometric Exponent'].median()
        plt.text(i, median_value + 0.05, text_label, fontsize=8, ha='center', va='center',
                 rotation=90, bbox=dict(facecolor='white', alpha=0.7))
    
    if filename:
        plt.savefig(filename)
        logging.info(f"Best 20 models plot saved at {filename}")
    # plt.close()


def plot_top_5_models_with_abbreviations(models_results: list, model_colors: list, models: list,
                                         num_subjects: int, title_suffix: str = "", filename: str = None):
    """
    Create and save a boxplot for the top 5 models with abbreviated organ labels and legends.
    Each result tuple is expected to have 10 elements:
    (slopes, avg_slope, std_slope, intercepts, avg_intercept, std_intercept, avg_r2, std_r2, n_valid, label)
    """
    selected = sorted(models_results, key=lambda x: x[6] if x[6] is not None else -1, reverse=True)[:5]
    plt.figure(figsize=(10, 6))
    
    combined_data = []
    for slopes, avg_slope, std_slope, intercepts, avg_intercept, std_intercept, avg_r2, std_r2, n_valid, label in selected:
        df_model = pd.DataFrame({'Allometric Exponent': slopes, 'Model': label})
        combined_data.append(df_model)
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    sns.boxplot(x='Model', y='Allometric Exponent', data=combined_df, palette=model_colors[:5])
    plt.xticks(rotation=45, fontsize=9)
    plt.title(f'Top 5 Models {title_suffix}\nN={num_subjects}')
    plt.ylabel('Allometric Exponent')
    
    for i, (slopes, avg_slope, std_slope, intercepts, avg_intercept, std_intercept, avg_r2, std_r2, n_valid, label) in enumerate(selected):
        model_index = models_results.index(selected[i])
        organs_in_model = models[model_index]['organs_names']
        abbreviated_organs = ', '.join([ORGAN_ABBREVIATIONS.get(organ, organ) for organ in organs_in_model])
        median_value = combined_df[combined_df['Model'] == label]['Allometric Exponent'].median()
        plt.text(i, median_value, abbreviated_organs, fontsize=9, ha='center', va='center',
                 rotation=45, bbox=dict(facecolor='white', alpha=0.5))
    
    legend_labels = [
        f'{label} (Avg Slope: {avg_slope:.3f}, SD Slope: {std_slope:.3f}, R²: {avg_r2:.3f}, SD R²: {std_r2:.3f})'
        for slopes, avg_slope, std_slope, intercepts, avg_intercept, std_intercept, avg_r2, std_r2, n_valid, label in selected
    ]
    plt.legend(handles=[plt.Line2D([0], [0], color=model_colors[i], lw=4) for i in range(5)],
               labels=legend_labels, title="Models", loc='upper right')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if filename:
        plt.savefig(filename)
        logging.info(f"Top 5 models plot saved at {filename}")
    plt.close()



def run_two_cases():
    """
    Run the two processing cases:
      1. With Brain (loads CSV_ALL_51)
      2. Without Brain (loads CSV_ALL_325)
    """
    # -------- Case 1: With Brain --------
    tqdm.write("Processing case: With Brain")
    df_with = pd.read_csv(CSV_ALL_51)
    num_subjects_with = df_with.shape[0]
    models_with_brain = generate_model_combinations(BASE_ORGANS, UNIQUE_ORGANS)
    results_with_brain = process_models_in_parallel(df_with, models_with_brain)
    excel_filename_with = str(OUTPUT_EXCEL_DIR / 'with_Brain.xlsx')
    save_models_to_excel(results_with_brain, models_with_brain, excel_filename_with)
    plot_best_20_models(
        results_with_brain,
        sns.color_palette("tab10", 20),
        models_with_brain,
        num_subjects_with,
        title_suffix="(With Brain)",
        filename=str(OUTPUT_PLOT_DIR / "best_20_with_Brain.png")
    )
    plot_top_5_models_with_abbreviations(
        results_with_brain,
        sns.color_palette("tab10", 5),
        models_with_brain,
        num_subjects_with,
        title_suffix="(With Brain)",
        filename=str(OUTPUT_PLOT_DIR / "top_5_with_Brain.png")
    )

    # -------- Case 2: Without Brain --------
    tqdm.write("Processing case: Without Brain")
    df_without = pd.read_csv(CSV_ALL_325)
    num_subjects_without = df_without.shape[0]
    models_without_brain = generate_model_combinations_exclude(BASE_ORGANS, UNIQUE_ORGANS, 'Brain')
    results_without_brain = process_models_in_parallel(df_without, models_without_brain)
    excel_filename_without = str(OUTPUT_EXCEL_DIR / 'without_Brain.xlsx')
    save_models_to_excel(results_without_brain, models_without_brain, excel_filename_without)
    plot_best_20_models(
        results_without_brain,
        sns.color_palette("tab10", 20),
        models_without_brain,
        num_subjects_without,
        title_suffix="(Without Brain)",
        filename=str(OUTPUT_PLOT_DIR / "best_20_without_Brain.png")
    )
    plot_top_5_models_with_abbreviations(
        results_without_brain,
        sns.color_palette("tab10", 5),
        models_without_brain,
        num_subjects_without,
        title_suffix="(Without Brain)",
        filename=str(OUTPUT_PLOT_DIR / "top_5_without_Brain.png")
    )


if __name__ == "__main__":
    run_two_cases()
