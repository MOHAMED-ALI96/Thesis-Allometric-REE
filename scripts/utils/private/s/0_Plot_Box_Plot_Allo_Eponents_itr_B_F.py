import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import itertools
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Load CSV file
CSV_PATH_95 = "E:\Project\CSV\data all_corrected\collected_metrics_Full_Brain_51_sum_t.csv"
df_95 = pd.read_csv(CSV_PATH_95)

# Fixed base organs
base_organs = ['Brain', 'Heart', 'Liver', 'Kidneys']

# Updated unique organs
unique_organs = [
    'Muscles', 'Fat', 'Gastrointestinal_Tract', 'Pancreas', 'Spinal_cord',
    'Spleen', 'Endocrine', 'Skin', 'Bone', 'Blood_Vessels', 'Respiratory'
    #, 'Eyes' , 'Gallbladder'
]

# Abbreviations for organs
organ_abbreviations = {
    'Brain': 'Br', 'Heart': 'Ht', 'Liver': 'Lv', 'Kidneys': 'Kd', 'Muscles': 'Ms', 'Fat': 'Ft',
    'Gastrointestinal_Tract': 'GI', 'Pancreas': 'Pc', 'Spinal_cord': 'Sc',
    'Spleen': 'Sp', 'Endocrine': 'Ed', 'Skin': 'Sk', 'Bone': 'Bn', 'Blood_Vessels': 'BV',
    'Respiratory': 'Rs',
   # 'Eyes': 'Ey', 'Gallbladder': 'GB'
}

# Output path for saving plots and Excel files
output_plot_path = r"E:\1Plots\plots_feb_24\Plot_Box_Plot_Allo_Eponents_itr_B_F"
if not os.path.exists(output_plot_path):
    os.makedirs(output_plot_path)


# Generate all combinations of unique organs with the base organs
def generate_model_combinations(base_organs, unique_organs):
    models = []
    models.append({'organs_names': base_organs})  # Add base model

    for r in tqdm(range(1, len(unique_organs) + 1), desc="Generating Model Combinations"):
        for combo in itertools.combinations(unique_organs, r):
            model_organs = base_organs + list(combo)
            models.append({'organs_names': model_organs})
    return models


# Exclude an organ from the base and unique organs
def generate_model_combinations_exclude(base_organs, unique_organs, exclude_organ):
    modified_base = [organ for organ in base_organs if organ != exclude_organ]
    modified_unique = [organ for organ in unique_organs if organ != exclude_organ]
    return generate_model_combinations(modified_base, modified_unique)


# Fit the linear model
def fit_linear_model(x_values, y_values):
    if len(x_values) < 3:  # Changed from 4 to 3
        return None
    X = sm.add_constant(x_values)
    try:
        model = sm.OLS(y_values, X).fit()
        return model
    except np.linalg.LinAlgError as e:
        return None


def get_organ_columns(df, organs_names):
    x_columns = [f"Log_V({organ})" for organ in organs_names]
    y_columns = [f"Log_SUV_A({organ})" for organ in organs_names]
    return x_columns, y_columns


def calculate_allometric_exponents(df, organs_names):
    x_columns, y_columns = get_organ_columns(df, organs_names)
    df_x = df[x_columns]
    df_y = df[y_columns]

    results_list = []
    r_squared_list = []
    for i, row in df_x.iterrows():
        x_values = pd.to_numeric(row, errors='coerce').dropna().reset_index(drop=True)
        y_values = pd.to_numeric(df_y.iloc[i], errors='coerce').dropna().reset_index(drop=True)
        if len(x_values) < 3 or len(y_values) < 3:
            continue

        model = fit_linear_model(x_values, y_values)
        if model is not None:
            coef_x_series = [round(coef, 2) for coef in model.params[1:]]
            results_list.append(coef_x_series[0])
            r_squared_list.append(model.rsquared)

    avg_allometric_exponent = np.mean(results_list) if results_list else None
    std_dev = np.std(results_list) if results_list else None
    avg_r_squared = np.mean(r_squared_list) if r_squared_list else None
    std_dev_r_squared = np.std(r_squared_list) if r_squared_list else None
    return results_list, avg_allometric_exponent, std_dev, avg_r_squared, std_dev_r_squared


# Process models in parallel without using lambda function
def process_single_model(model, df):
    organs_names = model['organs_names']
    return calculate_allometric_exponents(df, organs_names)


def process_models_in_parallel(df, models):
    models_results = []
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_model, models, [df] * len(models)), total=len(models),
                            desc="Processing Models in Parallel"))
        for i, result in enumerate(results):
            allometric_exponents, avg_allometric_exponent, std_dev, avg_r_squared, std_dev_r_squared = result
            models_results.append((allometric_exponents, avg_allometric_exponent, std_dev, avg_r_squared,
                                   std_dev_r_squared, f'Model {i + 1}'))
    return models_results


# Saving Excel files for each case (with equations)
def save_models_to_excel(models_results, models, filename):
    data = []
    for i, (allometric_exponents, avg_exponent, std_dev, avg_r_squared, std_dev_r_squared, _) in enumerate(
            models_results):
        organs = models[i]['organs_names']
        organs_str = ', '.join(organs)
        equation = f"Log_SUV_A = {round(avg_exponent, 2)} * Log_V"  # Equation column
        data.append([i + 1, organs_str, avg_exponent, std_dev, avg_r_squared, std_dev_r_squared, equation])

    df_results = pd.DataFrame(data, columns=[
        'Model Number', 'Organs Included', 'Avg Allometric Exponent', 'SD Allometric Exponent', 'Avg R²', 'SD R²',
        'Equation'
    ])
    df_results.to_excel(filename, index=False)


# Plotting the best 20 models based on R² with avg R² and SD on each box (Vertical Text)
def plot_best_20_models(models_results, model_colors, models, num_subjects, title_suffix="", filename=None):
    models_results_sorted = sorted(models_results, key=lambda x: x[3], reverse=True)[:20]

    plt.figure(figsize=(16, 10))
    data = []
    labels = []
    for i, (exponents, avg_exponent, std_dev, avg_r_squared, std_dev_r_squared, label) in enumerate(models_results_sorted):
        df_model = pd.DataFrame({'Allometric Exponent': exponents, 'Model': label})
        data.append(df_model)
        labels.append(label)

    combined_df = pd.concat(data, ignore_index=True)
    sns.boxplot(x='Model', y='Allometric Exponent', data=combined_df, palette=model_colors[:20])
    plt.xticks(rotation=90, fontsize=8)
    plt.title(f'Best 20 Models {title_suffix}\nN={num_subjects}')
    plt.ylabel('Allometric Exponent')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Place R² and SD on top of each box (with vertical rotation)
    for i, (_, avg_exponent, std_dev, avg_r_squared, std_dev_r_squared, label) in enumerate(models_results_sorted):
        text_label = f'{avg_r_squared:.2f} ± {std_dev_r_squared:.2f}'
        # Get the median value of each boxplot to place the text inside the box
        median_value = combined_df[combined_df['Model'] == label]['Allometric Exponent'].median()
        # Place the text vertically
        plt.text(i, median_value + 0.05, text_label, fontsize=8, ha='center', va='center', rotation=90, bbox=dict(facecolor='white', alpha=0.7))

    if filename:
        plt.savefig(filename)

    #plt.show()



# Plotting the top 5 models based on R² with abbreviated labels and legends
def plot_top_5_models_with_abbreviations(models_results, model_colors, models, num_subjects, title_suffix="", filename=None):
    models_results_sorted = sorted(models_results, key=lambda x: x[3], reverse=True)[:5]

    plt.figure(figsize=(10, 6))
    data = []
    labels = []
    for i, (exponents, avg_exponent, std_dev, avg_r_squared, std_dev_r_squared, label) in enumerate(models_results_sorted):
        df_model = pd.DataFrame({'Allometric Exponent': exponents, 'Model': label})
        data.append(df_model)
        labels.append(label)

    combined_df = pd.concat(data, ignore_index=True)
    sns.boxplot(x='Model', y='Allometric Exponent', data=combined_df, palette=model_colors[:5])
    plt.xticks(rotation=45, fontsize=9)
    plt.title(f'Top 5 Models {title_suffix}\nN={num_subjects}')
    plt.ylabel('Allometric Exponent')

    # Add abbreviated organ labels inside each boxplot
    for i, (_, avg_exponent, std_dev, avg_r_squared, std_dev_r_squared, label) in enumerate(models_results_sorted):
        model_index = models_results.index(models_results_sorted[i])
        organs_in_model = models[model_index]['organs_names']
        abbreviated_organs = ', '.join([organ_abbreviations.get(organ, organ) for organ in organs_in_model])

        # Get the median value of each boxplot to place the text inside the box
        median_value = combined_df[combined_df['Model'] == label]['Allometric Exponent'].median()

        # Place the text on top of the median line or near it
        plt.text(i, median_value, abbreviated_organs, fontsize=9, ha='center', va='center', rotation=45, bbox=dict(facecolor='white', alpha=0.5))

    # Add legends with avg R² and SD values
    legend_labels = [
        f'{label} (Avg: {avg_exponent:.3f}, sd: {std_dev:.3f}, R²: {avg_r_squared:.3f}, R² sd: {std_dev_r_squared:.3f})'
        for _, avg_exponent, std_dev, avg_r_squared, std_dev_r_squared, label in models_results_sorted
    ]
    plt.legend(handles=[plt.Line2D([0], [0], color=model_colors[i], lw=4) for i in range(5)],
               labels=legend_labels, title="Models", loc='upper right')

    plt.grid(True, linestyle='--', alpha=0.6)

    if filename:
        plt.savefig(filename)

    plt.show()


# Run two cases: with Brain and without Brain
def run_two_cases(df, num_subjects):
    # 1. Case with Brain included
    tqdm.write("Processing case: With Brain")
    models_with_brain = generate_model_combinations(base_organs, unique_organs)
    models_results_with_brain = process_models_in_parallel(df, models_with_brain)
    save_models_to_excel(models_results_with_brain, models_with_brain, os.path.join(output_plot_path, 'with_brain.xlsx'))
    plot_best_20_models(models_results_with_brain, sns.color_palette("tab10", 20), models_with_brain, num_subjects, title_suffix="(With Brain)", filename=os.path.join(output_plot_path, "best_20_with_brain.png"))
    plot_top_5_models_with_abbreviations(models_results_with_brain, sns.color_palette("tab10", 5), models_with_brain, num_subjects, title_suffix="(With Brain)", filename=os.path.join(output_plot_path, "top_5_with_brain.png"))

    # 2. Case without Brain
    tqdm.write("Processing case: Without Brain")
    models_without_brain = generate_model_combinations_exclude(base_organs, unique_organs, 'Brain')
    models_results_without_brain = process_models_in_parallel(df, models_without_brain)
    save_models_to_excel(models_results_without_brain, models_without_brain, os.path.join(output_plot_path, 'without_brain.xlsx'))
    plot_best_20_models(models_results_without_brain, sns.color_palette("tab10", 20), models_without_brain, num_subjects, title_suffix="(Without Brain)", filename=os.path.join(output_plot_path, "best_20_without_brain.png"))
    plot_top_5_models_with_abbreviations(models_results_without_brain, sns.color_palette("tab10", 5), models_without_brain, num_subjects, title_suffix="(Without Brain)", filename=os.path.join(output_plot_path, "top_5_without_brain.png"))

if __name__ == "__main__":
    num_subjects = df_95.shape[0]
    run_two_cases(df_95, num_subjects)
