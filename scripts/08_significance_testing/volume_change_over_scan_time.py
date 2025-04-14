import os
import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Set project root and append to sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# Configured output paths
from scripts.utils.puplic.config_paths import PLOTS_SIGNIFICANCE, RESULTS_SIGNIFICANCE

# === User Input for Dataset Selection ===
use_full_brain = input("Do you want the full brain subjects? (y/n): ").strip().lower()
merged_file = (
    r"E:\Project\CSV\data_all_metadata\data_metadata_fullBrain.csv"
    if use_full_brain == 'y'
    else r"E:\Project\CSV\data_all_metadata\data_metadata_all.csv"
)

label_groups_file = r"E:\Project\CSV\updated_label_name.csv"

# === Output Directories ===
output_dir = PLOTS_SIGNIFICANCE / "acquisition_time_scatter"
output_dir.mkdir(parents=True, exist_ok=True)

# === Load Data ===
df = pd.read_csv(merged_file)
label_data = pd.read_csv(label_groups_file)
label_data.columns = label_data.columns.str.strip()

# Ensure time is in total seconds and readable format
df['Min Acquisition Time'] = pd.to_timedelta(df['Min Acquisition Time'])
df['Min Acquisition Time (seconds)'] = df['Min Acquisition Time'].dt.total_seconds()
df['Min Acquisition Time (h:mm:ss)'] = df['Min Acquisition Time'].apply(
    lambda x: str(timedelta(seconds=x.total_seconds()))
)

# === Validation ===
if 'Group' not in label_data.columns:
    raise KeyError("Missing 'Group' column in label group file.")

required_columns = ['Min Acquisition Time (seconds)', 'Age']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

# === Scatter Plot per Organ Group ===
correlation_results = []

for group in label_data['Group'].drop_duplicates():
    organ_col = f"V_bmi({group})"

    if organ_col not in df.columns:
        print(f"⚠️ Skipping {group}: {organ_col} not found.")
        continue

    x = df['Min Acquisition Time (seconds)']
    y = df[organ_col]
    age = df['Age']

    r_val, p_val = pearsonr(x, y)
    correlation_results.append({
        "Organ": group,
        "Pearson r": r_val,
        "p-value": p_val
    })

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=age, cmap='viridis', alpha=0.6, edgecolor='w')
    plt.colorbar(scatter, label='Age')
    sns.regplot(x=x, y=y, scatter=False, color='red')

    plt.legend([f'r = {r_val:.3f}, p = {p_val:.3f}'], loc='best')
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda sec, _: str(timedelta(seconds=sec)))
    )

    plt.title(f'{group}: V_bmi vs Min Acquisition Time')
    plt.xlabel('Min Acquisition Time (h:mm:ss)')
    plt.ylabel(f'V_bmi({group})')
    plt.xticks(rotation=45)

    save_path = output_dir / f"scatter_plot_{group}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {save_path}")

# === Save Summary Excel ===
results_df = pd.DataFrame(correlation_results)
excel_path = RESULTS_SIGNIFICANCE / "acquisition_time_vs_volume_correlation.xlsx"
results_df.to_excel(excel_path, index=False)
print(f"\n📊 Correlation summary saved: {excel_path}")
