import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from scripts.utils.puplic.config_paths import CSV_ALL_325, PLOTS_CONTRIBUTION, RESULTS_CONTRIBUTION

# --- Configuration ---
selected_organs = [
    'Brain', 'Heart', 'Liver', 'Kidneys', 'Muscles', 'Fat', 'Bone',
    'Gastrointestinal_Tract', 'Pancreas', 'Spinal_cord', 'Spleen',
    'Endocrine', 'Skin', 'Blood_Vessels', 'Respiratory'
]

# Define columns and labels
def format_label(organ):
    return organ.replace('_', ' ')

features_dict = {
    f'SUV_A({organ})': format_label(organ) for organ in selected_organs
}

# Output directories
(PLOTS_CONTRIBUTION / "suv_shares").mkdir(parents=True, exist_ok=True)
(PLOTS_CONTRIBUTION / "volume_shares").mkdir(parents=True, exist_ok=True)
(RESULTS_CONTRIBUTION).mkdir(parents=True, exist_ok=True)

# Read data
df = pd.read_csv(CSV_ALL_325)
organ_stats = []

# Labels without problematic LaTeX backslashes
MAX_SUV = "Max SUV [1]"
MIN_SUV = "Min SUV [1]"
AVG_SUV = "Avg SUV [1]"
MAX_VOL = "Max V [m5/kg]"
MIN_VOL = "Min V [m5/kg]"
AVG_VOL = "Avg V [m5/kg]"

# Analysis Loop
for organ_suv, organ_name in features_dict.items():
    organ = organ_suv.split("(")[1].strip(")")
    vol_col = f"V_bmi({organ})"
    if organ_suv not in df.columns or vol_col not in df.columns:
        continue

    # Exclude zero or missing values
    valid_mask = (df[organ_suv] > 0) & (df[vol_col] > 0)
    sub_df = df.loc[valid_mask]

    suv_vals = sub_df[organ_suv]
    vol_vals = sub_df[vol_col]

    organ_stats.append({
        "Organ Name": organ_name,
        "Number of subjects": len(suv_vals),
        MAX_SUV: suv_vals.max(),
        MIN_SUV: suv_vals.min(),
        AVG_SUV: suv_vals.mean(),
        MAX_VOL: vol_vals.max(),
        MIN_VOL: vol_vals.min(),
        AVG_VOL: vol_vals.mean()
    })

# Save as Excel summary
summary_df = pd.DataFrame(organ_stats)
summary_df.to_excel(RESULTS_CONTRIBUTION / "organ_summary_stats.xlsx", index=False)

# Top 10 by SUV and by Volume
top10_suv = summary_df.sort_values(AVG_SUV, ascending=False).head(10)
top10_vol = summary_df.sort_values(AVG_VOL, ascending=False).head(10)

# Plot: Average SUV bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=top10_suv, x="Organ Name", y=AVG_SUV, palette="Spectral")
plt.xticks(rotation=45, ha='right', fontsize=12, fontfamily='sans')
plt.title("Top 10 Organs by Average SUV", fontsize=20, fontweight='bold', fontfamily='sans')
plt.ylabel("Avg SUV [1]", fontsize=14, fontweight='bold', fontfamily='sans')
plt.xlabel("")
plt.tight_layout()
plt.savefig(PLOTS_CONTRIBUTION / "suv_shares" / "top10_avg_suv_per_organ.png")
plt.close()

# Plot: Pie chart of average SUV share (top 10)
plt.figure(figsize=(8, 8))
plt.pie(top10_suv[AVG_SUV], labels=top10_suv["Organ Name"],  
        textprops={'fontsize': 12, 'family': 'sans', 'weight': 'bold'},
        autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Spectral", len(top10_suv)))
plt.title("Top 10 Organs in Average SUV Distribution", fontsize=20, fontweight='bold', fontfamily='sans')
plt.tight_layout()
plt.savefig(PLOTS_CONTRIBUTION / "suv_shares" / "top10_avg_suv_pie.png")
plt.close()

# Plot: Average Volume bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=top10_vol, x="Organ Name", y=AVG_VOL, palette="RdBu")
plt.xticks(rotation=45, ha='right', fontsize=12, fontfamily='sans')
plt.title("Top 10 Organs by Average Volume", fontsize=20, fontweight='bold', fontfamily='sans')
plt.ylabel("Avg Volume [m5/kg]", fontsize=14, fontweight='bold', fontfamily='sans')
plt.xlabel("")
plt.tight_layout()
plt.savefig(PLOTS_CONTRIBUTION / "volume_shares" / "top10_avg_volume_per_organ.png")
plt.close()

# Plot: Pie chart of average volume share (top 10)
plt.figure(figsize=(8, 8))
plt.pie(top10_vol[AVG_VOL], labels=top10_vol["Organ Name"],
        textprops={'fontsize': 12, 'family': 'sans', 'weight': 'bold'},
        autopct='%1.1f%%', startangle=140, colors=sns.color_palette("RdBu", len(top10_vol)))
plt.title("Average Volume Distribution among Top 10 Organs", fontsize=20, fontweight='bold', fontfamily='sans')
plt.tight_layout()
plt.savefig(PLOTS_CONTRIBUTION / "volume_shares" / "top10_avg_volume_pie.png")
plt.close()