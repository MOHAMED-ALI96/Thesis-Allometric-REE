from pathlib import Path

# Go up to project root
BASE_DIR = Path(__file__).resolve().parents[2]

# Define folder structure
folders = [
    "scripts/utils",
    "scripts/data_description",
    "scripts/organ_contribution",
    "scripts/modeling",
    "scripts/significance_tests",
    "scripts/cross_validation",
    "scripts/mixed_effects",
    "scripts/age_analysis"
]

# Create all folders
for f in folders:
    path = BASE_DIR / f
    path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created: {path.relative_to(BASE_DIR)}")

print("\n✅ Script folders created. Ready for you to organize your code!")
