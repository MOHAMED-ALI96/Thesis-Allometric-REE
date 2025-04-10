from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# List of folders to create
folders = [
    BASE_DIR / "scripts",
    BASE_DIR / "notebooks",
    BASE_DIR / "bash",
    BASE_DIR / "data" / "public",
    BASE_DIR / "data" / "private",
    BASE_DIR / "plots",
    BASE_DIR / "results",
    BASE_DIR / "thesis"
]

# Create folders if they don't exist
for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Created: {folder.relative_to(BASE_DIR)}")

print("\n✅ Folder structure created successfully.")
