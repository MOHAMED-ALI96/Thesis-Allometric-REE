from pathlib import Path

# Set project root
BASE_DIR = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {"venv", ".venv", "__pycache__", ".git", ".idea", ".mypy_cache"}

# Traverse all folders
for folder in BASE_DIR.rglob("*"):
    if folder.is_dir():
        # Skip excluded folders
        if any(part in EXCLUDE_DIRS for part in folder.parts):
            continue

        # Check if the folder has no files (or only other folders)
        files = list(folder.iterdir())
        if not any(f.is_file() for f in files):
            gitkeep = folder / ".gitkeep"
            gitkeep.touch(exist_ok=True)
            print(f"➕ Added .gitkeep to: {folder.relative_to(BASE_DIR)}")

print("\n✅ All empty (non-excluded) folders now have .gitkeep files.")
