from pathlib import Path

# Set project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Scan all subfolders
for folder in BASE_DIR.rglob("*"):
    if folder.is_dir():
        files = list(folder.iterdir())
        if not any(f.is_file() for f in files):  # If no files inside
            gitkeep = folder / ".gitkeep"
            gitkeep.touch()
            print(f"Added .gitkeep to: {folder.relative_to(BASE_DIR)}")

print("\n✅ All empty folders now have .gitkeep files.")
