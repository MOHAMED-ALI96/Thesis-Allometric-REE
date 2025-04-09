from pathlib import Path

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Recursively find all folders
for folder in BASE_DIR.rglob("*"):
    if folder.is_dir():
        # Check if folder is empty (or contains only .gitkeep)
        contents = list(folder.glob("*"))
        if not any(f.is_file() and f.name != ".gitkeep" for f in contents):
            # Create .gitkeep if it doesn't exist
            gitkeep = folder / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
                print(f"Added: {gitkeep.relative_to(BASE_DIR)}")

print("\n✅ All empty folders now contain .gitkeep files.")
