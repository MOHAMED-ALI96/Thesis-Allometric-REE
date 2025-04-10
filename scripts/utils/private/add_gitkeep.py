import os

def remove_gitkeep_files(root_dir):
    removed = []
    for foldername, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == ".gitkeep":
                file_path = os.path.join(foldername, filename)
                os.remove(file_path)
                removed.append(file_path)
    return removed

if __name__ == "__main__":
    # Set root to 'scripts/' directory
    scripts_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    removed_files = remove_gitkeep_files(scripts_root)

    if removed_files:
        print("🗑️ Removed the following .gitkeep files:")
        for path in removed_files:
            print(f" - {path}")
    else:
        print("✅ No .gitkeep files found in the 'scripts' folder.")

