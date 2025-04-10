import os
import shutil

def is_effectively_empty(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return True  # truly empty or only whitespace
        if len(lines) == 1 and lines[0].startswith("#"):
            return True  # only one comment line
    except Exception as e:
        print(f"⚠️ Could not read {file_path}: {e}")
    return False

def find_empty_or_comment_only_scripts(root_dir):
    target_files = []
    for foldername, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(foldername, filename)
                if is_effectively_empty(file_path):
                    target_files.append(file_path)
    return target_files

def find_pycache_folders(root_dir):
    pycache_dirs = []
    for foldername, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == "__pycache__":
                pycache_dirs.append(os.path.join(foldername, dirname))
    return pycache_dirs

def delete_files(file_list):
    for path in file_list:
        os.remove(path)

def delete_dirs(dir_list):
    for path in dir_list:
        shutil.rmtree(path)

if __name__ == "__main__":
    search_root = os.path.abspath(os.path.dirname(__file__))

    targets = find_empty_or_comment_only_scripts(search_root)
    caches = find_pycache_folders(search_root)

    if targets:
        print("🗂️ Empty or comment-only .py files:")
        for path in targets:
            print(f" - {path}")
    else:
        print("✅ No empty or comment-only .py files found.")

    if caches:
        print("\n🧊 Found __pycache__ folders:")
        for path in caches:
            print(f" - {path}")
    else:
        print("\n✅ No __pycache__ folders found.")

    if targets or caches:
        confirm = input("\n❓ Do you want to delete these files and folders? (yes/no): ").strip().lower()
        if confirm == "yes":
            if targets:
                delete_files(targets)
            if caches:
                delete_dirs(caches)
            print("🧹 Cleanup complete.")
        else:
            print("❌ Cleanup canceled.")
