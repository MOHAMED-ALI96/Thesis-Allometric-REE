from pathlib import Path

IGNORE_DIRS = {"venv", ".git", "__pycache__", ".vscode", ".ipynb_checkpoints"}

def print_tree(base_dir, prefix=""):
    base = Path(base_dir)
    entries = sorted([p for p in base.iterdir() if p.name not in IGNORE_DIRS])
    for i, path in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + path.name)
        if path.is_dir():
            new_prefix = prefix + ("    " if i == len(entries) - 1 else "│   ")
            print_tree(path, new_prefix)

# Run it from your repo root
print_tree(".")
