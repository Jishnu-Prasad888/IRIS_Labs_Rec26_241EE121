import os

EXCLUDE_DIRS = {"node_modules", "venv", "__pycache__",".git",".vscode"}

def print_tree(root, prefix=""):
    try:
        entries = sorted(os.listdir(root))
    except PermissionError:
        return

    entries = [e for e in entries if e not in EXCLUDE_DIRS]

    for i, entry in enumerate(entries):
        path = os.path.join(root, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry)

        if os.path.isdir(path):
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    print_tree(".")
