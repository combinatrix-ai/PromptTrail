import glob

from tqdm import tqdm

# load all files in examples and tests with its name in text


def load_all_important_files():
    text = ""
    for file in tqdm(list(glob.glob("examples/**/*.py", recursive=True))):
        text += f"Example filename: {file}\n"
        text += f"```python\n{open(file, 'r').read()}\n```\n"

    for file in tqdm(list(glob.glob("tests/**/*.py", recursive=True))):
        text += f"Test filename: {file}\n"
        text += f"```python\n{open(file, 'r').read()}\n```\n"

    # add README.md content
    text += f"```README\n{open('README.md', 'r').read()}\n```\n"

    # add docs *.md content
    for file in tqdm(list(glob.glob("docs/*.md", recursive=False))):
        text += f"Docs filename: {file}\n"
        text += f"```markdown\n{open(file, 'r').read()}\n```\n"

    return text


if __name__ == "__main__":
    print(load_all_important_files())
