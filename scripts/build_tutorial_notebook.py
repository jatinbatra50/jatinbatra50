"""Turn the concise Markdown lesson into a self-contained Jupyter notebook."""

import hashlib
import json
from pathlib import Path
import re


def build():
    root = Path(__file__).resolve().parents[1]
    source = (root / "TUTORIAL.md").read_text(encoding="utf-8")
    # The adjacent code cells render these figures inside Jupyter.
    source = re.sub(r"!\[[^\]]*\]\(results/demo/[^)]+\)\n?", "", source)
    parts = re.split(r"^```python\n(.*?)^```[ \t]*$", source, flags=re.M | re.S)
    cells = []
    for index, part in enumerate(parts):
        content = part.strip() + "\n"
        if not content.strip():
            continue
        kind = "code" if index % 2 else "markdown"
        cell = {
            "cell_type": kind,
            "id": hashlib.sha256(f"{index}:{content}".encode()).hexdigest()[:12],
            "metadata": {},
            "source": content.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = root / "Provable_UQ_Tutorial.ipynb"
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Built {destination.name}: {len(cells)} cells")


if __name__ == "__main__":
    build()
