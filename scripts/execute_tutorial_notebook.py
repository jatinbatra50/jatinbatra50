"""Execute this project's pure-Python tutorial cells and retain notebook outputs.

This lightweight runner does not require a Jupyter kernel. It deliberately
executes in an empty temporary directory, proving that the tutorial cells do
not depend on the repository's working directory or local modules. It supports
ordinary Python cells only; use Jupyter for notebooks containing magics.
"""

import argparse
import base64
import contextlib
import io
import json
import os
import platform
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


def execute(path):
    path = Path(path).resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    count = 0
    original_show = plt.show
    original_cwd = Path.cwd()
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="uq-notebook-") as directory:
            os.chdir(directory)
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                count += 1
                outputs = []
                stream = io.StringIO()

                def show(*args, **kwargs):
                    for number in plt.get_fignums():
                        figure = plt.figure(number)
                        buffer = io.BytesIO()
                        figure.savefig(buffer, format="png", bbox_inches="tight", dpi=130)
                        outputs.append({
                            "output_type": "display_data",
                            "data": {
                                "image/png": base64.b64encode(buffer.getvalue()).decode(),
                                "text/plain": ["<matplotlib figure>"],
                            },
                            "metadata": {},
                        })
                        plt.close(figure)

                plt.show = show
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    exec(compile("".join(cell["source"]), f"{path.name}:cell{index}", "exec"),
                         namespace)
                printed = stream.getvalue()
                if printed:
                    outputs.insert(0, {"output_type": "stream", "name": "stdout",
                                       "text": printed.splitlines(keepends=True)})
                cell["execution_count"] = count
                cell["outputs"] = outputs
                print(f"Cell {index}: completed", flush=True)
                if printed:
                    print(printed, end="", flush=True)
    finally:
        plt.show = original_show
        plt.close("all")
        os.chdir(original_cwd)
    # Save only after every cell succeeds, retaining the old file on failure.
    notebook.setdefault("metadata", {}).setdefault("language_info", {})["version"] = platform.python_version()
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"Executed {count} code cells in {time.perf_counter()-started:.2f}s: {path}")
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", default="Provable_UQ_Tutorial.ipynb")
    args = parser.parse_args()
    execute(args.notebook)
