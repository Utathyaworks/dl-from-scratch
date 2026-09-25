"""
build_nb.py -- compile lesson sources into runnable notebooks.

Lessons are AUTHORED as percent-format Python in `lessons/NN-slug.py`:

    # %% [markdown]
    # # A heading
    # Some prose with $x^2$ maths.

    # %%
    print("a code cell")

and COMPILED into `notebooks/NN-slug.ipynb`, which is what you open on
Kaggle. Authoring in .py keeps git diffs readable -- a notebook diff is
unreadable JSON noise, and this repo gets pushed every day.

    python tools/build_nb.py          # build every lesson
    python tools/build_nb.py 4        # build just lesson 4

The .ipynb files are committed too, because Kaggle and GitHub's preview
need them. The .py file is the source of truth: never hand-edit the .ipynb.
"""
import json
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "lessons"
NOTEBOOKS = ROOT / "notebooks"
SOLUTIONS = ROOT / "solutions"

CELL_RE = re.compile(r"^# %%(.*)$")


def parse_source(text):
    """Split percent-format source into (kind, source) cell tuples."""
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        src = "\n".join(buf).strip("\n")
        if kind == "markdown":
            # strip the leading "# " comment marker off every line
            src = "\n".join(l[2:] if l.startswith("# ") else (l[1:] if l == "#" else l)
                            for l in src.split("\n"))
        if src.strip():
            cells.append((kind, src))

    for line in text.split("\n"):
        m = CELL_RE.match(line)
        if m:
            flush()
            marker = m.group(1)
            kind = ("markdown" if "[markdown]" in marker
                    else "exercise" if "[exercise]" in marker
                    else "code")
            buf = []
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


def build(src_path):
    cells = parse_source(src_path.read_text(encoding="utf-8"))
    if not cells:
        raise SystemExit(f"{src_path.name}: no cells found (needs '# %%' markers)")

    nb = nbformat.v4.new_notebook()
    nb.cells = []
    for kind, src in cells:
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(src))
            continue
        cell = nbformat.v4.new_code_cell(src)
        if kind == "exercise":
            # Exercises ship unanswered, so their asserts fail on purpose.
            # "raises-exception" lets the headless verifier run past them while
            # every other assert in the lesson stays a hard gate.
            cell.metadata["tags"] = ["exercise", "raises-exception"]
        nb.cells.append(cell)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }

    # `NN-slug.py` -> notebooks/ ; `NN-slug-solutions.py` -> solutions/
    out_dir = SOLUTIONS if src_path.stem.endswith("-solutions") else NOTEBOOKS
    out = out_dir / (src_path.stem + ".ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(out))
    n = {"markdown": 0, "code": 0, "exercise": 0}
    for k, _ in cells:
        n[k] += 1
    print(f"  built {out.relative_to(ROOT)}  "
          f"({n['markdown']} md + {n['code']} code + {n['exercise']} exercise cells)")
    return out


def sources(which=None):
    all_src = sorted(LESSONS.glob("*.py"))
    if which is None:
        return all_src
    prefix = f"{int(which):02d}-"
    hits = [p for p in all_src if p.name.startswith(prefix)]
    if not hits:
        raise SystemExit(f"No lesson source matching {prefix}*.py in lessons/")
    return hits


def main(argv):
    targets = sources(argv[0] if argv else None)
    if not targets:
        print("  no lesson sources yet -- run tools/new_lesson.py <id> first")
        return 0
    for src in targets:
        build(src)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
