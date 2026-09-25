"""
verify.py -- execute notebooks headless and fail on any error.

This is the gate before a push. It runs every cell top to bottom in a fresh
kernel, exactly as Kaggle would, so a lesson can never be published broken.

    python tools/verify.py            # verify every built notebook
    python tools/verify.py 4          # verify just lesson 4

Cells tagged `exercise` are allowed to raise: exercises ship unanswered, so
their asserts fail by design. Every OTHER assert -- in particular the
scratch/NumPy/TensorFlow agreement checks -- is a hard gate.

Executed output is written to `notebooks/` so GitHub renders the plots
without anyone having to run anything.
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
SOLUTIONS = ROOT / "solutions"


def verify(path, timeout=600):
    nb = nbformat.read(str(path), as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    # Let the intentionally-unanswered exercises raise without failing the run.
    # build_nb.py normally sets this tag already; add it only if it is missing,
    # or the executed notebook gets written back with a duplicated tag and
    # drifts out of sync with its source.
    for cell in nb.cells:
        tags = cell.get("metadata", {}).get("tags", [])
        if "exercise" in tags and "raises-exception" not in tags:
            cell.metadata["tags"] = tags + ["raises-exception"]

    t0 = time.perf_counter()
    try:
        client.execute()
    except CellExecutionError as err:
        print(f"  FAIL  {path.name}")
        print("  " + "-" * 60)
        for line in str(err).splitlines()[-25:]:
            print("  " + line)
        return False

    nbformat.write(nb, str(path))  # keep the executed outputs for GitHub preview
    n_ex = sum(1 for c in nb.cells if "exercise" in c.get("metadata", {}).get("tags", []))
    print(f"  PASS  {path.name}   {time.perf_counter() - t0:.1f}s"
          f"   ({len(nb.cells)} cells, {n_ex} exercises tolerated)")
    return True


def main(argv):
    pattern = f"{int(argv[0]):02d}-*.ipynb" if argv else "*.ipynb"
    targets = sorted(NOTEBOOKS.glob(pattern)) + sorted(SOLUTIONS.glob(pattern))
    if argv and not targets:
        raise SystemExit(f"No built notebook for lesson {argv[0]} -- run build_nb.py first")

    if not targets:
        print("  nothing to verify yet")
        return 0

    print(f"Verifying {len(targets)} notebook(s)...")
    results = [verify(p) for p in targets]
    ok = sum(results)
    print(f"\n{ok}/{len(results)} passed")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
