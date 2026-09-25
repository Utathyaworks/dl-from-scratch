"""
The ``dlfs`` command line.

    dlfs list                 the curriculum and what is finished
    dlfs info 7               one lesson in detail
    dlfs open 7               copy lesson 7 here and launch Jupyter
    dlfs site                 serve the interactive site locally
    dlfs verify 7             run a lesson headless and report failures
    dlfs where                where the installed notebooks live
"""
from __future__ import annotations

import argparse
import http.server
import os
import shutil
import socketserver
import subprocess
import sys
import webbrowser
from pathlib import Path

from dlfs import __version__
from dlfs.curriculum import (
    data_dir,
    lesson,
    lessons,
    notebook_path,
    phases,
    solution_path,
)

MARK = {"done": "[x]", "in-progress": "[~]", "todo": "[ ]"}


def _c(text: str, code: str) -> str:
    """Colour, unless the output is piped or NO_COLOR is set."""
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


# ------------------------------------------------------------------ list
def cmd_list(args) -> int:
    items = list(lessons())
    if args.phase is not None:
        items = [l for l in items if l.phase == args.phase]
    if args.status:
        items = [l for l in items if l.status == args.status]
    if args.available:
        items = [l for l in items if l.available]
    if args.search:
        needle = args.search.lower()
        items = [l for l in items
                 if needle in l.title.lower() or needle in l.goal.lower()
                 or needle in l.slug]

    if not items:
        print("  Nothing matches that filter.")
        return 0

    current = None
    for l in items:
        if l.phase != current:
            current = l.phase
            print(f"\n  {_c(f'Phase {current} - {l.phase_name}', '1;36')}")
            print("  " + "-" * 64)
        tag = _c(MARK[l.status], "32" if l.status == "done" else
                 "33" if l.status == "in-progress" else "90")
        runnable = _c(" *", "32") if l.available else ""
        print(f"  {tag} {l.number}  {l.title:<46}{runnable}")
        if args.verbose:
            print(f"         {_c(l.goal, '90')}")

    total = len(lessons())
    done = sum(1 for l in lessons() if l.status == "done")
    ready = sum(1 for l in lessons() if l.available)
    print(f"\n  {done}/{total} written  ({done * 100 // total}%)   "
          f"{_c('*', '32')} = {ready} runnable now")
    print(f"  {_c('dlfs open <number>', '1')} to start one\n")
    return 0


# ------------------------------------------------------------------ info
def cmd_info(args) -> int:
    l = lesson(args.lesson)
    nb, sol = notebook_path(l.id), solution_path(l.id)
    print(f"\n  {_c(f'Lesson {l.number} - {l.title}', '1;36')}")
    print(f"  Phase {l.phase}: {l.phase_name}")
    print(f"\n  {l.goal}\n")
    print(f"  status     {l.status}")
    if l.completed:
        print(f"  completed  {l.completed}")
    print(f"  notebook   {nb if nb else _c('not written yet', '90')}")
    print(f"  solutions  {sol if sol else _c('not written yet', '90')}")
    if nb:
        print(f"\n  {_c(f'dlfs open {l.id}', '1')} to work through it")
    print()
    return 0


# ------------------------------------------------------------------ open
def cmd_open(args) -> int:
    l = lesson(args.lesson)
    nb = notebook_path(l.id)
    if nb is None:
        print(f"  Lesson {l.number} ({l.title}) has not been written yet.")
        print("  Run 'dlfs list --available' to see what is ready.")
        return 1

    dest_dir = Path(args.dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / nb.name

    if dest.exists() and not args.force:
        print(f"  {dest} already exists -- keeping your copy.")
        print("  Use --force to overwrite it with a fresh one.")
    else:
        shutil.copy2(nb, dest)
        print(f"  copied  {dest}")

    if args.solutions:
        sol = solution_path(l.id)
        if sol:
            sol_dest = dest_dir / sol.name
            if not sol_dest.exists() or args.force:
                shutil.copy2(sol, sol_dest)
                print(f"  copied  {sol_dest}")
        else:
            print("  (no solutions notebook for this lesson)")

    if args.no_launch:
        print(f"\n  Open it with:  jupyter lab {dest}")
        return 0

    for cmd in (["jupyter", "lab"], ["jupyter", "notebook"]):
        if shutil.which(cmd[0]) is None:
            continue
        print(f"\n  launching {' '.join(cmd)} ...")
        try:
            subprocess.run([*cmd, str(dest)], check=False)
            return 0
        except OSError:
            break

    print("\n  Jupyter is not installed. Either:")
    print("    pip install 'dl-from-scratch[notebooks]'")
    print(f"  or open {dest} in VS Code / your editor of choice.")
    return 0


# ------------------------------------------------------------------ site
def cmd_site(args) -> int:
    root = data_dir() / "docs"
    if not root.is_dir():
        print("  The site is not bundled with this install.")
        print("  Browse it online: https://utathyaworks.github.io/dl-from-scratch")
        return 1

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a):        # keep the terminal quiet
            pass

    class Server(socketserver.TCPServer):
        allow_reuse_address = True

    port = args.port
    for attempt in range(20):
        try:
            httpd = Server(("", port), Handler)
            break
        except OSError:
            port += 1
    else:
        print(f"  Could not find a free port near {args.port}.")
        return 1

    url = f"http://localhost:{port}"
    print(f"\n  serving {root}")
    print(f"  {_c(url, '1;36')}")
    print(f"  {_c('Ctrl-C to stop', '90')}\n")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped\n")
    finally:
        httpd.server_close()
    return 0


# ------------------------------------------------------------------ verify
def cmd_verify(args) -> int:
    try:
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError
    except ImportError:
        print("  Verification needs the dev extras:")
        print("    pip install 'dl-from-scratch[dev]'")
        return 1

    if args.lesson:
        target = notebook_path(lesson(args.lesson).id)
        if target is None:
            print(f"  Lesson {args.lesson} has not been written yet.")
            return 1
        targets = [target]
    else:
        targets = sorted((data_dir() / "notebooks").glob("*.ipynb"))

    if not targets:
        print("  No notebooks found.")
        return 1

    failures = 0
    for path in targets:
        nb = nbformat.read(str(path), as_version=4)
        for cell in nb.cells:
            tags = cell.get("metadata", {}).get("tags", [])
            if "exercise" in tags and "raises-exception" not in tags:
                cell.metadata["tags"] = tags + ["raises-exception"]
        client = NotebookClient(nb, timeout=900, kernel_name="python3",
                                allow_errors=False,
                                resources={"metadata": {"path": str(path.parent)}})
        try:
            client.execute()
            print(f"  {_c('PASS', '32')}  {path.name}")
        except CellExecutionError as err:
            failures += 1
            print(f"  {_c('FAIL', '31')}  {path.name}")
            for line in str(err).splitlines()[-12:]:
                print(f"        {line}")

    print(f"\n  {len(targets) - failures}/{len(targets)} passed")
    return 1 if failures else 0


# ------------------------------------------------------------------ where
def cmd_where(args) -> int:
    root = data_dir()
    print(f"\n  data root  {root}")
    for name in ("notebooks", "solutions", "docs"):
        d = root / name
        n = len(list(d.glob('*.ipynb'))) if name != "docs" else None
        extra = f"  ({n} notebooks)" if n is not None else ""
        print(f"  {name:<10} {d if d.exists() else _c('missing', '90')}{extra}")
    print()
    return 0


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dlfs",
        description="Learn deep learning from first principles -- 45 notebooks "
                    "that derive the maths, work the numbers by hand, then "
                    "implement it in pure Python, NumPy and TensorFlow.",
        epilog="Start with:  dlfs list     then:  dlfs open 1",
    )
    p.add_argument("-V", "--version", action="version",
                   version=f"dl-from-scratch {__version__}")
    sub = p.add_subparsers(dest="command")

    pl = sub.add_parser("list", help="show the curriculum")
    pl.add_argument("-p", "--phase", type=int, help="only this phase (0-6)")
    pl.add_argument("-s", "--status", choices=["todo", "in-progress", "done"])
    pl.add_argument("-a", "--available", action="store_true",
                    help="only lessons that are written and runnable")
    pl.add_argument("--search", metavar="TEXT", help="filter by title or goal")
    pl.add_argument("-v", "--verbose", action="store_true", help="show each goal")
    pl.set_defaults(func=cmd_list)

    pi = sub.add_parser("info", help="details of one lesson")
    pi.add_argument("lesson", help="number or slug, e.g. 7 or loss-surfaces")
    pi.set_defaults(func=cmd_info)

    po = sub.add_parser("open", help="copy a lesson here and launch Jupyter")
    po.add_argument("lesson", help="number or slug")
    po.add_argument("-d", "--dir", default="dlfs-lessons",
                    help="where to copy it (default: ./dlfs-lessons)")
    po.add_argument("--solutions", action="store_true", help="copy the solutions too")
    po.add_argument("--no-launch", action="store_true", help="copy only")
    po.add_argument("-f", "--force", action="store_true", help="overwrite an existing copy")
    po.set_defaults(func=cmd_open)

    ps = sub.add_parser("site", help="serve the interactive site locally")
    ps.add_argument("-p", "--port", type=int, default=8000)
    ps.add_argument("--no-browser", action="store_true")
    ps.set_defaults(func=cmd_site)

    pv = sub.add_parser("verify", help="execute notebooks and report failures")
    pv.add_argument("lesson", nargs="?", help="number or slug; omit for all")
    pv.set_defaults(func=cmd_verify)

    pw = sub.add_parser("where", help="show where the notebooks are installed")
    pw.set_defaults(func=cmd_where)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyError as err:
        print(f"  {err.args[0] if err.args else err}")
        return 1
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
