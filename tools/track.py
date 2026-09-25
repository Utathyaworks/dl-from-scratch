"""
track.py -- the project todo list.

Lessons can be done in ANY order. This keeps the record straight:
who is todo / in-progress / done, when, and in what order you actually did them.

    python tools/track.py status              # the todo board
    python tools/track.py next                # suggest what to do next
    python tools/track.py start 4             # mark lesson 4 in-progress
    python tools/track.py done 4              # mark lesson 4 done (records the date)
    python tools/track.py todo 4              # send it back to todo
    python tools/track.py note 4 "revisit the lr sweep"
    python tools/track.py show 4              # everything about one lesson
    python tools/track.py sync                # regenerate PROGRESS.md + site data

Every command re-writes PROGRESS.md and docs/data/progress.json, so the
repo and the website always agree with reality.
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRICULUM = ROOT / "curriculum.json"
STATE = ROOT / "progress.json"
PROGRESS_MD = ROOT / "PROGRESS.md"
SITE_DATA = ROOT / "docs" / "data" / "progress.json"

STATUSES = ("todo", "in-progress", "done")
MARK = {"todo": "[ ]", "in-progress": "[~]", "done": "[x]"}


# ---------------------------------------------------------------- data access
def load_curriculum():
    return json.loads(CURRICULUM.read_text(encoding="utf-8"))


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"lessons": {}, "completed_order": []}


def save_state(state):
    STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def entry(state, lid):
    """Get (creating if needed) the state record for one lesson id."""
    return state["lessons"].setdefault(
        str(lid), {"status": "todo", "started": None, "completed": None, "notes": []}
    )


def find_lesson(cur, lid):
    for lesson in cur["lessons"]:
        if lesson["id"] == int(lid):
            return lesson
    raise SystemExit(f"No lesson {lid}. Valid ids are 1..{len(cur['lessons'])}.")


def phase_name(cur, pid):
    for p in cur["phases"]:
        if p["id"] == pid:
            return p["name"]
    return f"Phase {pid}"


def nb_path(lesson):
    return f"notebooks/{lesson['id']:02d}-{lesson['slug']}.ipynb"


def src_path(lesson):
    return f"lessons/{lesson['id']:02d}-{lesson['slug']}.py"


# ---------------------------------------------------------------- commands
def cmd_status(cur, state, args):
    counts = {s: 0 for s in STATUSES}
    for lesson in cur["lessons"]:
        counts[entry(state, lesson["id"])["status"]] += 1
    total = len(cur["lessons"])

    only = args[0] if args else None
    current_phase = None
    for lesson in cur["lessons"]:
        rec = entry(state, lesson["id"])
        if only and rec["status"] != only:
            continue
        if lesson["phase"] != current_phase:
            current_phase = lesson["phase"]
            print(f"\n  Phase {current_phase} - {phase_name(cur, current_phase)}")
            print("  " + "-" * 62)
        when = rec["completed"] or rec["started"] or ""
        flag = " *" if rec["notes"] else ""
        print(f"  {MARK[rec['status']]} {lesson['id']:02d}  {lesson['title'][:44]:<44} {when}{flag}")

    pct = counts["done"] * 100 // total
    bar = "#" * (pct // 4) + "." * (25 - pct // 4)
    print(f"\n  [{bar}] {counts['done']}/{total} done  ({pct}%)")
    print(f"  in-progress: {counts['in-progress']}   todo: {counts['todo']}")
    if state["completed_order"]:
        order = " -> ".join(f"{i:02d}" for i in state["completed_order"][-10:])
        print(f"  recent order: {order}")
    print()


def cmd_next(cur, state, args):
    inprog = [l for l in cur["lessons"] if entry(state, l["id"])["status"] == "in-progress"]
    if inprog:
        print("\n  Unfinished first:")
        for l in inprog:
            print(f"    {l['id']:02d}  {l['title']}")
        print()
        return
    for lesson in cur["lessons"]:
        if entry(state, lesson["id"])["status"] == "todo":
            print(f"\n  Next in sequence: {lesson['id']:02d}  {lesson['title']}")
            print(f"  Goal: {lesson['goal']}")
            print(f"\n    python tools/new_lesson.py {lesson['id']}\n")
            return
    print("\n  All 45 lessons done.\n")


def cmd_start(cur, state, args):
    lesson = find_lesson(cur, args[0])
    rec = entry(state, lesson["id"])
    rec["status"] = "in-progress"
    rec["started"] = rec["started"] or date.today().isoformat()
    print(f"  [~] {lesson['id']:02d} {lesson['title']} -> in-progress")


def cmd_done(cur, state, args):
    lesson = find_lesson(cur, args[0])
    rec = entry(state, lesson["id"])
    rec["status"] = "done"
    rec["started"] = rec["started"] or date.today().isoformat()
    rec["completed"] = date.today().isoformat()
    if lesson["id"] in state["completed_order"]:
        state["completed_order"].remove(lesson["id"])
    state["completed_order"].append(lesson["id"])
    n = len(state["completed_order"])
    print(f"  [x] {lesson['id']:02d} {lesson['title']} -> done  ({n}/{len(cur['lessons'])})")


def cmd_todo(cur, state, args):
    lesson = find_lesson(cur, args[0])
    rec = entry(state, lesson["id"])
    rec["status"] = "todo"
    rec["completed"] = None
    if lesson["id"] in state["completed_order"]:
        state["completed_order"].remove(lesson["id"])
    print(f"  [ ] {lesson['id']:02d} {lesson['title']} -> todo")


def cmd_note(cur, state, args):
    lesson = find_lesson(cur, args[0])
    text = " ".join(args[1:])
    if not text:
        raise SystemExit("Usage: track.py note <id> \"your note\"")
    entry(state, lesson["id"])["notes"].append({"date": date.today().isoformat(), "text": text})
    print(f"  noted on {lesson['id']:02d}: {text}")


def cmd_show(cur, state, args):
    lesson = find_lesson(cur, args[0])
    rec = entry(state, lesson["id"])
    print(f"\n  Lesson {lesson['id']:02d} - {lesson['title']}")
    print(f"  Phase {lesson['phase']}: {phase_name(cur, lesson['phase'])}")
    print(f"  Goal:      {lesson['goal']}")
    print(f"  Status:    {rec['status']}")
    print(f"  Started:   {rec['started'] or '-'}")
    print(f"  Completed: {rec['completed'] or '-'}")
    print(f"  Source:    {src_path(lesson)}")
    print(f"  Notebook:  {nb_path(lesson)}")
    if rec["notes"]:
        print("  Notes:")
        for n in rec["notes"]:
            print(f"    {n['date']}  {n['text']}")
    print()


def cmd_sync(cur, state, args):
    pass  # writing happens unconditionally in main()


# ---------------------------------------------------------------- generated files
def write_progress_md(cur, state):
    done = sum(1 for l in cur["lessons"] if entry(state, l["id"])["status"] == "done")
    total = len(cur["lessons"])
    pct = done * 100 // total

    out = [
        "# Progress",
        "",
        f"**{done} / {total} lessons complete ({pct}%)**",
        "",
        "`[ ]` todo &nbsp;&nbsp; `[~]` in progress &nbsp;&nbsp; `[x]` done",
        "",
        "Lessons are done in whatever order makes sense on the day; the",
        "*Done* column records when each one actually landed.",
        "",
    ]
    for phase in cur["phases"]:
        lessons = [l for l in cur["lessons"] if l["phase"] == phase["id"]]
        if not lessons:
            continue
        pdone = sum(1 for l in lessons if entry(state, l["id"])["status"] == "done")
        out += [
            f"## Phase {phase['id']} - {phase['name']} ({pdone}/{len(lessons)})",
            "",
            "| | # | Lesson | Goal | Done |",
            "|---|---|---|---|---|",
        ]
        for l in lessons:
            rec = entry(state, l["id"])
            title = l["title"]
            if rec["status"] == "done":
                title = f"[{title}]({nb_path(l)})"
            out.append(
                f"| `{MARK[rec['status']]}` | {l['id']:02d} | {title} | {l['goal']} | {rec['completed'] or ''} |"
            )
        out.append("")

    if state["completed_order"]:
        out += ["## Order completed", "", " -> ".join(f"{i:02d}" for i in state["completed_order"]), ""]

    notes = [(l, n) for l in cur["lessons"] for n in entry(state, l["id"])["notes"]]
    if notes:
        out += ["## Notes", ""]
        for l, n in notes:
            out.append(f"- **{l['id']:02d} {l['title']}** ({n['date']}): {n['text']}")
        out.append("")

    out += ["---", "", "*Generated by `tools/track.py` - do not edit by hand.*", ""]
    PROGRESS_MD.write_text("\n".join(out), encoding="utf-8")


def write_site_data(cur, state):
    data = {"phases": cur["phases"], "github_user": cur["github_user"], "repo": cur["repo"], "lessons": []}
    for l in cur["lessons"]:
        rec = entry(state, l["id"])
        data["lessons"].append(
            {
                **{k: l[k] for k in ("id", "phase", "slug", "title", "goal")},
                "status": rec["status"],
                "completed": rec["completed"],
                "notebook": nb_path(l),
            }
        )
    SITE_DATA.parent.mkdir(parents=True, exist_ok=True)
    SITE_DATA.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- entry point
COMMANDS = {
    "status": cmd_status, "next": cmd_next, "start": cmd_start, "done": cmd_done,
    "todo": cmd_todo, "note": cmd_note, "show": cmd_show, "sync": cmd_sync,
}


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    name, args = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"Unknown command '{name}'. Try: {', '.join(COMMANDS)}")
        return 1
    if name in ("start", "done", "todo", "note", "show") and not args:
        print(f"Usage: track.py {name} <lesson id>")
        return 1

    cur, state = load_curriculum(), load_state()
    COMMANDS[name](cur, state, args)
    save_state(state)
    write_progress_md(cur, state)
    write_site_data(cur, state)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
