"""
Locating lessons, whether installed as a wheel or run from a source checkout.

Installed, the notebooks sit in ``dlfs/_data/``. In a git checkout they are
at the repo root. Both are supported so contributors do not have to reinstall
after every edit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

_HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def data_dir() -> Path:
    """Where the notebooks, solutions and site live."""
    packaged = _HERE / "_data"
    if (packaged / "curriculum.json").exists():
        return packaged

    # Source checkout: src/dlfs/curriculum.py -> repo root
    repo = _HERE.parents[1]
    if (repo / "curriculum.json").exists():
        return repo
    repo = _HERE.parents[2]
    if (repo / "curriculum.json").exists():
        return repo

    raise FileNotFoundError(
        "Could not find the lesson data. If you are running from a git "
        "checkout, run this from inside the repository; if you installed "
        "the package, try reinstalling it."
    )


@dataclass(frozen=True)
class Lesson:
    """One lesson, with everything needed to find and describe it."""

    id: int
    phase: int
    slug: str
    title: str
    goal: str
    status: str = "todo"
    completed: str | None = None

    @property
    def number(self) -> str:
        return f"{self.id:02d}"

    @property
    def stem(self) -> str:
        return f"{self.number}-{self.slug}"

    @property
    def available(self) -> bool:
        """Has this lesson actually been written yet?"""
        return notebook_path(self.id) is not None

    @property
    def phase_name(self) -> str:
        return phases().get(self.phase, f"Phase {self.phase}")

    def __str__(self) -> str:
        mark = {"done": "[x]", "in-progress": "[~]", "todo": "[ ]"}[self.status]
        return f"{mark} {self.number}  {self.title}"


@lru_cache(maxsize=1)
def _raw() -> dict:
    return json.loads((data_dir() / "curriculum.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _progress() -> dict:
    path = data_dir() / "progress.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("lessons", {})


@lru_cache(maxsize=1)
def phases() -> dict[int, str]:
    return {p["id"]: p["name"] for p in _raw()["phases"]}


@lru_cache(maxsize=1)
def lessons() -> tuple[Lesson, ...]:
    """Every lesson in the curriculum, in order."""
    prog = _progress()
    out = []
    for item in _raw()["lessons"]:
        rec = prog.get(str(item["id"]), {})
        out.append(
            Lesson(
                id=item["id"],
                phase=item["phase"],
                slug=item["slug"],
                title=item["title"],
                goal=item["goal"],
                status=rec.get("status", "todo"),
                completed=rec.get("completed"),
            )
        )
    return tuple(out)


def lesson(which: int | str) -> Lesson:
    """Look a lesson up by number (``7``) or by name (``loss-surfaces``).

    Names resolve from most specific to least, so a partial name that is the
    obvious prefix of one lesson wins outright rather than being rejected as
    ambiguous:

    >>> lesson(11).slug
    'backpropagation'
    >>> lesson("backprop").id          # prefix of 'backpropagation'
    11
    >>> lesson("loss-surfaces").id     # exact slug
    7
    """
    try:
        n = int(which)
    except (TypeError, ValueError):
        pass
    else:
        for l in lessons():
            if l.id == n:
                return l
        raise KeyError(f"No lesson {n}. Valid ids are 1..{len(lessons())}.")

    needle = str(which).strip().lower()
    if not needle:
        raise KeyError("No lesson name given")

    all_lessons = lessons()
    for rule in (
        lambda l: l.slug == needle,                 # exact slug
        lambda l: l.title.lower() == needle,        # exact title
        lambda l: l.slug.startswith(needle),        # slug prefix
        lambda l: l.title.lower().startswith(needle),
        lambda l: needle in l.slug,                 # anywhere in the slug
        lambda l: needle in l.title.lower(),        # anywhere in the title
    ):
        hits = [l for l in all_lessons if rule(l)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            names = ", ".join(f"{l.number} {l.slug}" for l in hits[:6])
            raise KeyError(f"{which!r} is ambiguous: {names}")

    raise KeyError(f"No lesson matching {which!r}")


def _find(folder: str, lesson_id: int) -> Path | None:
    directory = data_dir() / folder
    if not directory.is_dir():
        return None
    hits = sorted(directory.glob(f"{lesson_id:02d}-*.ipynb"))
    return hits[0] if hits else None


def notebook_path(lesson_id: int) -> Path | None:
    """Path to a lesson's notebook, or None if it has not been written yet."""
    return _find("notebooks", lesson_id)


def solution_path(lesson_id: int) -> Path | None:
    """Path to a lesson's solutions notebook, or None if there isn't one."""
    return _find("solutions", lesson_id)


def available() -> Iterator[Lesson]:
    """Only the lessons that actually exist on disk."""
    return (l for l in lessons() if l.available)
