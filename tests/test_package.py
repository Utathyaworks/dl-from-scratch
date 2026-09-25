"""
Tests for the installable package.

These guard the things a user would hit first: does the data ship, does the
curriculum load, do the lookups behave, and does the gradient checker
actually catch a wrong gradient. Run with::

    pytest -q
"""
import doctest
import json

import pytest

import dlfs
from dlfs import curriculum, gradcheck
from dlfs.cli import build_parser


# --------------------------------------------------------------- packaging
def test_version_is_sane():
    assert dlfs.__version__.count(".") == 2
    assert all(part.isdigit() for part in dlfs.__version__.split("."))


def test_data_ships_and_is_findable():
    root = dlfs.data_dir()
    assert (root / "curriculum.json").is_file()
    assert (root / "notebooks").is_dir()


def test_curriculum_is_complete():
    lessons = dlfs.lessons()
    assert len(lessons) == 45
    assert [l.id for l in lessons] == list(range(1, 46))
    assert len({l.slug for l in lessons}) == 45, "slugs must be unique"
    assert {l.phase for l in lessons} == set(range(7))
    for l in lessons:
        assert l.title and l.goal
        assert l.status in {"todo", "in-progress", "done"}


def test_every_written_notebook_has_a_curriculum_entry():
    """A notebook on disk with no matching lesson id would be invisible."""
    ids = {l.id for l in dlfs.lessons()}
    for path in (dlfs.data_dir() / "notebooks").glob("*.ipynb"):
        assert int(path.name[:2]) in ids, f"{path.name} has no curriculum entry"


def test_done_lessons_are_actually_present():
    for l in dlfs.lessons():
        if l.status == "done":
            assert l.available, f"lesson {l.number} is marked done but has no notebook"
            assert dlfs.notebook_path(l.id).is_file()


def test_notebooks_are_valid_json_with_cells():
    for path in (dlfs.data_dir() / "notebooks").glob("*.ipynb"):
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb["nbformat"] >= 4
        assert nb["cells"], f"{path.name} has no cells"


def test_notebooks_do_not_import_dlfs():
    """Lessons must stay self-contained so they run unchanged on Kaggle."""
    for path in (dlfs.data_dir() / "notebooks").glob("*.ipynb"):
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            assert "import dlfs" not in src, f"{path.name} imports dlfs"
            assert "from dlfs" not in src, f"{path.name} imports from dlfs"


# ----------------------------------------------------------------- lookup
@pytest.mark.parametrize("key, expected", [
    (1, 1), (7, 7), ("7", 7),
    ("loss-surfaces", 7),
    ("backprop", 11),            # prefix wins over three substring matches
    ("backpropagation", 11),
    ("bptt", 31),
    ("vae", 43),
    ("Mini-GPT from Scratch", 40),
])
def test_lesson_lookup(key, expected):
    assert curriculum.lesson(key).id == expected


@pytest.mark.parametrize("key", [0, 46, 99, "zzzz", ""])
def test_lesson_lookup_rejects_nonsense(key):
    with pytest.raises(KeyError):
        curriculum.lesson(key)


def test_ambiguous_lookup_names_the_candidates():
    with pytest.raises(KeyError) as err:
        curriculum.lesson("attention")
    assert "ambiguous" in str(err.value)


# -------------------------------------------------------------- gradcheck
def _mse_and_grad():
    X, Y = [1.0, 2.0, 3.0], [2.0, 3.0, 5.0]

    def loss(p):
        return sum((p[0] * x + p[1] - y) ** 2 for x, y in zip(X, Y)) / 3

    def grad(p):
        e = [p[0] * x + p[1] - y for x, y in zip(X, Y)]
        return [2 / 3 * sum(ei * xi for ei, xi in zip(e, X)), 2 / 3 * sum(e)]

    return loss, grad


def test_gradcheck_accepts_a_correct_gradient():
    loss, grad = _mse_and_grad()
    for point in ([0.0, 0.0], [-2.0, 4.0], [1.5, 1 / 3]):
        chk = gradcheck.grad_check(grad(point), gradcheck.numeric_gradient(loss, point))
        assert chk.passed, f"correct gradient rejected at {point}: {chk}"


def test_gradcheck_catches_a_subtly_wrong_gradient():
    loss, grad = _mse_and_grad()
    wrong = [g * 0.999 for g in grad([0.0, 0.0])]       # 0.1% off
    chk = gradcheck.grad_check(wrong, gradcheck.numeric_gradient(loss, [0.0, 0.0]))
    assert not chk.passed


def test_gradcheck_survives_a_genuinely_zero_gradient():
    """Both gradients zero must not be reported as a 1.0 relative error."""
    chk = gradcheck.grad_check([0.0, 0.0], [1e-14, -2e-15])
    assert chk.passed and chk.relative > 0.5   # ratio is meaningless, abs saves it


def test_gradcheck_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        gradcheck.grad_check([1.0, 2.0], [1.0])


def test_numeric_hessian_matches_lesson_07():
    """The Lesson 07 hand calculation: H = [[28/3, 4], [4, 2]]."""
    loss, _ = _mse_and_grad()
    H = gradcheck.numeric_hessian(loss, [0.0, 0.0])
    assert abs(H[0][0] - 28 / 3) < 1e-4
    assert abs(H[0][1] - 4.0) < 1e-4
    assert abs(H[1][0] - 4.0) < 1e-4
    assert abs(H[1][1] - 2.0) < 1e-4


# ---------------------------------------------------------------- the CLI
@pytest.mark.parametrize("argv", [
    ["list"], ["list", "--phase", "1"], ["list", "--available"],
    ["list", "--status", "done"], ["list", "--search", "gradient"],
    ["info", "7"], ["where"],
])
def test_cli_commands_succeed(argv, capsys):
    from dlfs.cli import main
    assert main(argv) == 0
    assert capsys.readouterr().out.strip()


def test_cli_reports_missing_lesson_without_crashing(capsys):
    from dlfs.cli import main
    assert main(["info", "999"]) == 1
    assert "No lesson" in capsys.readouterr().out


def test_parser_exposes_every_command():
    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    names = set(actions[0].choices) if actions else set()
    assert {"list", "info", "open", "site", "verify", "where"} <= names


# --------------------------------------------------------------- doctests
@pytest.mark.parametrize("module", [dlfs, gradcheck, curriculum])
def test_doctests(module):
    result = doctest.testmod(module, verbose=False)
    assert result.failed == 0, f"{module.__name__}: {result.failed} doctest failures"
