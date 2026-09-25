"""
new_lesson.py -- scaffold the next lesson in the fixed 8-section structure.

    python tools/new_lesson.py 4

Creates `lessons/04-gradient-descent-1d.py` pre-filled with the section
skeleton (Plan / Math / Numerical / Visualization / Scratch / NumPy /
TensorFlow / Exercises), pulling the title and goal from curriculum.json,
and marks the lesson in-progress on the tracker.

Refuses to overwrite an existing source file.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track import load_curriculum, find_lesson, phase_name, src_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = '''# %% [markdown]
# # Lesson {id:02d} - {title}
#
# **Phase {phase}: {phase_name}** &nbsp;|&nbsp; Deep Learning From Scratch
#
# > {goal}
#
# Runs top to bottom on Kaggle, Colab or locally. No files to download.
#
# ---

# %% [markdown]
# ## 1. Plan
#
# **What we build today**
#
# TODO
#
# **Why it exists**
#
# TODO
#
# **What breaks without it**
#
# TODO

# %% [markdown]
# ## 2. From scratch: the maths
#
# TODO -- full derivation, every partial derivative written out.

# %% [markdown]
# ## 3. Numerical: worked by hand
#
# TODO -- two or three numbers, every intermediate shown, so the code below
# has something to be checked against.

# %%
# Setup
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
plt.rcParams.update({{"figure.figsize": (9, 4), "axes.grid": True, "grid.alpha": 0.3}})

# %% [markdown]
# ## 4. Visualization

# %%
# TODO -- plot the idea

# %% [markdown]
# ## 5. Scratch code -- pure Python, no imports
#
# Must reproduce section 3 exactly.

# %%
# TODO


# %% [markdown]
# ## 6. NumPy -- vectorized

# %%
# TODO


# %% [markdown]
# ## 7. TensorFlow

# %%
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:  # not installed locally; always present on Kaggle
    HAS_TF = False
    print("TensorFlow not available -- skipping (this section runs on Kaggle).")

# %%
if HAS_TF:
    pass  # TODO

# %% [markdown]
# ## 8. Agreement check
#
# The lesson is only honest if all three implementations agree.

# %%
# TODO -- assert scratch == numpy == tensorflow to ~1e-6
print("All implementations agree.")

# %% [markdown]
# ## 9. Exercises
#
# Each one has an `assert` that tells you when you got it right.

# %%
# Exercise 1: TODO


# %% [markdown]
# ## 10. What you learned
#
# - TODO
#
# **Next:** Lesson {next_id:02d}
'''


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    cur = load_curriculum()
    lesson = find_lesson(cur, argv[0])
    dest = ROOT / src_path(lesson)

    if dest.exists():
        print(f"  {dest.relative_to(ROOT)} already exists -- not overwriting.")
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        TEMPLATE.format(
            id=lesson["id"],
            title=lesson["title"],
            phase=lesson["phase"],
            phase_name=phase_name(cur, lesson["phase"]),
            goal=lesson["goal"],
            next_id=lesson["id"] + 1,
        ),
        encoding="utf-8",
    )
    print(f"  created {dest.relative_to(ROOT)}")
    subprocess.run([sys.executable, str(ROOT / "tools" / "track.py"), "start", str(lesson["id"])])
    print(f"\n  Fill it in, then:  python tools/build_nb.py {lesson['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
