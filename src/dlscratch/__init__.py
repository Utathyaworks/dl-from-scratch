"""
dl-from-scratch -- learn deep learning by deriving it.

45 notebooks that derive the maths, work a tiny example by hand, visualize
it, then implement it three times (pure Python, NumPy, TensorFlow) and
assert that all three agree with the hand calculation.

Quick start::

    dlscratch list              # the curriculum and what is finished
    dlscratch open 7            # copy lesson 7 here and launch Jupyter
    dlscratch site              # serve the interactive site locally

The notebooks are deliberately **self-contained**: they import nothing from
this package, so the same file runs unchanged on Kaggle, on Colab, and on
your machine. This package exists to find, copy, serve and verify them --
plus a few utilities worth reusing outside a notebook:

    >>> from dlscratch import grad_check, numeric_gradient
    >>> f = lambda p: (p[0] - 3) ** 2
    >>> analytic = lambda p: [2 * (p[0] - 3)]
    >>> grad_check(analytic([0.0]), numeric_gradient(f, [0.0])).passed
    True
"""

__version__ = "0.4.0"

from dlscratch.curriculum import (
    Lesson,
    lessons,
    lesson,
    phases,
    notebook_path,
    solution_path,
    data_dir,
)
from dlscratch.gradcheck import GradCheck, grad_check, numeric_gradient, numeric_hessian

__all__ = [
    "__version__",
    "Lesson",
    "lessons",
    "lesson",
    "phases",
    "notebook_path",
    "solution_path",
    "data_dir",
    "GradCheck",
    "grad_check",
    "numeric_gradient",
    "numeric_hessian",
]
