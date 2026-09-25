"""
Gradient checking -- verifying hand-derived calculus against the definition.

This is the single most useful debugging tool in the course, so it lives here
as well as being written out longhand inside each lesson. Lesson 06 derives
why the test needs *two* criteria rather than one:

- **Relative** error catches a wrong derivation at any scale, since a gradient
  of 1e6 and one of 1e-6 cannot share an absolute tolerance.
- An **absolute** escape hatch is still needed, because at a converged
  parameter both gradients are genuinely zero and a pure ratio then divides
  floating-point noise by floating-point noise, reporting ~1.0 and condemning
  correct code.

A check passes when *either* criterion is satisfied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

Vector = Sequence[float]


@dataclass(frozen=True)
class GradCheck:
    """The outcome of one gradient check."""

    relative: float
    absolute: float
    passed: bool
    rel_tol: float
    abs_tol: float

    @property
    def verdict(self) -> str:
        if not self.passed:
            return "FAIL"
        if self.relative < 1e-7:
            return "excellent"
        if self.relative < 1e-5:
            return "fine"
        return "pass (small gradient)"

    def __bool__(self) -> bool:
        return self.passed

    def __str__(self) -> str:
        return (f"rel {self.relative:.2e}  abs {self.absolute:.2e}  "
                f"-> {self.verdict}")


def numeric_gradient(f: Callable[[Vector], float], point: Vector,
                     eps: float = 1e-6) -> list[float]:
    """Central-difference gradient of ``f`` at ``point``.

    Central differences have error O(eps^2) against O(eps) for a one-sided
    step, which is worth the second function evaluation per coordinate.
    """
    point = list(point)
    out = []
    for i in range(len(point)):
        hi, lo = list(point), list(point)
        hi[i] += eps
        lo[i] -= eps
        out.append((f(hi) - f(lo)) / (2 * eps))
    return out


def numeric_hessian(f: Callable[[Vector], float], point: Vector,
                    eps: float = 1e-4) -> list[list[float]]:
    """Central-difference Hessian. ``eps`` is larger than for a gradient
    because second differences divide by eps**2 and lose precision faster."""
    point = list(point)
    n = len(point)
    base = f(point)
    H = [[0.0] * n for _ in range(n)]

    def shifted(**deltas) -> float:
        p = list(point)
        for idx, d in deltas.items():
            p[int(idx)] += d
        return f(p)

    for i in range(n):
        H[i][i] = (shifted(**{str(i): eps}) - 2 * base
                   + shifted(**{str(i): -eps})) / eps ** 2
        for j in range(i + 1, n):
            pp = list(point); pp[i] += eps; pp[j] += eps
            pm = list(point); pm[i] += eps; pm[j] -= eps
            mp = list(point); mp[i] -= eps; mp[j] += eps
            mm = list(point); mm[i] -= eps; mm[j] -= eps
            H[i][j] = H[j][i] = (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * eps ** 2)
    return H


def grad_check(analytic: Vector, numeric: Vector,
               rel_tol: float = 1e-5, abs_tol: float = 1e-8) -> GradCheck:
    """Compare a hand-derived gradient against a numerical one.

    >>> grad_check([2.0, -3.0], [2.0000001, -3.0000002]).passed
    True
    >>> grad_check([2.0], [2.2]).passed          # 10% wrong
    False
    >>> grad_check([0.0], [1e-12]).passed        # both zero: not a failure
    True
    """
    analytic, numeric = list(analytic), list(numeric)
    if len(analytic) != len(numeric):
        raise ValueError(
            f"length mismatch: {len(analytic)} analytic vs {len(numeric)} numeric"
        )
    if not analytic:
        raise ValueError("nothing to check -- both gradients are empty")

    rel = max(
        abs(a - n) / max(1e-300, abs(a) + abs(n))
        for a, n in zip(analytic, numeric)
    )
    dif = max(abs(a - n) for a, n in zip(analytic, numeric))
    return GradCheck(
        relative=rel,
        absolute=dif,
        passed=(rel < rel_tol or dif < abs_tol),
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def check_against_numeric(f: Callable[[Vector], float],
                          analytic_grad: Callable[[Vector], Vector],
                          point: Vector, **kwargs) -> GradCheck:
    """Convenience wrapper: differentiate ``f`` numerically at ``point`` and
    compare it with ``analytic_grad``.

    >>> f = lambda p: p[0] ** 2 + 3 * p[1]
    >>> g = lambda p: [2 * p[0], 3.0]
    >>> bool(check_against_numeric(f, g, [1.5, -2.0]))
    True
    """
    return grad_check(analytic_grad(point), numeric_gradient(f, point), **kwargs)
