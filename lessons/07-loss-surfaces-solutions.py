# %% [markdown]
# # Lesson 07 - Solutions
#
# **Loss Surfaces and MSE** &nbsp;|&nbsp; Deep Learning From Scratch
#
# Work each one on paper before reading. The eigenvalue arithmetic in
# Exercise 1 is worth doing by hand at least once — after that, use NumPy.

# %%
import math

import numpy as np

X_DATA = [1.0, 2.0, 3.0]
Y_DATA = [2.0, 3.0, 5.0]

LAM_MAX = (34 + math.sqrt(1060)) / 6
LAM_MIN = (34 - math.sqrt(1060)) / 6
KAPPA = LAM_MAX / LAM_MIN
ETA_STAR = 3 / 17
RHO = (KAPPA - 1) / (KAPPA + 1)


def hessian(xs):
    m = len(xs)
    return np.array([[2 / m * sum(x * x for x in xs), 2 / m * sum(xs)],
                     [2 / m * sum(xs), 2.0]])


# %% [markdown]
# ## Exercise 1 — the Hessian for $x = [0, 1]$
#
# With $m = 2$, $\sum x^2 = 0 + 1 = 1$, $\sum x = 0 + 1 = 1$:
#
# $$H = \frac{2}{2}\begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix}
#     = \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix}$$
#
# $\operatorname{tr}H = 3$, $\det H = 2 - 1 = 1$, so
#
# $$\lambda = \frac{3 \pm \sqrt{9 - 4}}{2} = \frac{3 \pm \sqrt5}{2}$$
#
# $$\lambda_{\max} = \frac{3+\sqrt5}{2} \approx 2.618034,
#   \qquad \lambda_{\min} = \frac{3-\sqrt5}{2} \approx 0.381966$$
#
# Those are $\varphi^2$ and $\varphi^{-2}$ for the golden ratio — a coincidence
# of this particular dataset, but a memorable one.
#
# $$\kappa = \frac{3+\sqrt5}{3-\sqrt5} \approx 6.854$$
#
# Far gentler than our $\kappa = 46.1$. The reason is visible in the data:
# $x = [0,1]$ is already roughly centred and of unit spread, whereas
# $x = [1,2,3]$ has a mean of 2, so the slope and intercept are strongly
# coupled. **Centring the inputs is most of what standardization buys you.**

# %%
ex1_H = [[1.0, 1.0], [1.0, 2.0]]
ex1_lam_max = (3 + math.sqrt(5)) / 2
ex1_lam_min = (3 - math.sqrt(5)) / 2

H1 = hessian([0.0, 1.0])
ev = np.linalg.eigvalsh(H1)
print(f"  H          = {H1.tolist()}")
print(f"  trace, det = {np.trace(H1):.1f}, {np.linalg.det(H1):.1f}")
print(f"  lambdas    = {ex1_lam_max:.6f}, {ex1_lam_min:.6f}   numpy {ev[::-1].round(6)}")
print(f"  kappa      = {ex1_lam_max / ex1_lam_min:.6f}  vs {KAPPA:.3f} for x=[1,2,3]")
assert np.allclose(np.array(ex1_H), H1)
assert abs(ex1_lam_max - ev[1]) < 1e-9 and abs(ex1_lam_min - ev[0]) < 1e-9
print(f"\n  max learning rate {2/ex1_lam_max:.4f}, converges {RHO:.4f} -> "
      f"{(ex1_lam_max/ex1_lam_min - 1)/(ex1_lam_max/ex1_lam_min + 1):.4f} per step")

# %% [markdown]
# ## Exercise 2 — how $\lambda_{\max}$ scales with the inputs
#
# **Quadratically — but the constant is not 1.**
#
# $H = \frac{2}{m}X^\top X$, and scaling $x$ by $c$ scales the top-left entry
# by $c^2$, the off-diagonal by $c$, and leaves the bottom-right at 2. So for
# large $c$ the matrix is dominated by that one entry and
#
# $$\lambda_{\max}(c\,x) \;\longrightarrow\; \frac{2}{m}c^2\sum_i x_i^2$$
#
# For our data $\frac{2}{3}\sum x_i^2 = \frac{2}{3}(14) = 9.3\overline{3}$,
# while $\lambda_{\max} = 11.09294$, so the ratio tends to
#
# $$\frac{9.3\overline{3}}{11.09294}\,c^2 \approx 0.8414\,c^2$$
#
# — which is exactly the 0.8414 the convergence table shows. The gap exists
# because **the bias column does not scale**: it stays at 1 no matter what you
# do to $x$. That asymmetry between a scaled feature and an unscaled bias is a
# small, permanent nuisance, and it is one more reason to centre and scale
# features rather than fight it.
#
# The practical consequence: **the maximum safe learning rate falls off like
# $1/c^2$.** Switch a feature from metres to centimetres and your learning
# rate must drop ten-thousand-fold.

# %%
def lam_max(xs):
    return np.linalg.eigvalsh(hessian(xs)).max()


base = lam_max(X_DATA)
ex2_ratio = lam_max([10 * v for v in X_DATA]) / base

print(f"{'c':>7} {'lambda_max':>14} {'ratio':>14} {'ratio / c^2':>13}")
for c in (1, 2, 10, 100, 1000):
    lm = lam_max([c * v for v in X_DATA])
    print(f"{c:>7} {lm:>14,.2f} {lm / base:>14,.2f} {lm / base / c**2:>13.4f}")

limit = (2 / 3 * sum(v * v for v in X_DATA)) / base
print(f"\npredicted limit of ratio/c^2 = (2/3)(14)/lambda_max = {limit:.4f}")
assert abs(lam_max([1000 * v for v in X_DATA]) / base / 1e6 - limit) < 1e-3
print("Confirmed to three decimals by c = 1000.")

# %% [markdown]
# ## Exercise 3 — the closed-form optimal learning rate
#
# Straight from section 2.4, with the $2\times2$ eigensolve inlined:
#
# $$\eta^{*} = \frac{2}{\lambda_{\max}+\lambda_{\min}} = \frac{2}{\operatorname{tr}H}$$
#
# — note you do not even need the eigenvalues for $\eta^*$, since the trace
# *is* their sum. You do need them for $\rho$.
#
# ### Centring is not the same as standardizing
#
# It is tempting to think $x = [-1, 0, 1]$ should give $\rho = 0$: it is
# symmetric about zero, so the off-diagonal term $\sum x_i$ vanishes and
# $H$ is diagonal. But diagonal is not the same as *isotropic*:
#
# $$\tfrac{1}{3}\textstyle\sum x_i^2 = \tfrac{2}{3}
#   \quad\Rightarrow\quad
#   H = \begin{bmatrix} 4/3 & 0 \\ 0 & 2 \end{bmatrix},
#   \quad \kappa = 1.5, \quad \rho = 0.2$$
#
# Good — nine steps instead of 319 — but not one step. For $H = 2I$ you need
# $\frac{1}{m}\sum z_i^2 = 1$ as well, i.e. **unit standard deviation**. The
# standardized version of $[1,2,3]$ is $[-\sqrt{3/2},\,0,\,\sqrt{3/2}]
# \approx [-1.2247,\,0,\,1.2247]$, and *that* gives $\rho = 0$ exactly.
#
# **Centring decouples the parameters; scaling equalises the curvature.** Both
# are needed, which is why the standard recipe is always subtract-the-mean
# *and* divide-by-the-std.

# %%
def optimal_lr(xs):
    H = hessian(xs)
    tr = H[0, 0] + H[1, 1]
    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
    root = math.sqrt(max(0.0, tr * tr - 4 * det))
    lo_max, lo_min = (tr + root) / 2, (tr - root) / 2
    kappa = lo_max / lo_min
    return 2 / (lo_max + lo_min), (kappa - 1) / (kappa + 1)


eta, rho = optimal_lr(X_DATA)
print(f"  x = [1,2,3]   eta* = {eta:.8f} (= 3/17 = {3/17:.8f})   rho = {rho:.8f}")
assert abs(eta - ETA_STAR) < 1e-12 and abs(rho - RHO) < 1e-12

std_x = [(v - 2.0) / math.sqrt(2 / 3) for v in X_DATA]
print()
for xs, note in [(X_DATA, "raw: neither centred nor scaled"),
                 ([-1.0, 0.0, 1.0], "centred only"),
                 (std_x, "centred AND scaled"),
                 ([0.0, 1.0], "the Exercise 1 data"),
                 ([100.0, 200.0, 300.0], "terrible units")]:
    e, r = optimal_lr(xs)
    steps = "1" if r < 1e-12 else f"{math.log(1e-6) / math.log(r):,.0f}"
    label = "[" + ", ".join(f"{v:g}" for v in xs) + "]"
    print(f"  {label:<26} rho = {r:.8f}  ~{steps:>9} steps   {note}")

assert abs(optimal_lr([-1.0, 0.0, 1.0])[1] - 0.2) < 1e-9, "centred-only gives rho = 0.2"
assert abs(optimal_lr(std_x)[1]) < 1e-12, "fully standardized gives rho = 0"
print(f"\n  H for the standardized data =\n{hessian(std_x).round(12)}")

# %% [markdown]
# ## Exercise 4 — the geometric law, measured
#
# Theory says the distance to the optimum shrinks by a constant factor $\rho$
# every step. Measure it by taking the ratio of consecutive distances once the
# transient has died away.
#
# Take the ratio **late** — early steps are contaminated by the fast
# eigendirection, which dies off quickly and leaves the slow one dominating.
# That is exactly why the asymptotic rate depends only on $\kappa$.

# %%
W_STAR, B_STAR = 1.5, 1 / 3


def measured_rho(xs, ys, steps=50):
    eta, _ = optimal_lr(xs)
    xa, ya = np.array(xs), np.array(ys)
    m = len(xs)
    w = b = 0.0
    dists = []
    for _ in range(steps + 2):
        dists.append(math.hypot(w - W_STAR, b - B_STAR))
        e = w * xa + b - ya
        w, b = w - eta * 2 / m * np.sum(e * xa), b - eta * 2 / m * np.sum(e)
    return dists[-1] / dists[-2]


got = measured_rho(X_DATA, Y_DATA)
print(f"  measured  rho = {got:.8f}")
print(f"  predicted rho = {RHO:.8f}   (kappa-1)/(kappa+1) with kappa = {KAPPA:.4f}")
print(f"  difference    = {abs(got - RHO):.2e}")
assert abs(got - RHO) < 1e-3

print("\n  how the measurement settles onto the theoretical rate:")
for n in (2, 5, 10, 25, 50, 100):
    print(f"    after {n:>3} steps: {measured_rho(X_DATA, Y_DATA, n):.8f}")
print(f"    theory          : {RHO:.8f}")

# %% [markdown]
# ## Exercise 5 — a singular Hessian
#
# **Any dataset where every $x$ is identical**, e.g. `[2, 2, 2]`.
#
# $$H = \frac{2}{3}\begin{bmatrix} 12 & 6 \\ 6 & 3\end{bmatrix}
#     = \begin{bmatrix} 8 & 4 \\ 4 & 2\end{bmatrix},
#   \qquad \det H = 16 - 16 = 0$$
#
# The second row is half the first, so the matrix has rank 1 and one
# eigenvalue is exactly zero.
#
# **What it means.** With every $x$ equal to 2, the prediction is $2w + b$ — a
# single number. Any $(w, b)$ on the line $2w + b = \text{const}$ fits the data
# identically well, so the minimum is a *line*, not a point. The data simply
# cannot distinguish slope from intercept.
#
# Along that zero-curvature direction the gradient is zero, so gradient descent
# never moves: whatever component of the initialisation lies along it is frozen
# there permanently. The model still predicts correctly — it is only the
# *parameters* that are unidentifiable.
#
# The general condition is that the design matrix loses rank, which happens
# whenever one feature is an exact linear combination of the others. In real
# work this is **collinearity**, and it is why adding a feature that is the sum
# of two existing ones makes a model's coefficients meaningless while leaving
# its predictions untouched. L2 regularization (Lesson 17) fixes it by adding
# $\lambda I$ to the Hessian, lifting the zero eigenvalue off the floor.

# %%
ex5_xs = [2.0, 2.0, 2.0]
H5 = hessian(ex5_xs)
print(f"  H = {H5.tolist()}   det = {np.linalg.det(H5):.2e}")
print(f"  eigenvalues = {np.linalg.eigvalsh(H5).round(10)}")
print(f"  rank of H   = {np.linalg.matrix_rank(H5)} (out of 2)")
assert abs(np.linalg.det(H5)) < 1e-9

# every (w, b) on this line is equally optimal
print("\n  parameter pairs that all fit the data identically:")
for w in (0.0, 1.0, 2.0):
    b = 3.0 - 2 * w
    pred = [w * x + b for x in ex5_xs]
    loss = sum((p - y) ** 2 for p, y in zip(pred, Y_DATA)) / 3
    print(f"    w = {w:.1f}, b = {b:>5.1f}  ->  prediction {pred[0]:.1f}  loss {loss:.6f}")
print("\n  Identical loss, completely different parameters. Nothing to choose between them.")

# %% [markdown]
# ## Exercise 6 — the general case
#
# Nothing changes except the size of the matrix. The full eigenspectrum of
# $H = \frac{2}{m}X^\top X$ gives $\kappa = \lambda_{\max}/\lambda_{\min}$ and
# $\eta^* = 2/(\lambda_{\max}+\lambda_{\min})$ exactly as before — the
# derivation in section 2.4 never assumed two dimensions.
#
# Use `eigvalsh`, not `eigvals`: $H$ is symmetric, and the symmetric solver is
# both faster and guaranteed to return real eigenvalues rather than ones with
# $10^{-17}i$ attached.
#
# The result below is the practical payoff of the whole lesson. Features with
# standard deviations of 50 and 0.2 give $\kappa$ in the tens of thousands;
# standardizing brings it near 1. **That single preprocessing line is worth
# more than most optimiser tuning.**

# %%
def surface_facts_multi(X):
    m = len(X)
    Xb = np.hstack([X, np.ones((m, 1))])
    ev = np.linalg.eigvalsh(2 / m * Xb.T @ Xb)
    return ev.max() / ev.min(), 2 / (ev.max() + ev.min())


np.random.seed(3)
Xm = np.random.randn(80, 4) @ np.diag([1.0, 5.0, 0.2, 50.0])

kap_raw, eta_raw = surface_facts_multi(Xm)
Z = (Xm - Xm.mean(0)) / Xm.std(0)
kap_std, eta_std = surface_facts_multi(Z)

print(f"  feature std devs : {Xm.std(0).round(3)}")
print(f"  raw          kappa = {kap_raw:>12,.1f}   eta* = {eta_raw:.3e}")
print(f"  standardized kappa = {kap_std:>12,.1f}   eta* = {eta_std:.3e}")
print(f"  improvement        : {kap_raw / kap_std:,.0f}x better conditioned")

Xb = np.hstack([Xm, np.ones((80, 1))])
ev = np.linalg.eigvalsh(2 / 80 * Xb.T @ Xb)
assert abs(kap_raw - ev.max() / ev.min()) < 1e-6
assert abs(eta_raw - 2 / (ev.max() + ev.min())) < 1e-9

r_raw = (kap_raw - 1) / (kap_raw + 1)
r_std = (kap_std - 1) / (kap_std + 1)
print(f"\n  steps to 1e-6, raw          : {math.log(1e-6)/math.log(r_raw):>10,.0f}")
print(f"  steps to 1e-6, standardized : {math.log(1e-6)/math.log(r_std):>10,.0f}")
print("""
Standardizing does not change the model, the data, or the answer -- the fitted
predictions are identical. It changes only the SHAPE of the surface gradient
descent has to walk on. That is the entire lesson.""")

# %% [markdown]
# ---
#
# **Back to** `notebooks/07-loss-surfaces.ipynb` &nbsp;|&nbsp;
# **Next:** Lesson 08 — Logistic Regression and the Sigmoid
