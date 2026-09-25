# %% [markdown]
# # Lesson 06 - Solutions
#
# **Linear Regression from Scratch** &nbsp;|&nbsp; Deep Learning From Scratch
#
# Attempt each one on paper before reading. A solution you have not struggled
# with teaches you that the answer looks reasonable, which is not the same as
# being able to produce it.

# %%
import numpy as np

X_DATA = [1.0, 2.0, 3.0]
Y_DATA = [2.0, 3.0, 5.0]
LR = 0.1


def mse(w, b, xs, ys):
    return sum((w * x + b - y) ** 2 for x, y in zip(xs, ys)) / len(xs)


def gradients(w, b, xs, ys):
    m = len(xs)
    errs = [w * x + b - y for x, y in zip(xs, ys)]
    return 2 / m * sum(e * x for e, x in zip(errs, xs)), 2 / m * sum(errs)


# %% [markdown]
# ## Exercise 1 — the second step
#
# Starting from $w_1 = \tfrac{23}{15}$, $b_1 = \tfrac{2}{3}$:
#
# **Predictions.** $\hat{y} = \tfrac{23}{15}x + \tfrac{2}{3}$, giving
# $2.2,\ 3.7\overline{3},\ 5.2\overline{6}$.
#
# **Errors.** $e = \hat{y} - y$, so
# $e^{(1)} = \tfrac{1}{5},\quad e^{(2)} = \tfrac{11}{15},\quad e^{(3)} = \tfrac{4}{15}$
#
# Every error is now **positive** — after one big step the line has overshot
# and sits above all three points.
#
# **Gradients.**
#
# $$\frac{\partial L}{\partial w} = \frac{2}{3}\left[\tfrac{1}{5}(1) + \tfrac{11}{15}(2) + \tfrac{4}{15}(3)\right]
#  = \frac{2}{3}\cdot\frac{37}{15} = \frac{74}{45} \approx 1.64444$$
#
# $$\frac{\partial L}{\partial b} = \frac{2}{3}\left[\tfrac{1}{5} + \tfrac{11}{15} + \tfrac{4}{15}\right]
#  = \frac{2}{3}\cdot\frac{6}{5} = \frac{4}{5} = 0.8$$
#
# **Update.**
#
# $$w_2 = \tfrac{23}{15} - 0.1\cdot\tfrac{74}{45} = \tfrac{308}{225} \approx 1.36889$$
#
# $$b_2 = \tfrac{2}{3} - 0.1\cdot\tfrac{4}{5} = \tfrac{44}{75} \approx 0.58667$$
#
# **Both parameters went *down* this step**, having gone up last step. That is
# the overshoot-and-correct oscillation you saw as a zig-zag in section 4.2.

# %%
ex1_w2, ex1_b2 = 308 / 225, 44 / 75
gw, gb = gradients(23 / 15, 2 / 3, X_DATA, Y_DATA)
print(f"gradients : dL/dw = {gw:.6f}   dL/db = {gb:.6f}")
print(f"w2 = {ex1_w2:.6f}   b2 = {ex1_b2:.6f}")
print(f"loss went {mse(23/15, 2/3, X_DATA, Y_DATA):.6f} -> {mse(ex1_w2, ex1_b2, X_DATA, Y_DATA):.6f}")
assert abs(23 / 15 - LR * gw - ex1_w2) < 1e-12
assert abs(2 / 3 - LR * gb - ex1_b2) < 1e-12
print("\nStep 1 raised both; step 2 lowered both. Oscillation, converging.")

# %% [markdown]
# ## Exercise 2 — MAE gradients
#
# $$L = \frac{1}{m}\sum_i \left|e^{(i)}\right|, \qquad e^{(i)} = wx^{(i)} + b - y^{(i)}$$
#
# Since $\frac{d|u|}{du} = \operatorname{sign}(u)$, the chain rule gives
#
# $$\frac{\partial L}{\partial w} = \frac{1}{m}\sum_i \operatorname{sign}(e^{(i)})\,x^{(i)},
#   \qquad
#   \frac{\partial L}{\partial b} = \frac{1}{m}\sum_i \operatorname{sign}(e^{(i)})$$
#
# At $w = b = 0$ every error is negative, so every sign is $-1$:
#
# $$\frac{\partial L}{\partial w} = \frac{-(1 + 2 + 3)}{3} = -2,
#   \qquad \frac{\partial L}{\partial b} = \frac{-3}{3} = -1$$
#
# **The crucial difference from MSE.** The magnitude of the error has vanished
# from the gradient entirely — only its *sign* survives. Consequences:
#
# - **Robust to outliers.** A point that is wrong by 1000 pulls exactly as hard
#   as one wrong by 1. That is why Exercise 3's outlier would barely move an
#   MAE fit.
# - **Awkward to converge.** The gradient does not shrink as you approach the
#   answer, so a fixed learning rate makes you bounce around the minimum
#   forever instead of settling. MAE needs a decaying learning rate.
# - **Not differentiable at $e = 0$.** `sign(0)` is a convention, not a
#   derivative — this is a *subgradient*. The same wrinkle appears with ReLU
#   at exactly 0 in Lesson 12.

# %%
def sign(u):
    return (u > 0) - (u < 0)


def mae_gradients(w, b, xs, ys):
    m = len(xs)
    errs = [w * x + b - y for x, y in zip(xs, ys)]
    return (sum(sign(e) * x for e, x in zip(errs, xs)) / m,
            sum(sign(e) for e in errs) / m)


assert abs(mae_gradients(0.0, 0.0, X_DATA, Y_DATA)[0] - (-2.0)) < 1e-9
assert abs(mae_gradients(0.0, 0.0, X_DATA, Y_DATA)[1] - (-1.0)) < 1e-9
print("at (0,0)      MAE:", tuple(round(g, 4) for g in mae_gradients(0., 0., X_DATA, Y_DATA)),
      "  MSE:", tuple(round(g, 4) for g in gradients(0., 0., X_DATA, Y_DATA)))
print("at (1.4, 0.4) MAE:", tuple(round(g, 4) for g in mae_gradients(1.4, .4, X_DATA, Y_DATA)),
      "  MSE:", tuple(round(g, 4) for g in gradients(1.4, .4, X_DATA, Y_DATA)))
print("\nMSE's gradient shrank by ~10x as the fit improved. MAE's barely moved.")

# %% [markdown]
# ## Exercise 3 — one outlier, and what it does to the slope
#
# With $x = [1,2,3,4]$, $y = [2,3,5,20]$: $\bar{x} = 2.5$, $\bar{y} = 7.5$.
#
# $$\sum (x-\bar{x})^2 = 2.25 + 0.25 + 0.25 + 2.25 = 5$$
#
# $$\sum (x-\bar{x})(y-\bar{y}) = (-1.5)(-5.5) + (-0.5)(-4.5) + (0.5)(-2.5) + (1.5)(12.5)$$
# $$= 8.25 + 2.25 - 1.25 + 18.75 = 28$$
#
# $$w^* = \frac{28}{5} = 5.6, \qquad b^* = 7.5 - 5.6(2.5) = -6.5$$
#
# The slope went from **1.5 to 5.6** — nearly four times steeper — because of
# a single point. Worse, the intercept went *negative*, so the model now
# predicts $-0.9$ at $x = 1$ where the data says 2. One bad row has made the
# model wrong about every other row.
#
# That single point contributes $(1.5)(12.5) = 18.75$ of the 28 in the
# numerator: **67% of the fitted slope comes from one of four samples.** This
# is the practical case for either cleaning your data or choosing a loss that
# does not square the errors.

# %%
ex3_w = 5.6
Xo = np.array([1., 2., 3., 4.])
Yo = np.array([2., 3., 5., 20.])
D = np.stack([Xo, np.ones(4)], axis=1)
theta = np.linalg.solve(D.T @ D, D.T @ Yo)
print(f"with outlier   : w = {theta[0]:.4f}, b = {theta[1]:.4f}")
print(f"without it     : w = 1.5000, b = 0.3333")
print(f"prediction at x=1: {theta[0] + theta[1]:.4f}   actual y: 2.0")
assert abs(ex3_w - theta[0]) < 1e-9
print("\nMSE squares the error, so a point 12.5 above the mean gets 12.5x the")
print("leverage of one 1 above it. Squaring is what makes outliers so loud.")

# %% [markdown]
# ## Exercise 4 — stopping when it stops helping
#
# Track the previous loss, and halt as soon as the improvement drops below
# `tol`. Two details matter:
#
# - Compare the **improvement** `prev - now`, not the loss itself. A loss of
#   0.0556 is not small in absolute terms here — it is the best achievable.
# - Guard against the loss *rising*. If the learning rate is too large the
#   improvement goes negative, and a naive `< tol` test would stop and report
#   success on a diverging run.

# %%
def train_until_converged(xs, ys, lr=LR, tol=1e-12, max_steps=100000):
    w = b = 0.0
    prev = mse(w, b, xs, ys)
    for step in range(1, max_steps + 1):
        gw, gb = gradients(w, b, xs, ys)
        w, b = w - lr * gw, b - lr * gb
        now = mse(w, b, xs, ys)
        if 0 <= prev - now < tol:          # improving, but no longer usefully
            return w, b, step
        prev = now
    return w, b, max_steps


w, b, steps = train_until_converged(X_DATA, Y_DATA)
print(f"converged in {steps} steps: w = {w:.8f}, b = {b:.8f}")
print(f"target                    : w = 1.50000000, b = 0.33333333")
assert abs(w - 1.5) < 1e-4 and abs(b - 1 / 3) < 1e-4

for tol in (1e-6, 1e-9, 1e-12, 1e-15):
    _w, _b, s = train_until_converged(X_DATA, Y_DATA, tol=tol)
    print(f"  tol {tol:<7g} -> {s:>5} steps, w = {_w:.8f}  (error {abs(_w - 1.5):.2e})")
print("\nA thousand times tighter tolerance buys maybe twice the steps -- the")
print("loss falls geometrically, so late steps are cheap. Early stopping in")
print("Lesson 16 uses the same machinery for a completely different reason:")
print("there we stop early to avoid overfitting, not to save time.")

# %% [markdown]
# ## Exercise 5 — doubling the targets
#
# **New values: $w^* = 3$, $b^* = \tfrac{2}{3}$ — both exactly doubled.**
#
# The closed form is *linear in $y$*. Look at where $y$ appears:
#
# $$w^* = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sum (x - \bar{x})^2}$$
#
# The numerator contains $y - \bar{y}$ to the first power and the denominator
# has no $y$ at all, so scaling every $y$ by $c$ scales $w^*$ by $c$. Then
# $b^* = \bar{y} - w^*\bar{x}$ scales by $c$ as well, since both terms do.
#
# **Doubling $x$ instead does something quite different.** The numerator picks
# up one factor of $c$, the denominator picks up $c^2$, so
#
# $$w^* \rightarrow \frac{w^*}{c}, \qquad b^* \rightarrow b^* \text{ (unchanged)}$$
#
# which is exactly right: if you measure the same distance in centimetres
# instead of metres, the slope per unit must shrink 100-fold while the
# intercept — the value at $x=0$ — cannot move. This asymmetry is the reason
# rescaling inputs changes the safe learning rate quadratically, as section 9.2
# showed.

# %%
def closed_form(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    w = ((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum()
    return w, ys.mean() - w * xs.mean()


print(f"  original            : w = {closed_form(X_DATA, Y_DATA)[0]:.6f}, b = {closed_form(X_DATA, Y_DATA)[1]:.6f}")
print(f"  y doubled           : w = {closed_form(X_DATA, [2*v for v in Y_DATA])[0]:.6f}, "
      f"b = {closed_form(X_DATA, [2*v for v in Y_DATA])[1]:.6f}")
print(f"  x doubled           : w = {closed_form([2*v for v in X_DATA], Y_DATA)[0]:.6f}, "
      f"b = {closed_form([2*v for v in X_DATA], Y_DATA)[1]:.6f}")
assert abs(closed_form(X_DATA, [2 * v for v in Y_DATA])[0] - 3.0) < 1e-9
assert abs(closed_form(X_DATA, [2 * v for v in Y_DATA])[1] - 2 / 3) < 1e-9
assert abs(closed_form([2 * v for v in X_DATA], Y_DATA)[0] - 0.75) < 1e-9
print("\ny doubled -> both double.  x doubled -> w halves, b unchanged.")

# %% [markdown]
# ## Exercise 6 — the general linear model
#
# Nothing new is needed. Append a column of ones so the bias becomes just
# another weight, and the two scalar gradients collapse into one matrix
# expression:
#
# $$\nabla_\theta L = \frac{2}{m}X^{\top}(X\theta - y)$$
#
# Check the shapes, since that is where this goes wrong: $X$ is $(m, n{+}1)$,
# $\theta$ is $(n{+}1,)$, so $X\theta - y$ is $(m,)$, and
# $X^{\top}(m{\times}1)$ gives $(n{+}1,)$ — one gradient per parameter, exactly
# as required.
#
# This single expression is every linear layer in every network you will build.
# Lesson 10 stacks two of them with a nonlinearity in between; Lesson 11 works
# out how the gradient flows back through the stack.

# %%
def fit_multi(X, y, lr=0.05, steps=20000):
    m = len(y)
    Xb = np.hstack([X, np.ones((m, 1))])          # append the bias column
    theta = np.zeros(Xb.shape[1])
    for _ in range(steps):
        theta -= lr * (2 / m * Xb.T @ (Xb @ theta - y))
    return theta


np.random.seed(1)
Xm = np.random.randn(50, 3)
true = np.array([2.0, -1.0, 0.5, 3.0])
ym = Xm @ true[:3] + true[3] + 0.01 * np.random.randn(50)

theta = fit_multi(Xm, ym)
Xb = np.hstack([Xm, np.ones((50, 1))])
exact = np.linalg.solve(Xb.T @ Xb, Xb.T @ ym)

print(f"  gradient descent : {np.round(theta, 6)}")
print(f"  normal equation  : {np.round(exact, 6)}")
print(f"  true parameters  : {true}")
assert np.allclose(theta, exact, atol=1e-6)
print(f"\n  agreement to {np.abs(theta - exact).max():.2e}")
print("""
The recovered parameters are not exactly the true ones, and should not be:
we added noise to y, so the best fit to THIS sample differs slightly from the
process that generated it. Telling those two apart -- fitting the data versus
fitting the world -- is the whole subject of Lesson 16.""")

# %% [markdown]
# ---
#
# **Back to** `notebooks/06-linear-regression.ipynb` &nbsp;|&nbsp;
# **Next:** Lesson 07 — Loss Surfaces and MSE
