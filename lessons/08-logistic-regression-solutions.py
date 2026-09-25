# %% [markdown]
# # Lesson 08 - Solutions
#
# **Logistic Regression and the Sigmoid** &nbsp;|&nbsp; Deep Learning From Scratch
#
# Exercise 5 is the one to spend time on. It fixes, in four lines, the runaway
# from section 9.2 — and it is the whole argument for Lesson 17.

# %%
import math

import numpy as np

X_DATA = [-1.0, 0.0, 2.0]
Y_DATA = [0.0, 0.0, 1.0]


def stable_sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    t = math.exp(z)
    return t / (1.0 + t)


def bce_gradients(w, b, xs, ys):
    m = len(xs)
    gw = gb = 0.0
    for x, y in zip(xs, ys):
        e = stable_sigmoid(w * x + b) - y
        gw += e * x
        gb += e
    return gw / m, gb / m


# %% [markdown]
# ## Exercise 1 — the second step
#
# From $w_1 = \tfrac12$, $b_1 = -\tfrac16$, the predictions were already
# printed in section 3.5:
#
# $$\sigma = [0.3392436,\ 0.4584295,\ 0.6970593], \qquad y = [0,\ 0,\ 1]$$
#
# so the errors $\sigma - y$ are
# $[0.3392436,\ 0.4584295,\ -0.3029407]$ and
#
# $$\frac{\partial L}{\partial w} = \frac{1}{3}\big[(0.3392436)(-1) + (0.4584295)(0) + (-0.3029407)(2)\big]
#  = \frac{-0.9451250}{3} = -0.3150417$$
#
# $$\frac{\partial L}{\partial b} = \frac{1}{3}\big[0.3392436 + 0.4584295 - 0.3029407\big] = 0.1649108$$
#
# $$w_2 = 0.5 + 0.3150417 = 0.8150417, \qquad b_2 = -0.1666667 - 0.1649108 = -0.3315775$$
#
# Compare the gradient magnitudes: step one had $\partial L/\partial w = -0.5$,
# step two has $-0.315$. The gradient is shrinking as the fit improves — the
# opposite of the MAE behaviour in Lesson 06, and a sign of a well-matched
# loss.

# %%
ex1_w2, ex1_b2 = 0.8150417, -0.3315775
gw, gb = bce_gradients(0.5, -1 / 6, X_DATA, Y_DATA)
print(f"  errors   : {[round(stable_sigmoid(0.5*x - 1/6) - y, 7) for x, y in zip(X_DATA, Y_DATA)]}")
print(f"  dL/dw    : {gw:.7f}      dL/db: {gb:.7f}")
print(f"  w2, b2   : {0.5 - gw:.7f}, {-1/6 - gb:.7f}")
assert abs(ex1_w2 - (0.5 - gw)) < 1e-6 and abs(ex1_b2 - (-1 / 6 - gb)) < 1e-6
print(f"\n  |dL/dw| fell from 0.5000 to {abs(gw):.4f} in one step.")

# %% [markdown]
# ## Exercise 2 — the loss of knowing nothing
#
# **Binary: $\log 2 \approx 0.693147$. Ten classes: $\log 10 \approx 2.302585$.**
#
# If the model outputs $\tfrac1k$ for every class, then whichever class is
# correct, the loss contributed is
#
# $$-\log\frac{1}{k} = \log k$$
#
# independent of the labels entirely. So $\log k$ is the score of *maximum
# ignorance*, and it is also the best you can do when the input genuinely
# carries no information about the label.
#
# This is the most useful diagnostic number in classification:
#
# - loss stuck at $\log k$ → the model has learned **nothing**. Usually a
#   broken data pipeline, a learning rate of zero, or labels shuffled relative
#   to inputs.
# - loss *above* $\log k$ → worse than guessing. Usually inverted labels, or a
#   learning rate so high the model has diverged.
# - loss just below $\log k$ and refusing to move → the features may simply not
#   predict the target.
#
# Always print $\log k$ before you start training, so you know what "no
# progress" looks like.

# %%
ex2_binary, ex2_tenclass = math.log(2), math.log(10)
print(f"{'classes':>9} {'log k':>10} {'accuracy of random guessing':>30}")
for k in (2, 3, 10, 100, 1000):
    print(f"{k:>9} {math.log(k):>10.6f} {1/k:>30.3%}")
assert abs(ex2_binary - math.log(2)) < 1e-12
assert abs(ex2_tenclass - math.log(10)) < 1e-12
print("\n  ImageNet has 1000 classes, so an untrained model starts near 6.9.")

# %% [markdown]
# ## Exercise 3 — a branchless stable sigmoid
#
# The identity to exploit is $\sigma(-z) = 1 - \sigma(z)$, but the neat trick
# is arranging for `exp` to only ever see a non-positive argument:
#
# ```python
# math.exp(min(z, 0.0)) / (1.0 + math.exp(-abs(z)))
# ```
#
# Check both branches:
#
# - $z \ge 0$: numerator is $e^{0} = 1$, denominator is $1 + e^{-z}$ → the
#   standard form.
# - $z < 0$: $\min(z,0) = z$ and $-|z| = z$, giving $e^{z}/(1 + e^{z})$ → the
#   algebraically equal safe form.
#
# Both `exp` calls have arguments $\le 0$, so the worst that can happen is
# **underflow to 0**, which is harmless. Overflow is now impossible for any
# finite input. `numpy` does the same thing internally in `expit`.
#
# At $z = -1000$ this returns exactly `0.0` — which is correct to every bit of
# float64 precision available, since the true value is about $10^{-435}$, far
# below the smallest representable double.

# %%
def one_line_sigmoid(z):
    return math.exp(min(z, 0.0)) / (1.0 + math.exp(-abs(z)))


for z in (-1000.0, -50.0, -1.0, 0.0, 1.0, 50.0, 1000.0):
    a, b = one_line_sigmoid(z), stable_sigmoid(z)
    print(f"  z = {z:>8}   {a:.17g}   matches branching version: {abs(a - b) < 1e-15}")
    assert abs(a - b) < 1e-12

try:
    1.0 / (1.0 + math.exp(-(-1000.0)))
except OverflowError as e:
    print(f"\n  the naive form at z = -1000 raises OverflowError: {e}")

# %% [markdown]
# ## Exercise 4 — L2 on the weight, not the bias
#
# $$L = \text{BCE} + \frac{\lambda}{2}w^2
#   \quad\Longrightarrow\quad
#   \frac{\partial L}{\partial w} = \frac{1}{m}\sum_i(\sigma^{(i)} - y^{(i)})x^{(i)} + \lambda w$$
#
# The $\tfrac{\lambda}{2}$ is chosen precisely so the derivative comes out as a
# clean $\lambda w$ — the 2 from the power rule cancels it.
#
# ### Why the bias is exempt
#
# The penalty exists to stop the *boundary getting sharper* than the evidence
# warrants. The slope $w$ controls sharpness: doubling $w$ doubles every logit
# and pushes every probability toward 0 or 1. The bias only controls *where*
# the boundary sits.
#
# Penalising $b$ would drag the decision boundary toward $x = 0$, which is a
# statement about your coordinate system, not about your data. Shift every $x$
# by 100 and a bias penalty would suddenly mean something completely different
# — the model would stop being equivariant to a change of origin. Every serious
# implementation exempts the bias for this reason, and the same argument covers
# batch-norm shifts in Lesson 19.

# %%
def l2_gradients(w, b, xs, ys, lam):
    gw, gb = bce_gradients(w, b, xs, ys)
    return gw + lam * w, gb          # the bias is deliberately untouched


base = bce_gradients(2.0, -1.0, X_DATA, Y_DATA)
pen = l2_gradients(2.0, -1.0, X_DATA, Y_DATA, 0.5)
print(f"  plain    dL/dw = {base[0]:>9.6f}   dL/db = {base[1]:>9.6f}")
print(f"  with L2  dL/dw = {pen[0]:>9.6f}   dL/db = {pen[1]:>9.6f}")
print(f"  difference       {pen[0] - base[0]:>9.6f}   {pen[1] - base[1]:>9.6f}   = lambda*w, and 0")
assert abs(pen[0] - (base[0] + 0.5 * 2.0)) < 1e-12
assert abs(pen[1] - base[1]) < 1e-12

# %% [markdown]
# ## Exercise 5 — the runaway, stopped
#
# Section 9.2 ran 200,000 steps and reached $w = 10.56$, still climbing. With
# $\lambda = 0.1$ the weight settles at **1.4996** and stays there — five times
# more steps moves it by less than $10^{-6}$.
#
# The mechanism is a balance of two forces:
#
# - The data term always wants $|w|$ larger, because on separable data every
#   increase makes every prediction more confident and the loss lower. Its pull
#   weakens as $\sigma$ saturates, decaying like $e^{-w}$.
# - The penalty term pulls back with force $\lambda w$, which **grows**
#   linearly with $w$.
#
# One force decays exponentially, the other grows linearly, so they must cross
# — and they cross at exactly one point. That crossing is the minimum, and it
# now exists at a finite $w$ where before there was none.
#
# The payoff is calibration. Unregularized, the model eventually predicts
# $P = 0.99999$ from three data points, which is an absurd claim. Regularized,
# it predicts something defensible.

# %%
def train_l2(xs, ys, lam=0.1, lr=0.5, steps=20000):
    w = b = 0.0
    for _ in range(steps):
        gw, gb = l2_gradients(w, b, xs, ys, lam)
        w, b = w - lr * gw, b - lr * gb
    return w, b


def train_plain(xs, ys, lr=0.5, steps=20000):
    w = b = 0.0
    for _ in range(steps):
        gw, gb = bce_gradients(w, b, xs, ys)
        w, b = w - lr * gw, b - lr * gb
    return w, b


print(f"{'steps':>9} {'w (no penalty)':>17} {'w (lambda=0.1)':>17}")
for n in (1_000, 20_000, 100_000):
    print(f"{n:>9,} {train_plain(X_DATA, Y_DATA, steps=n)[0]:>17.6f} "
          f"{train_l2(X_DATA, Y_DATA, steps=n)[0]:>17.6f}")

w20, b20 = train_l2(X_DATA, Y_DATA)
w100, _ = train_l2(X_DATA, Y_DATA, steps=100_000)
assert abs(w20) < 10 and abs(w20 - w100) < 1e-3
print(f"\n  converged: w = {w20:.6f}, b = {b20:.6f}")

wp, bp = train_plain(X_DATA, Y_DATA, steps=100_000)
for name, (w, b) in [("unregularized", (wp, bp)), ("lambda = 0.1", (w20, b20))]:
    print(f"  {name:<14} P(y=1 | x=2) = {stable_sigmoid(w * 2 + b):.6f}")
print("""
  Three data points cannot justify 0.9999 confidence. The penalty is not
  hurting accuracy here -- both classify all three correctly -- it is
  stopping the model from lying about how sure it is.""")

# %% [markdown]
# ## Exercise 6 — many features
#
# The design-matrix form again, and once more nothing about the derivation
# changed:
#
# $$\nabla_\theta L = \frac{1}{m}X^\top\big(\sigma(X\theta) - y\big)$$
#
# Compare with Lesson 06's $\frac{2}{m}X^\top(X\theta - y)$: the only
# differences are the constant and a $\sigma$ wrapped around the prediction.
# The "$X^\top$ times error" skeleton is the same, and it is the skeleton of
# backprop too.
#
# ### On the recovered parameters
#
# They come out near the true values but not equal to them, and that is
# correct rather than a bug. The labels were **sampled** from those
# probabilities, not computed from them — a point with $P = 0.7$ came up 0
# about 30% of the time. The fit describes the 200 coin flips we happened to
# observe, not the coin. With 200 samples and 4 parameters that gap is large;
# with 20,000 it would mostly close.

# %%
def fit_logistic(X, y, lr=0.5, steps=5000):
    m = len(y)
    Xb = np.hstack([X, np.ones((m, 1))])
    theta = np.zeros(Xb.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Xb @ theta)))       # inputs are modest here
        theta -= lr * (Xb.T @ (p - y) / m)
    return theta


np.random.seed(2)
X = np.random.randn(200, 3)
true = np.array([1.5, -2.0, 0.5, 0.3])
p_true = 1 / (1 + np.exp(-(X @ true[:3] + true[3])))
y = (np.random.rand(200) < p_true).astype(float)

theta = fit_logistic(X, y)
Xb = np.hstack([X, np.ones((200, 1))])
pred = 1 / (1 + np.exp(-(Xb @ theta)))
acc = float(np.mean((pred >= 0.5) == y))

print(f"  recovered : {np.round(theta, 4)}")
print(f"  true      : {true}")
print(f"  accuracy  : {acc:.4f}")
assert acc > 0.8

# a gradient check on the vectorized form, to prove the derivation transferred
def loss(th):
    z = Xb @ th
    return float(np.mean(np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z)))))


analytic = Xb.T @ (1 / (1 + np.exp(-(Xb @ theta))) - y) / 200
numeric = np.array([
    (loss(theta + e) - loss(theta - e)) / 2e-6
    for e in 1e-6 * np.eye(4)
])
rel = np.max(np.abs(analytic - numeric) / np.maximum(1e-300, np.abs(analytic) + np.abs(numeric)))
print(f"\n  gradient check on the n-feature form: rel err {rel:.2e}")
assert rel < 1e-5 or np.max(np.abs(analytic - numeric)) < 1e-8

# and the point about sampled labels
print(f"\n  how well the fit recovers the TRUE probabilities:")
print(f"    mean |p_fitted - p_true| = {np.mean(np.abs(pred - p_true)):.4f}")
print(f"    best achievable accuracy given the sampling = "
      f"{np.mean((p_true >= 0.5) == y):.4f}")
print("""
  That last number is the ceiling: even a model that knew the true
  probabilities exactly would misclassify that many, because the labels
  were random draws. We are already close to it.""")

# %% [markdown]
# ---
#
# **Back to** `notebooks/08-logistic-regression.ipynb` &nbsp;|&nbsp;
# **Next:** Lesson 09 — The Perceptron and its Limits
