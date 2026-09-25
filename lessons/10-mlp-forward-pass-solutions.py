# %% [markdown]
# # Lesson 10 - Solutions
#
# **The MLP Forward Pass** &nbsp;|&nbsp; Deep Learning From Scratch
#
# Exercise 6 has the surprising result: extra hidden units add no expressive
# power to XOR whatsoever, and still take the success rate from 50% to 100%.

# %%
import math

import numpy as np

W1 = [[0.5, -1.0], [-0.5, 0.5]]
B1 = [0.0, 1.0]
W2 = [1.0, 2.0]
B2 = -0.5


def forward_scratch(x, W1_, b1_, W2_, b2_):
    z1 = [sum(x[i] * W1_[i][j] for i in range(len(x))) + b1_[j] for j in range(len(b1_))]
    h = [math.tanh(v) for v in z1]
    z2 = sum(h[j] * W2_[j] for j in range(len(W2_))) + b2_
    return z1, h, z2, 1 / (1 + math.exp(-z2))


# %% [markdown]
# ## Exercise 1 — a second forward pass
#
# With $\mathbf{x} = [0, 1]$:
#
# $$z^{(1)}_1 = (0)(0.5) + (1)(-0.5) + 0.0 = -0.5$$
# $$z^{(1)}_2 = (0)(-1.0) + (1)(0.5) + 1.0 = 1.5$$
#
# $$\mathbf{h} = [\tanh(-0.5),\ \tanh(1.5)] = [-0.4621172,\ 0.9051483]$$
#
# $$z^{(2)} = -0.4621172 + 2(0.9051483) - 0.5 = 0.8481794
#   \quad\Rightarrow\quad \hat{y} = \sigma(0.8481794) = 0.7001851$$
#
# Two things worth noticing by comparing with section 3.
#
# **$z^{(1)}_1$ is unchanged at $-0.5$**, even though the input changed. That
# is a coincidence of these particular numbers — $(1)(0.5)+(2)(-0.5)$ and
# $(0)(0.5)+(1)(-0.5)$ both give $-0.5$ — but it illustrates something real:
# a hidden unit sees only one projection of the input, and many different
# inputs collapse onto the same projection. Each unit is deliberately blind to
# everything outside its own direction.
#
# **The output rose from 0.6367 to 0.7002** because $h_2$ grew. Unit 2 carries
# weight $+2$, twice unit 1's, so it dominates. Reading off which units matter
# by the size of their outgoing weights is the crudest form of the
# interpretability work that Lesson 28 does properly.

# %%
ex1_z1, ex1_yhat = [-0.5, 1.5], 0.7001851
z1, h, z2, yh = forward_scratch([0.0, 1.0], W1, B1, W2, B2)
print(f"  z1   = {[round(v, 7) for v in z1]}")
print(f"  h    = {[round(v, 7) for v in h]}")
print(f"  z2   = {z2:.7f}")
print(f"  yhat = {yh:.7f}")
assert all(abs(a - b) < 1e-9 for a, b in zip(ex1_z1, z1))
assert abs(ex1_yhat - yh) < 1e-6

print(f"\n  section 3 (x = [1, 2]) gave yhat = 0.6367003")
print(f"  here      (x = [0, 1]) gives yhat = {yh:.7f}")

# %% [markdown]
# ## Exercise 2 — counting parameters
#
# **235,146.**
#
# $$\underbrace{784(256) + 256}_{200{,}960} +
#   \underbrace{256(128) + 128}_{32{,}896} +
#   \underbrace{128(10) + 10}_{1{,}290} = 235{,}146$$
#
# The first layer holds **85.5%** of them, and it is doing the least
# interesting work — connecting every pixel to every hidden unit, with no
# notion that neighbouring pixels are related. Move a digit one pixel to the
# right and every single one of those 200,960 weights sees different inputs.
#
# That waste is the entire motivation for convolution (Lesson 22), which
# replaces this layer with a few hundred shared parameters that slide across
# the image — fewer parameters, *and* translation awareness built in.

# %%
ex2_params = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
print(f"  {'layer':<18} {'weights':>12} {'biases':>8} {'total':>12} {'share':>8}")
running = []
for name, n_in, n_out in [("784 -> 256", 784, 256), ("256 -> 128", 256, 128),
                          ("128 -> 10", 128, 10)]:
    tot = n_in * n_out + n_out
    running.append(tot)
    print(f"  {name:<18} {n_in*n_out:>12,} {n_out:>8,} {tot:>12,} "
          f"{tot/ex2_params:>7.1%}")
print(f"  {'TOTAL':<18} {'':>12} {'':>8} {ex2_params:>12,}")
assert ex2_params == 235146

# %% [markdown]
# ## Exercise 3 — three layers collapse to one
#
# Apply the section 2.4 identity twice:
#
# $$\big((xW_1 + b_1)W_2 + b_2\big)W_3 + b_3
#  = x\underbrace{(W_1W_2W_3)}_{W_{\text{eq}}} + \underbrace{b_1W_2W_3 + b_2W_3 + b_3}_{b_{\text{eq}}}$$
#
# Each bias is pushed forward through every *later* weight matrix — $b_1$
# passes through both $W_2$ and $W_3$, $b_2$ through only $W_3$, and $b_3$
# through none.
#
# The parameter count collapses from **57 to 8**. Those 49 extra parameters
# were not adding capability; they were adding redundant ways to describe the
# same $3\times2$ affine map. Without a nonlinearity, depth is pure
# over-parameterisation.

# %%
def collapse_linear(W1_, b1_, W2_, b2_, W3_, b3_):
    W_eq = W1_ @ W2_ @ W3_
    b_eq = b1_ @ W2_ @ W3_ + b2_ @ W3_ + b3_
    return W_eq, b_eq


rng = np.random.default_rng(7)
Ws = [rng.normal(size=(3, 4)), rng.normal(size=(4, 5)), rng.normal(size=(5, 2))]
bs = [rng.normal(size=4), rng.normal(size=5), rng.normal(size=2)]

W_eq, b_eq = collapse_linear(Ws[0], bs[0], Ws[1], bs[1], Ws[2], bs[2])
x = rng.normal(size=(6, 3))
deep = ((x @ Ws[0] + bs[0]) @ Ws[1] + bs[1]) @ Ws[2] + bs[2]
flat = x @ W_eq + b_eq

print(f"  three layers : {np.round(deep[0], 8)}")
print(f"  one layer    : {np.round(flat[0], 8)}")
print(f"  max abs diff : {np.abs(deep - flat).max():.2e}")
assert np.allclose(deep, flat)

deep_params = sum(W.size for W in Ws) + sum(b.size for b in bs)
print(f"\n  parameters: {deep_params} -> {W_eq.size + b_eq.size}   "
      f"({deep_params - W_eq.size - b_eq.size} were redundant)")

# %% [markdown]
# ## Exercise 4 — a general forward pass
#
# The whole point of the shape convention is that depth needs no special
# handling: a loop over layers is the entire implementation, and the same code
# runs one sample or a whole batch.
#
# The `"linear"` option exists because it is a real choice, not an omission —
# a regression output layer uses it deliberately. The danger flagged in
# section 9.1 is getting it *by accident*.

# %%
ACTIVATIONS = {
    "tanh": np.tanh,
    "sigmoid": lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
    "linear": lambda z: z,
    "relu": lambda z: np.maximum(0, z),       # a free preview of Lesson 12
}


def forward_deep(X, layers):
    A = np.atleast_2d(X)
    for W, b, name in layers:
        if name not in ACTIVATIONS:
            raise ValueError(f"unknown activation {name!r}; "
                             f"expected one of {sorted(ACTIVATIONS)}")
        A = ACTIVATIONS[name](A @ W + b)
    return A


W1n, b1n = np.array(W1), np.array(B1)
W2n, b2n = np.array(W2).reshape(2, 1), np.array([B2])

out = forward_deep(np.array([[1.0, 2.0]]), [(W1n, b1n, "tanh"), (W2n, b2n, "sigmoid")])
print(f"  reproduces section 3: yhat = {float(out[0, 0]):.10f}   (hand: 0.6367003486)")
assert abs(float(out[0, 0]) - 0.6367003486) < 1e-9

deep5 = [(np.eye(2), np.zeros(2), "tanh")] * 4 + [(W2n, b2n, "sigmoid")]
print(f"  5 layers deep       : yhat = {float(forward_deep(np.array([[1.0, 2.0]]), deep5)[0,0]):.10f}")

batch = np.array([[1., 2.], [0., 1.], [-1., -1.]])
print(f"  batch of 3          : {np.round(forward_deep(batch, [(W1n, b1n, 'tanh'), (W2n, b2n, 'sigmoid')]).ravel(), 6)}")

try:
    forward_deep(batch, [(W1n, b1n, "sigmid")])
except ValueError as e:
    print(f"\n  and a typo is caught loudly: {e}")

# %% [markdown]
# ## Exercise 5 — the permutation symmetry
#
# Swapping hidden units 1 and 2 requires **three** coordinated changes, and
# missing any one of them changes the function:
#
# 1. **Columns of $W_1$** — the incoming weights travel with the unit
# 2. **Entries of $\mathbf{b}_1$** — so does the bias
# 3. **Entries of $W_2$** — and the outgoing weight must follow it too
#
# Forget (3) and unit 1's activation gets multiplied by unit 2's outgoing
# weight, which is simply a different network. Forget (2) and each unit keeps
# the wrong offset.
#
# Get all three right and the output is identical to the last bit — not
# approximately, *exactly*, because it is the same sum with its terms written
# in a different order.
#
# This is the symmetry that proves non-convexity (section 2.5), and it has a
# practical corollary you will meet in Lesson 14: **you must not initialise
# all hidden units identically.** Identical units have identical gradients, so
# they stay identical forever and your width-100 layer behaves as width 1.
# Random initialization exists to break this symmetry.

# %%
def swap_hidden_units(W1_, b1_, W2_):
    W1_sw = [[row[1], row[0]] for row in W1_]     # columns swapped
    b1_sw = [b1_[1], b1_[0]]
    W2_sw = [W2_[1], W2_[0]]
    return W1_sw, b1_sw, W2_sw


sw = swap_hidden_units([r[:] for r in W1], list(B1), list(W2))
_, h_o, _, y_o = forward_scratch([1.0, 2.0], W1, B1, W2, B2)
_, h_s, _, y_s = forward_scratch([1.0, 2.0], sw[0], sw[1], sw[2], B2)

print(f"  original  W1 = {W1}, b1 = {B1}, W2 = {W2}")
print(f"  swapped   W1 = {sw[0]}, b1 = {sw[1]}, W2 = {sw[2]}")
print(f"\n  original  h = {[round(v, 7) for v in h_o]}  ->  yhat = {y_o:.12f}")
print(f"  swapped   h = {[round(v, 7) for v in h_s]}  ->  yhat = {y_s:.12f}")
print(f"  difference: {abs(y_o - y_s):.2e}")
assert abs(y_o - y_s) < 1e-15

# what happens if you forget to move the outgoing weights
_, _, _, y_bad = forward_scratch([1.0, 2.0], sw[0], sw[1], list(W2), B2)
print(f"\n  forgetting to swap W2: yhat = {y_bad:.12f}   <- a different network")
assert abs(y_bad - y_o) > 1e-6

print("""
  For a width-n hidden layer there are n! such rearrangements, all computing
  the same function. Hence the initialization rule: never start two units in
  the same place, or they will never separate.""")

# %% [markdown]
# ## Exercise 6 — width buys reliability, not capability
#
# | hidden units | success rate | why |
# |---|---|---|
# | 1 | **0%** | one unit is one boundary — Lesson 09's proof applies |
# | 2 | **50%** | the minimum that *can* work, and often does not |
# | 4 | **90%** | no more expressive, much easier to find |
# | 8 | **100%** | no more expressive, easier still |
#
# The $H = 1$ row is a **proof**, not bad luck. A single hidden unit gives a
# single linear boundary, then the output layer applies a monotonic squash to
# it — which cannot change which side of the boundary anything is on. That is
# exactly the perceptron of Lesson 09, and XOR is exactly what it cannot do.
#
# Every other row is the interesting one. **Two units are enough** — we built
# the solution by hand in Lesson 09 — so units 3 through 8 add *no expressive
# power at all*. What they add is **starting directions**. Each extra unit is
# another random draw, another chance that some subset of units lands in a
# configuration from which descent reaches the solution. Failure needs *every*
# unit to be badly placed, and that becomes unlikely fast.
#
# This is a large part of why real networks are far wider than any
# expressiveness argument demands. Over-parameterisation is not mainly about
# capacity; it is about making the optimization problem easy. Lesson 14 makes
# the same point from the initialization side, and Lesson 16 asks what it
# costs in overfitting.

# %%
Xxor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
yxor = np.array([0., 1., 1., 0.])


def train_xor(seed, H, steps=4000, lr=1.0):
    r = np.random.default_rng(seed)
    Wa, ba = r.normal(0, 1, (2, H)), np.zeros(H)
    Wb, bb = r.normal(0, 1, (H, 1)), np.zeros(1)
    for _ in range(steps):
        Hh = np.tanh(Xxor @ Wa + ba)
        p = 1 / (1 + np.exp(-(Hh @ Wb + bb)))
        d2 = (p - yxor[:, None]) / 4
        d1 = (d2 @ Wb.T) * (1 - Hh ** 2)
        Wb -= lr * (Hh.T @ d2); bb -= lr * d2.sum(0)
        Wa -= lr * (Xxor.T @ d1); ba -= lr * d1.sum(0)
    Hh = np.tanh(Xxor @ Wa + ba)
    p = 1 / (1 + np.exp(-(Hh @ Wb + bb)))
    return float(np.mean((p.ravel() >= .5) == yxor))


def xor_success_rate_by_width(widths=(1, 2, 4, 8), steps=4000, n_seeds=10):
    return {H: sum(1 for s in range(n_seeds) if train_xor(s, H, steps) == 1.0) / n_seeds
            for H in widths}


rates = xor_success_rate_by_width()
print(f"  {'hidden units':>13} {'parameters':>12} {'success rate':>14}")
for H, r in rates.items():
    print(f"  {H:>13} {2*H + H + H + 1:>12} {r:>13.0%}")
assert rates[1] == 0.0 and rates[2] > 0.0 and rates[8] >= rates[2]

print("\n  per-seed detail for H = 2 (the marginal case):")
print("   ", {s: train_xor(s, 2) for s in range(10)})
print("""
  Same code, same data, same hyperparameters. Half of these runs return a
  model that is 50% accurate on a problem with a known exact solution, and
  none of them raises anything.

  Two habits follow: report results over several seeds, and treat "the model
  cannot learn this" as a claim needing more than one run behind it.""")

# %% [markdown]
# ---
#
# **Back to** `notebooks/10-mlp-forward-pass.ipynb` &nbsp;|&nbsp;
# **Next:** Lesson 11 — Backpropagation, Derived in Full
