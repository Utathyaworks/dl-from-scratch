# %% [markdown]
# # Lesson 09 - Solutions
#
# **The Perceptron and its Limits** &nbsp;|&nbsp; Deep Learning From Scratch
#
# Exercise 6 is the one with a moral. It solves XOR with no hidden layer at
# all — and shows you exactly what a hidden layer is doing for free.

# %%
import itertools

import numpy as np

INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
XOR = [0.0, 1.0, 1.0, 0.0]
AND = [0.0, 0.0, 0.0, 1.0]


def step(z):
    return 1.0 if z > 0 else 0.0


def perceptron_train(inputs, targets, lr=1.0, epochs=200):
    w = [0.0, 0.0]
    b = 0.0
    history = []
    for ep in range(1, epochs + 1):
        errors = 0
        for x, y in zip(inputs, targets):
            d = y - step(w[0] * x[0] + w[1] * x[1] + b)
            if d != 0.0:
                w = [w[0] + lr * d * x[0], w[1] + lr * d * x[1]]
                b += lr * d
                errors += 1
        history.append((ep, list(w), b, errors))
        if errors == 0:
            return w, b, ep, history
    return w, b, None, history


# %% [markdown]
# ## Exercise 1 — epoch 3 of AND
#
# **$\mathbf{w} = [2, 1]$, $b = -1$, 3 errors.**
#
# Starting from $\mathbf{w} = [2,1]$, $b = 0$:
#
# | sample | $z$ | $\hat{y}$ | $y$ | $y-\hat{y}$ | after |
# |---|---|---|---|---|---|
# | $(0,0)$ | $0$ | 0 | 0 | 0 | unchanged |
# | $(0,1)$ | $0+1+0 = 1$ | 1 | 0 | $-1$ | $[2,0]$, $b=-1$ |
# | $(1,0)$ | $2+0-1 = 1$ | 1 | 0 | $-1$ | $[1,0]$, $b=-2$ |
# | $(1,1)$ | $1+0-2 = -1$ | 0 | **1** | $+1$ | $[2,1]$, $b=-1$ |
#
# Three errors again — the same count as epoch 2, and the weights have
# returned to $[2,1]$ with the bias one lower. It looks like it is going
# nowhere, and yet it finishes three epochs later.
#
# That is the perceptron's character: **the bias ratchets downward** while the
# weights oscillate, and the theorem guarantees termination without promising
# it will look like progress.

# %%
ex1_w, ex1_b, ex1_errors = [2.0, 1.0], -1.0, 3
_, _, _, hist = perceptron_train(INPUTS, AND)
print(f"  {'epoch':>6} {'w':>12} {'b':>6} {'errors':>8}")
for ep, w, b, errs in hist:
    mark = "  <- exercise 1" if ep == 3 else ("  <- solved" if errs == 0 else "")
    print(f"  {ep:>6} {str([f'{v:g}' for v in w]):>12} {b:>6g} {errs:>8}{mark}")
assert hist[2][1] == ex1_w and hist[2][2] == ex1_b and hist[2][3] == ex1_errors
print("\n  Errors went 1, 3, 3, 2, 1, 0. Not monotone -- but finite, as promised.")

# %% [markdown]
# ## Exercise 2 — how narrow the wall really is
#
# **Exactly 2 of the 16.** XOR $(0,1,1,0)$ and its negation XNOR $(1,0,0,1)$.
#
# There are $2^4 = 16$ ways to label the four corners of the square. Fourteen
# of them can be cut with a straight line:
#
# - 2 constants (all 0, all 1) — trivially separable
# - 8 functions that isolate a single corner — a line can always cut one corner off
# - 4 that depend on one input only ($x_1$, $\neg x_1$, $x_2$, $\neg x_2$) — an axis-aligned line
#
# That leaves the two where the positives sit on *opposite corners*. This is
# worth sitting with: the perceptron's limitation is **extremely narrow** —
# 87.5% of two-input boolean functions are fine — and yet it was enough to
# stall the field for over a decade, because the gap does not shrink with more
# inputs. It explodes. For $n$ inputs there are $2^{2^n}$ boolean functions and
# only a vanishing fraction are linearly separable.

# %%
def separable(bits):
    return perceptron_train(INPUTS, [float(v) for v in bits])[2] is not None


not_sep = [b for b in itertools.product([0, 1], repeat=4) if not separable(b)]
ex2_count = len(not_sep)

names = {(0, 1, 1, 0): "XOR", (1, 0, 0, 1): "XNOR"}
print(f"  separable    : {16 - len(not_sep)} of 16")
print(f"  NOT separable: {len(not_sep)} of 16")
for b in not_sep:
    print(f"      {b}  = {names[b]}")
assert ex2_count == 2

print("\n  how the wall scales with the number of inputs:")
print(f"  {'inputs':>7} {'boolean functions':>20} {'linearly separable':>20} {'fraction':>12}")
# counts of threshold functions of n variables (known sequence)
for n, sep in [(1, 4), (2, 14), (3, 104), (4, 1882), (5, 94572)]:
    total = 2 ** (2 ** n)
    print(f"  {n:>7} {total:>20,} {sep:>20,} {sep/total:>11.2%}")
print("""
  At 2 inputs you lose 12.5% of functions. At 5 inputs you lose 99.99998%.
  Minsky and Papert's 1969 book made exactly this point, and funding for
  neural networks collapsed for fifteen years.""")

# %% [markdown]
# ## Exercise 3 — NAND
#
# **$\mathbf{w} = [-1, -1]$, $b = 1.5$.**
#
# NAND is "not both", so it should fire *unless* both inputs are 1. Negative
# weights and a positive bias give exactly that: start above threshold, and let
# each active input push you down.
#
# | input | $z = -x_1 - x_2 + 1.5$ | output |
# |---|---|---|
# | $(0,0)$ | $1.5$ | 1 |
# | $(0,1)$ | $0.5$ | 1 |
# | $(1,0)$ | $0.5$ | 1 |
# | $(1,1)$ | $-0.5$ | 0 |
#
# NAND matters because it is **functionally complete**: every boolean function
# can be built from NAND alone. So a network of perceptrons can compute *any*
# boolean function — the limitation was never about what perceptrons can
# express, only about what a **single** one can. Depth was always the answer.

# %%
ex3_w, ex3_b = [-1.0, -1.0], 1.5
for x, t in zip(INPUTS, [1.0, 1.0, 1.0, 0.0]):
    z = ex3_w[0] * x[0] + ex3_w[1] * x[1] + ex3_b
    print(f"  NAND({x[0]:.0f},{x[1]:.0f})  z = {z:>5.1f}  ->  {step(z):.0f}   expected {t:.0f}")
    assert step(z) == t
print("\n  Also note: the perceptron can LEARN this by itself --",
      perceptron_train(INPUTS, [1.0, 1.0, 1.0, 0.0])[2], "epochs.")

# %% [markdown]
# ## Exercise 4 — XOR from four NAND gates
#
# $$\text{XOR}(a,b) = \text{NAND}\big(\text{NAND}(a, c),\ \text{NAND}(b, c)\big),
#   \qquad c = \text{NAND}(a,b)$$
#
# Trace $(1,1)$: $c = \text{NAND}(1,1) = 0$, then $\text{NAND}(1,0) = 1$ and
# $\text{NAND}(1,0) = 1$, then $\text{NAND}(1,1) = 0$. Correct.
#
# Trace $(0,1)$: $c = 1$, then $\text{NAND}(0,1) = 1$ and $\text{NAND}(1,1) = 0$,
# then $\text{NAND}(1,0) = 1$. Correct.
#
# The circuit is **two layers deep** (after computing $c$), and that depth is
# exactly what buys the expressiveness. Four perceptrons side by side in one
# layer would still be one linear boundary each, combined linearly — and a
# linear combination of linear functions is linear. **Composition, not
# quantity, is what escapes the wall.** That single sentence is why "deep"
# learning is called deep rather than wide.

# %%
def nand(a, b):
    return step(-1.0 * a + -1.0 * b + 1.5)


def xor_from_nand(a, b):
    c = nand(a, b)
    return nand(nand(a, c), nand(b, c))


print(f"  {'input':>8} {'c=NAND(a,b)':>13} {'output':>8} {'target':>8}")
for x, t in zip(INPUTS, XOR):
    c = nand(x[0], x[1])
    got = xor_from_nand(x[0], x[1])
    print(f"  ({x[0]:.0f},{x[1]:.0f})    {c:>13.0f} {got:>8.0f} {t:>8.0f}")
    assert got == t
print("\n  Four identical units. The only thing that changed is how they are wired.")

# %% [markdown]
# ## Exercise 5 — counting updates, not epochs
#
# **AND converges after 10 weight updates.**
#
# The convergence theorem bounds *updates*, not epochs, and the bound is
#
# $$\#\text{updates} \le \left(\frac{R}{\gamma}\right)^2$$
#
# where $R = \max_i\lVert \mathbf{x}_i\rVert$ and $\gamma$ is the margin of the
# best separating hyperplane. Both are properties of the **data geometry
# alone**.
#
# Two things follow that are worth internalising:
#
# - **The learning rate does not appear.** With $\mathbf{w}$ starting at zero,
#   scaling $\eta$ scales every weight identically, and the sign of
#   $\mathbf{w}\cdot\mathbf{x} + b$ is unchanged. $\eta$ is genuinely
#   irrelevant here — the only algorithm in this course where that is true.
# - **A small margin is expensive.** Halving $\gamma$ quadruples the work. This
#   is the same quantity support vector machines maximise deliberately.

# %%
def count_updates(inputs, targets, lr=1.0, epochs=200):
    w = [0.0, 0.0]
    b = 0.0
    updates = 0
    for _ in range(epochs):
        errors = 0
        for x, y in zip(inputs, targets):
            d = y - step(w[0] * x[0] + w[1] * x[1] + b)
            if d != 0.0:
                w = [w[0] + lr * d * x[0], w[1] + lr * d * x[1]]
                b += lr * d
                errors += 1
                updates += 1
        if errors == 0:
            return updates
    return None


print(f"  AND : {count_updates(INPUTS, AND)} updates")
print(f"  XOR : {count_updates(INPUTS, XOR, epochs=50)}   (never converges)")
assert count_updates(INPUTS, AND) is not None
assert count_updates(INPUTS, XOR, epochs=50) is None

print("\n  the learning rate genuinely does not matter:")
for lr in (0.01, 0.1, 1.0, 10.0, 1000.0):
    w, b, ep, _ = perceptron_train(INPUTS, AND, lr=lr)
    print(f"    eta = {lr:>7}: {count_updates(INPUTS, AND, lr=lr)} updates, "
          f"epoch {ep}, w = {[f'{v:g}' for v in w]}, b = {b:g}")
print("""
  Identical update counts at every learning rate, with only the SCALE of the
  final weights changing. Contrast Lesson 06, where eta had a hard stability
  cliff. The difference is that the step function cares only about the sign
  of z, so uniformly rescaling the weights changes nothing it can observe.""")

# %% [markdown]
# ## Exercise 6 — XOR without a hidden layer
#
# **Converges in 8 epochs with $\mathbf{w} = [1, 1, -3]$, $b = 0$.**
#
# Add $x_1x_2$ as a third input feature and the perceptron solves XOR
# immediately:
#
# $$z = x_1 + x_2 - 3x_1x_2$$
#
# | input | $x_1 + x_2$ | $-3x_1x_2$ | $z$ | output |
# |---|---|---|---|---|
# | $(0,0)$ | 0 | 0 | $0$ | 0 |
# | $(0,1)$ | 1 | 0 | $1$ | 1 |
# | $(1,0)$ | 1 | 0 | $1$ | 1 |
# | $(1,1)$ | 2 | $-3$ | $-1$ | 0 |
#
# The product term does nothing until *both* inputs are on, then it slams the
# output down. It is the "but not both" clause, made into a coordinate.
#
# ### The moral
#
# Nothing about the *learning rule* was ever the problem. The problem was the
# **representation**. Given the right features, a linear model is enough; the
# impossibility proof in section 2.5 applies to the raw coordinates
# $(x_1, x_2)$ and to nothing else.
#
# So there are two ways out of the XOR wall:
#
# 1. **Design the features yourself** — this exercise. Works, and for decades
#    this was the job: hand-crafted edge detectors, SIFT, MFCCs, n-grams.
# 2. **Let the network learn the features** — the hidden layer. Lesson 10.
#
# Option 1 requires knowing in advance which products matter. With 2 inputs
# there is one product to try; with 1000 inputs there are ~500,000 pairwise
# products, and the useful ones are not pairwise. Option 2 scales, and that is
# the entire argument for deep learning over feature engineering.

# %%
def xor_with_product_feature(epochs=200, lr=1.0):
    feats = [(x1, x2, x1 * x2) for x1, x2 in INPUTS]
    w = [0.0, 0.0, 0.0]
    b = 0.0
    for ep in range(1, epochs + 1):
        errors = 0
        for f, y in zip(feats, XOR):
            d = y - step(sum(wi * fi for wi, fi in zip(w, f)) + b)
            if d != 0.0:
                w = [wi + lr * d * fi for wi, fi in zip(w, f)]
                b += lr * d
                errors += 1
        if errors == 0:
            return w, b, ep
    return w, b, None


w, b, ep = xor_with_product_feature()
print(f"  converged at epoch {ep}:  w = {[f'{v:g}' for v in w]}, b = {b:g}")
print(f"\n  {'input':>8} {'x1+x2':>8} {'-3*x1x2':>9} {'z':>6} {'out':>5} {'target':>7}")
for (x1, x2), t in zip(INPUTS, XOR):
    f = (x1, x2, x1 * x2)
    z = sum(wi * fi for wi, fi in zip(w, f)) + b
    print(f"  ({x1:.0f},{x2:.0f})    {x1+x2:>8.0f} {w[2]*x1*x2:>9.0f} {z:>6.0f} "
          f"{step(z):>5.0f} {t:>7.0f}")
    assert step(z) == t
assert ep is not None

print("""
  The same four points that were provably inseparable in 2-D are trivially
  separable in 3-D. Nothing about them changed -- we just stopped insisting
  on looking at them in the original coordinates.

  A hidden layer does this without being told which coordinate to add.""")

# %% [markdown]
# ---
#
# **Back to** `notebooks/09-the-perceptron.ipynb` &nbsp;|&nbsp;
# **Next:** Lesson 10 — The MLP Forward Pass
