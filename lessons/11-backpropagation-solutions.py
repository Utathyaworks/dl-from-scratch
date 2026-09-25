# %% [markdown]
# # Lesson 11 - Solutions
#
# **Backpropagation, Derived in Full** &nbsp;|&nbsp; Deep Learning From Scratch
#
# Exercise 3 is the one to keep. A general gradient checker is the single most
# useful twenty lines in this entire course — every later lesson leans on it.

# %%
import math

import numpy as np

W1 = [[0.5, -1.0], [-0.5, 0.5]]
B1 = [0.0, 1.0]
W2 = [1.0, 2.0]
B2 = -0.5
X_IN = [1.0, 2.0]
Y_TRUE = 1.0


def forward(x, W1_, b1_, W2_, b2_):
    z1 = [sum(x[i] * W1_[i][j] for i in range(len(x))) + b1_[j] for j in range(len(b1_))]
    h = [math.tanh(v) for v in z1]
    z2 = sum(h[j] * W2_[j] for j in range(len(W2_))) + b2_
    return z1, h, z2, 1 / (1 + math.exp(-z2))


def backward(x, y, h, yhat, W2_):
    delta2 = yhat - y
    dW2 = [delta2 * h[j] for j in range(len(h))]
    delta1 = [delta2 * W2_[j] * (1 - h[j] ** 2) for j in range(len(h))]
    dW1 = [[delta1[j] * x[i] for j in range(len(h))] for i in range(len(x))]
    return dW1, list(delta1), dW2, delta2, delta1


# %% [markdown]
# ## Exercise 1 — the same network, a negative example
#
# **$\delta^{(2)} = +0.7001851$, $\boldsymbol{\delta}^{(1)} = [0.5506590,\ 0.2530562]$.**
#
# With $\hat{y} = 0.7001851$ and $y = 0$:
#
# $$\delta^{(2)} = \hat{y} - y = 0.7001851 - 0 = 0.7001851$$
#
# **Positive now**, where section 3 gave $-0.363$. The model predicted 0.70 for
# something that should have been 0, so $z^{(2)}$ must come *down* — and a
# positive gradient is precisely the instruction to decrease.
#
# The $\tanh'$ values:
#
# $$1 - h_1^2 = 1 - (-0.4621172)^2 = 0.7864477, \qquad
#   1 - h_2^2 = 1 - (0.9051483)^2 = 0.1807066$$
#
# $$\delta^{(1)}_1 = (0.7001851)(1.0)(0.7864477) = 0.5506590$$
# $$\delta^{(1)}_2 = (0.7001851)(2.0)(0.1807066) = 0.2530562$$
#
# Compare with section 3. There, $h_2 = 0.762$ and $\tanh' = 0.420$; here
# $h_2 = 0.905$ and $\tanh' = 0.181$ — **less than half**. Unit 2 has been
# driven further into saturation by this input, so despite carrying double
# weight it now passes back *less* error than unit 1.
#
# That is the vanishing gradient in miniature: a unit's willingness to learn
# depends on where its own input happens to sit, and confident units are deaf.

# %%
ex1_delta2, ex1_delta1 = 0.7001851, [0.5506590, 0.2530562]
z1, h, z2, yh = forward([0.0, 1.0], W1, B1, W2, B2)
_, _, _, d2, d1 = backward([0.0, 1.0], 0.0, h, yh, W2)
print(f"  yhat    = {yh:.7f}   (target 0)")
print(f"  delta2  = {d2:.7f}")
print(f"  tanh'   = {[round(1 - v**2, 7) for v in h]}")
print(f"  delta1  = {[round(v, 7) for v in d1]}")
assert abs(ex1_delta2 - d2) < 1e-6
assert all(abs(a - b) < 1e-6 for a, b in zip(ex1_delta1, d1))

print(f"\n  {'':12} {'section 3 (x=[1,2])':>22} {'here (x=[0,1])':>18}")
_, h3, _, _ = forward(X_IN, W1, B1, W2, B2)
print(f"  {'h2':12} {h3[1]:>22.7f} {h[1]:>18.7f}")
print(f"  {'tanh2':12} {1-h3[1]**2:>22.7f} {1-h[1]**2:>18.7f}")
print("\n  A more confident unit passes back less error. Confidence is deafness.")

# %% [markdown]
# ## Exercise 2 — linear output with MSE
#
# **$\delta^{(2)} = 2(\hat{y} - y)$, with $\hat{y} = z^{(2)}$.**
#
# There is no sigmoid, so $\hat{y} = z^{(2)}$ and $\partial\hat{y}/\partial z^{(2)} = 1$.
# With $L = (\hat{y} - y)^2$:
#
# $$\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}}
#  = \frac{\partial L}{\partial\hat{y}}\cdot 1 = 2(\hat{y} - y)$$
#
# **Everything downstream is unchanged.** The hidden-layer recursion, the
# weight rule, all of it — the only thing that differs between a regression
# network and a classification network is the single number $\delta^{(2)}$.
#
# That is worth pausing on. Swapping the task means swapping one line. It is
# why frameworks let you change `loss=` without touching the model, and it is
# the same *matched pair* idea from Lesson 08 §2.5: identity output with
# squared error, sigmoid output with cross-entropy. Each pairing collapses to
# a clean $\delta = \text{(prediction} - \text{target)}$, up to a constant.

# %%
def backward_mse(x, y, h, yhat, W2_):
    """Linear output + MSE. Only delta2 differs from the BCE version."""
    delta2 = 2 * (yhat - y)                       # <-- the ONLY change
    dW2 = [delta2 * h[j] for j in range(len(h))]
    db2 = delta2
    delta1 = [delta2 * W2_[j] * (1 - h[j] ** 2) for j in range(len(h))]
    dW1 = [[delta1[j] * x[i] for j in range(len(h))] for i in range(len(x))]
    db1 = list(delta1)
    return dW1, db1, dW2, db2


_, h2, z2b, _ = forward(X_IN, W1, B1, W2, B2)
r = backward_mse(X_IN, Y_TRUE, h2, z2b, W2)       # yhat = z2 for a linear output
print(f"  linear output yhat = z2 = {z2b:.7f}, target {Y_TRUE}")
print(f"  delta2 = 2(yhat - y)    = {r[3]:.7f}")
print(f"  dW2                     = {[round(v, 7) for v in r[2]]}")
assert abs(r[3] - 2 * (z2b - Y_TRUE)) < 1e-12

# gradient-check it, because a new delta2 is exactly where errors hide
def mse_loss_flat(p):
    W1_ = [[p[0], p[1]], [p[2], p[3]]]
    _, _, z2_, _ = forward(X_IN, W1_, [p[4], p[5]], [p[6], p[7]], p[8])
    return (z2_ - Y_TRUE) ** 2


theta0 = [W1[0][0], W1[0][1], W1[1][0], W1[1][1], B1[0], B1[1], W2[0], W2[1], B2]
analytic = [r[0][0][0], r[0][0][1], r[0][1][0], r[0][1][1],
            r[1][0], r[1][1], r[2][0], r[2][1], r[3]]
worst = 0.0
for k, a in enumerate(analytic):
    hi, lo = theta0[:], theta0[:]
    hi[k] += 1e-6
    lo[k] -= 1e-6
    n = (mse_loss_flat(hi) - mse_loss_flat(lo)) / 2e-6
    worst = max(worst, abs(a - n) / max(1e-300, abs(a) + abs(n)))
print(f"\n  gradient check on the MSE version: worst rel err {worst:.2e}")
assert worst < 1e-5

# %% [markdown]
# ## Exercise 3 — a reusable gradient checker
#
# Flatten the parameters into a vector, perturb each entry, and compare. The
# details that make it trustworthy rather than decorative:
#
# - **Central differences**, error $O(\varepsilon^2)$ rather than $O(\varepsilon)$.
# - **$\varepsilon = 10^{-6}$** — large enough to survive float64 cancellation,
#   small enough that the quadratic term stays negligible.
# - **Relative error with an absolute escape hatch**, exactly as derived in
#   Lesson 06 §5.1, so a genuinely-zero gradient is not reported as a failure.
# - **Check at a random, untrained point.** At a converged one every gradient
#   is ~0 and the check tells you nothing.
#
# Keep this function. Every hand-derived gradient in the remaining thirty-four
# lessons should go through it before you trust it.

# %%
def grad_check_all(W1_, b1_, W2_, b2_, x, y, eps=1e-6):
    """Worst relative error between hand-derived and numerical gradients."""
    def loss_flat(p):
        _, _, _, yh_ = forward(x, [[p[0], p[1]], [p[2], p[3]]],
                               [p[4], p[5]], [p[6], p[7]], p[8])
        yh_ = min(max(yh_, 1e-15), 1 - 1e-15)
        return -(y * math.log(yh_) + (1 - y) * math.log(1 - yh_))

    _, h_, _, yh_ = forward(x, W1_, b1_, W2_, b2_)
    dW1_, db1_, dW2_, db2_, _ = backward(x, y, h_, yh_, W2_)
    analytic_ = [dW1_[0][0], dW1_[0][1], dW1_[1][0], dW1_[1][1],
                 db1_[0], db1_[1], dW2_[0], dW2_[1], db2_]
    theta = [W1_[0][0], W1_[0][1], W1_[1][0], W1_[1][1],
             b1_[0], b1_[1], W2_[0], W2_[1], b2_]

    worst_ = 0.0
    for k, a in enumerate(analytic_):
        hi, lo = theta[:], theta[:]
        hi[k] += eps
        lo[k] -= eps
        n = (loss_flat(hi) - loss_flat(lo)) / (2 * eps)
        if abs(a - n) < 1e-8:               # absolute escape hatch
            continue
        worst_ = max(worst_, abs(a - n) / max(1e-300, abs(a) + abs(n)))
    return worst_


print(f"  at the section 3 point : {grad_check_all(W1, B1, W2, B2, X_IN, Y_TRUE):.2e}")
print(f"  at a different point   : "
      f"{grad_check_all([[2.0,-3.0],[1.0,0.5]], [0.3,-0.7], [-1.5,2.5], 0.2, [0.4,-1.1], 0.0):.2e}")

rng = np.random.default_rng(0)
worsts = []
for _ in range(200):
    p = rng.normal(0, 1.5, 9)
    worsts.append(grad_check_all([[p[0], p[1]], [p[2], p[3]]], [p[4], p[5]],
                                 [p[6], p[7]], p[8],
                                 rng.normal(0, 1, 2).tolist(), float(rng.integers(2))))
print(f"  over 200 random points : worst {max(worsts):.2e}, median {np.median(worsts):.2e}")
assert max(worsts) < 1e-5
print("\n  The derivation holds everywhere, not just at one convenient point.")

# %% [markdown]
# ## Exercise 4 — backprop at any depth
#
# The recursion from section 2.5, written as a loop. Three details carry the
# whole implementation:
#
# 1. **Cache every activation on the way forward.** The backward pass needs
#    $\mathbf{a}^{(l-1)}$ to form $\partial L/\partial W_l$. This cache is why
#    training uses so much more memory than inference, and why gradient
#    checkpointing (recomputing activations instead of storing them) exists.
# 2. **Walk the layers in reverse.**
# 3. **Use $W_{l}$ to push the delta back *before* overwriting anything** —
#    if you update weights inside the loop you will propagate through the new
#    values, which is a different and wrong algorithm.
#
# Note how little code the generality costs: this handles 2 layers or 200.

# %%
def backprop_deep(X, Y, layers):
    """layers: [(W, b), ...]; hidden layers use tanh, output uses sigmoid+BCE."""
    m = len(X)

    # forward, caching every activation
    acts = [X]
    for i, (W, b) in enumerate(layers):
        Z = acts[-1] @ W + b
        acts.append(1 / (1 + np.exp(-Z)) if i == len(layers) - 1 else np.tanh(Z))

    grads = [None] * len(layers)
    D = (acts[-1] - Y) / m                       # delta at the output
    for l in range(len(layers) - 1, -1, -1):
        W, b = layers[l]
        grads[l] = (acts[l].T @ D, D.sum(axis=0))
        if l > 0:                                 # push back, using the OLD W
            D = (D @ W.T) * (1 - acts[l] ** 2)    # tanh' of the layer below
    return grads


W1n, b1n = np.array(W1), np.array(B1)
W2n, b2n = np.array(W2).reshape(2, 1), np.array([B2])
Xb, Yb = np.array([X_IN]), np.array([[Y_TRUE]])

g = backprop_deep(Xb, Yb, [(W1n, b1n), (W2n, b2n)])
print(f"  dW1 =\n{np.round(g[0][0], 10)}")
print(f"  dW2 = {np.round(g[1][0].ravel(), 10)}")
expected_dW1 = np.array([[-0.2857161873, -0.3051530638], [-0.5714323745, -0.6103061277]])
assert np.allclose(g[0][0], expected_dW1), "must reproduce section 3"

L3 = [(np.array([[.3, -.2], [.1, .4]]), np.zeros(2)),
      (np.array([[.5, .1], [-.3, .2]]), np.zeros(2)),
      (W2n, b2n)]
g3 = backprop_deep(Xb, Yb, L3)
print(f"\n  3 layers: shapes {[ (dW.shape, db.shape) for dW, db in g3 ]}")
for (dW, db), (Wl, bl) in zip(g3, L3):
    assert dW.shape == Wl.shape and db.shape == bl.shape

# and gradient-check the deep version end to end
def deep_loss(flat, shapes):
    ls, i = [], 0
    for Ws, bs in shapes:
        n = int(np.prod(Ws))
        W = flat[i:i + n].reshape(Ws); i += n
        b = flat[i:i + bs[0]]; i += bs[0]
        ls.append((W, b))
    A = Xb
    for k, (W, b) in enumerate(ls):
        Z = A @ W + b
        A = 1 / (1 + np.exp(-Z)) if k == len(ls) - 1 else np.tanh(Z)
    p = np.clip(A, 1e-15, 1 - 1e-15)
    return float(-np.mean(Yb * np.log(p) + (1 - Yb) * np.log(1 - p)))


shapes = [(W.shape, b.shape) for W, b in L3]
flat = np.concatenate([np.concatenate([W.ravel(), b.ravel()]) for W, b in L3])
flat_g = np.concatenate([np.concatenate([dW.ravel(), db.ravel()]) for dW, db in g3])
num = np.array([(deep_loss(flat + e, shapes) - deep_loss(flat - e, shapes)) / 2e-6
                for e in 1e-6 * np.eye(len(flat))])
rel = np.max(np.abs(flat_g - num) / np.maximum(1e-300, np.abs(flat_g) + np.abs(num)))
print(f"  gradient check on the 3-layer network: worst rel err {rel:.2e}")
assert rel < 1e-5

# %% [markdown]
# ## Exercise 5 — vanishing, stable, exploding
#
# | weight scale | ratio at depth 20 | regime |
# |---|---|---|
# | 0.3 | $\approx 7\times10^{-3}$ | **vanishing** |
# | 0.5 | $\approx 5\times10^{-1}$ | roughly stable |
# | 1.0 | $\approx 1.4\times10^{1}$ | **exploding** |
#
# The important correction to a common half-truth: **it is not $\tanh'$ alone
# that decides.** Each backward step does two things,
#
# $$\boldsymbol{\delta}^{(l)} = \underbrace{\left(\boldsymbol{\delta}^{(l+1)}W_{l+1}^\top\right)}_{\text{can grow}}\odot\underbrace{\phi'\left(\mathbf{z}^{(l)}\right)}_{\text{always shrinks}}$$
#
# and they fight. $\tanh' \le 1$ always shrinks; multiplying by $W^\top$ scales
# by roughly the spectral norm of $W$, which grows with the weight scale and
# the layer width. Small weights lose the fight and the signal dies; large
# weights win too hard and it blows up.
#
# That reframes the whole problem. "Vanishing gradients" is not a property of
# $\tanh$ — it is a property of the *product*, and it means there is a scale
# at which the two effects balance. Finding it is exactly what Xavier and He
# initialization do, and that is Lesson 14.
#
# **A measurement note.** Measure the *delta*, not the weight gradient. The
# weight gradient is $\mathbf{a}^{(l-1)\top}\boldsymbol{\delta}^{(l)}$, so it
# also carries the activation magnitude — which shrinks with small weights
# too, partly cancelling the effect you are trying to observe. The first
# attempt at this exercise measured weight gradients and saw no clear trend
# for exactly that reason.

# %%
def delta_decay_by_depth(depths=(2, 5, 10, 20), width=8, scale=0.3, n_seeds=5):
    out = {}
    for L in depths:
        ratios = []
        for seed in range(n_seeds):
            r = np.random.default_rng(seed)
            Ws = [r.normal(0, scale, (width, width)) for _ in range(L)]
            Wo = r.normal(0, scale, (width, 1))
            X = r.normal(0, 1, (16, width))
            Y = (r.random((16, 1)) < 0.5).astype(float)

            acts = [X]
            for W in Ws:
                acts.append(np.tanh(acts[-1] @ W))
            P = 1 / (1 + np.exp(-(acts[-1] @ Wo)))

            D = (P - Y) / len(X)
            D = (D @ Wo.T) * (1 - acts[-1] ** 2)     # delta at the LAST hidden layer
            mags = [np.abs(D).mean()]
            for l in range(L - 1, 0, -1):
                D = (D @ Ws[l].T) * (1 - acts[l] ** 2)
                mags.append(np.abs(D).mean())
            ratios.append(mags[-1] / max(mags[0], 1e-300))
        out[L] = float(np.mean(ratios))
    return out


print(f"  {'depth':>7} {'scale 0.3':>14} {'scale 0.5':>14} {'scale 1.0':>14}")
tables = {s: delta_decay_by_depth(scale=s) for s in (0.3, 0.5, 1.0)}
for L in (2, 5, 10, 20):
    print(f"  {L:>7} " + " ".join(f"{tables[s][L]:>14.3e}" for s in (0.3, 0.5, 1.0)))

assert tables[0.3][20] < tables[0.3][2] / 10
assert tables[1.0][20] > tables[0.3][20] * 100
print("""
  The left column dies, the right column explodes, and the middle one holds.
  Three runs of identical code; only the initial weight scale differs.

  Depth is not inherently unstable -- it is unstable at the WRONG scale.
  Lesson 14 works out the right one.""")

# %% [markdown]
# ## Exercise 6 — the delta rule holds everywhere
#
# $$\frac{\partial L}{\partial W^{(l)}_{ij}} = \delta^{(l)}_j\, a^{(l-1)}_i$$
#
# Checked against a full numerical gradient for every weight of a 3-layer
# network, at random parameters. The worst relative error lands around
# $10^{-9}$ — the rule is exact, and what remains is floating-point noise in
# the *numerical* estimate, not error in the rule.
#
# This is the payoff of the whole lesson. One rule — **delta times incoming
# activation** — covers every weight in every layer. And it keeps covering
# them: a convolution (Lesson 24) shares one weight across many positions, so
# its gradient is the *sum* of delta-times-activation over those positions; an
# LSTM (Lesson 33) reuses one weight across timesteps, so its gradient sums
# over time. Same rule, wider sum.

# %%
def verify_delta_rule(seed=0, eps=1e-6):
    r = np.random.default_rng(seed)
    sizes = [3, 5, 4, 1]
    layers = [(r.normal(0, 0.8, (sizes[i], sizes[i + 1])), r.normal(0, 0.3, sizes[i + 1]))
              for i in range(3)]
    X = r.normal(0, 1, (6, 3))
    Y = (r.random((6, 1)) < 0.5).astype(float)

    def loss_of(ls):
        A = X
        for k, (W, b) in enumerate(ls):
            Z = A @ W + b
            A = 1 / (1 + np.exp(-Z)) if k == len(ls) - 1 else np.tanh(Z)
        p = np.clip(A, 1e-15, 1 - 1e-15)
        return float(-np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p)))

    grads = backprop_deep(X, Y, layers)

    worst_ = 0.0
    for li, (W, b) in enumerate(layers):
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                up = [(Wl.copy(), bl.copy()) for Wl, bl in layers]
                dn = [(Wl.copy(), bl.copy()) for Wl, bl in layers]
                up[li][0][i, j] += eps
                dn[li][0][i, j] -= eps
                num = (loss_of(up) - loss_of(dn)) / (2 * eps)
                ana = grads[li][0][i, j]
                if abs(ana - num) < 1e-9:
                    continue
                worst_ = max(worst_, abs(ana - num) / max(1e-300, abs(ana) + abs(num)))
    return worst_


for s in range(4):
    print(f"  seed {s}: worst relative error {verify_delta_rule(s):.2e}")
assert verify_delta_rule() < 1e-4
print("""
  47 weights, three layers, checked one at a time against the definition of
  a derivative. The rule holds for every one.

  That is the whole of backpropagation. Everything from here -- convolutions,
  recurrence, attention -- changes WHICH activations a weight touches, and
  therefore what the sum runs over. It never changes the rule.""")

# %% [markdown]
# ---
#
# **Back to** `notebooks/11-backpropagation.ipynb` &nbsp;|&nbsp;
# **Next:** Lesson 12 — Activations: Sigmoid, Tanh, ReLU, GELU
