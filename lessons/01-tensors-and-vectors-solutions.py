# %% [markdown]
# # Lesson 01 - Solutions
#
# **Tensors, Vectors and Shapes** &nbsp;|&nbsp; Deep Learning From Scratch
#
# Try each exercise on paper first. Reading a solution you have not attempted
# teaches you that the answer looks reasonable, which is not the same as being
# able to produce it.

# %%
import numpy as np

# %% [markdown]
# ## Exercise 1 — the shape of an image batch
#
# `(64, 96, 128, 3)`
#
# The convention is `(batch, height, width, channels)`, and the trap is that
# **height comes before width** even though we say "128 by 96" in conversation.
# It follows from the matrix convention: rows first, and rows run down the
# image. RGB gives 3 channels.

# %%
ex1_shape = (64, 96, 128, 3)
assert ex1_shape == (64, 96, 128, 3)
print(f"total numbers: {64 * 96 * 128 * 3:,}  -- for just 64 small photos")

# %% [markdown]
# ## Exercise 2 — reduction shapes
#
# For `M` of shape `(5, 4)`:
#
# | expression | shape | why |
# |---|---|---|
# | `M.T` | `(4, 5)` | the two axes swap |
# | `M.sum(axis=0)` | `(4,)` | the row index is summed away, 4 columns remain |
# | `M.sum(axis=1)` | `(5,)` | the column index is summed away, 5 rows remain |
# | `M.sum(axis=0, keepdims=True)` | `(1, 4)` | same numbers, axis kept at size 1 |
#
# That last one is the important one. `keepdims=True` leaves the axis in place
# as size 1 so it will *broadcast back* against the original — which is exactly
# what you need to subtract a per-feature mean.

# %%
ex2 = {
    "M.T": (4, 5),
    "M.sum(axis=0)": (4,),
    "M.sum(axis=1)": (5,),
    "M.sum(axis=0, keepdims=True)": (1, 4),
}
M = np.arange(20).reshape(5, 4)
assert ex2 == {"M.T": M.T.shape, "M.sum(axis=0)": M.sum(axis=0).shape,
               "M.sum(axis=1)": M.sum(axis=1).shape,
               "M.sum(axis=0, keepdims=True)": M.sum(axis=0, keepdims=True).shape}
print("Ex 2 verified against NumPy.")

# %% [markdown]
# ## Exercise 3 — cosine similarity in pure Python
#
# $$\cos(u,v) = \frac{u \cdot v}{\lVert u \rVert \lVert v \rVert}
#             = \frac{\sum_i u_i v_i}{\sqrt{\sum_i u_i^2}\ \sqrt{\sum_i v_i^2}}$$
#
# The neat trick is that the norm is just the dot product of a vector with
# itself, square-rooted — so `dot` is the only primitive you need.

# %%
def dot(u, v):
    if len(u) != len(v):
        raise ValueError(f"dot needs equal lengths, got {len(u)} and {len(v)}")
    return sum(u[i] * v[i] for i in range(len(u)))


def cosine_similarity(u, v):
    norm_u = dot(u, u) ** 0.5
    norm_v = dot(v, v) ** 0.5
    if norm_u == 0 or norm_v == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return dot(u, v) / (norm_u * norm_v)


assert cosine_similarity([1, 0], [1, 0]) == 1.0
assert cosine_similarity([1, 0], [0, 1]) == 0.0
assert cosine_similarity([1, 0], [-1, 0]) == -1.0
assert abs(cosine_similarity([1, 2, 3], [4, 5, 6]) - 0.9746318) < 1e-6

# The point of the measure: it ignores magnitude, keeps direction.
print(f"[1,2,3] vs [4,5,6]      : {cosine_similarity([1,2,3],[4,5,6]):.4f}")
print(f"[1,2,3] vs [10,20,30]   : {cosine_similarity([1,2,3],[10,20,30]):.4f}  (same direction, 10x longer)")
print("\nIn Lesson 29 this is exactly how we ask whether two word embeddings")
print("mean similar things -- direction carries the meaning, length does not.")

# %% [markdown]
# ## Exercise 4 — standardizing each feature
#
# $$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
#   \qquad \mu_j = \frac{1}{m}\sum_i x_{ij}, \quad
#   \sigma_j = \sqrt{\frac{1}{m}\sum_i (x_{ij} - \mu_j)^2}$$
#
# `axis=0` because the statistics are **per feature**, computed across
# samples. `keepdims=True` so the `(1, 2)` result broadcasts back against the
# `(4, 2)` data.
#
# Without this step, a feature measured in the hundreds dominates the gradient
# purely because of its units, and training crawls. We will see exactly that
# happen in Lesson 06.

# %%
X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])


def standardize(X):
    mu = X.mean(axis=0, keepdims=True)     # (1, 2) -- one mean per feature
    sigma = X.std(axis=0, keepdims=True)   # (1, 2) -- one std per feature
    return (X - mu) / sigma                # (4,2) - (1,2) broadcasts down the rows


Z = standardize(X)
assert np.allclose(Z.mean(axis=0), [0, 0], atol=1e-9)
assert np.allclose(Z.std(axis=0), [1, 1], atol=1e-9)
assert np.allclose(Z[:, 0], Z[:, 1])

print("before -- wildly different scales:\n", X)
print("\nafter -- identical, comparable:\n", Z)
print("\nBoth columns held the same pattern; only the units differed.")
print("Standardizing made that visible.")

# %% [markdown]
# ## Exercise 5 — reading a broadcast without running it
#
# Right-align the shapes, pad the shorter with 1s on the **left**, then each
# axis pair must be equal or contain a 1.
#
# ```
# (3,1) & (1,4)    3 vs 1 -> 3      1 vs 4 -> 4        => (3, 4)
#                  both stretch; this is an outer operation
#
# (2,3) & (3,)     pad -> (1,3)
#                  2 vs 1 -> 2      3 vs 3 -> 3        => (2, 3)
#                  a bias added to every row
#
# (2,3) & (2,)     pad -> (1,2)
#                  2 vs 1 -> 2      3 vs 2 -> ERROR    => error
#                  the classic mistake: you meant (2,1)
#
# (5,1,4) & (3,4)  pad -> (1,3,4)
#                  5 vs 1 -> 5      1 vs 3 -> 3   4 vs 4 -> 4   => (5, 3, 4)
# ```
#
# The third case is the one to internalise. `(2,)` and `(2,1)` hold the same
# two numbers, but only `(2,1)` means "one value per row".

# %%
ex5 = {
    ((3, 1), (1, 4)): (3, 4),
    ((2, 3), (3,)): (2, 3),
    ((2, 3), (2,)): "error",
    ((5, 1, 4), (3, 4)): (5, 3, 4),
}

for (sa, sb), expected in ex5.items():
    try:
        got = (np.zeros(sa) + np.zeros(sb)).shape
    except ValueError:
        got = "error"
    assert got == expected, f"{sa} + {sb}: predicted {expected}, NumPy says {got}"
    print(f"  {str(sa):<12} + {str(sb):<8} -> {got}")
print("\nEvery prediction confirmed by NumPy.")

# %% [markdown]
# ## Exercise 6 — rank-3 reduction in pure Python
#
# The rule generalises without any new idea: **the index you sum over is the
# one that disappears**, and the surviving indices, in their original order,
# become the output axes.
#
# For a tensor `T[i][j][k]` of shape `(d0, d1, d2)`:
#
# - `axis=0` → output`[j][k] = sum_i T[i][j][k]`, shape `(d1, d2)`
# - `axis=1` → output`[i][k] = sum_j T[i][j][k]`, shape `(d0, d2)`
# - `axis=2` → output`[i][j] = sum_k T[i][j][k]`, shape `(d0, d1)`

# %%
def shape3(T):
    return (len(T), len(T[0]), len(T[0][0]))


def reduce_sum3(T, axis):
    d0, d1, d2 = shape3(T)
    if axis == 0:
        return [[sum(T[i][j][k] for i in range(d0)) for k in range(d2)] for j in range(d1)]
    if axis == 1:
        return [[sum(T[i][j][k] for j in range(d1)) for k in range(d2)] for i in range(d0)]
    if axis == 2:
        return [[sum(T[i][j][k] for k in range(d2)) for j in range(d1)] for i in range(d0)]
    raise ValueError(f"axis {axis} out of range for a rank-3 tensor")


np.random.seed(0)
T = np.random.randint(0, 9, (2, 3, 4))
T_list = T.tolist()

for ax in (0, 1, 2):
    got = np.array(reduce_sum3(T_list, ax))
    assert np.array_equal(got, T.sum(axis=ax)), f"axis {ax} wrong"
    print(f"  axis={ax}:  (2, 3, 4) -> {got.shape}   matches NumPy")

print("\nThat is genuinely the core of a tensor library. Real frameworks add")
print("strides, dtypes, GPU kernels and autodiff on top -- but the indexing")
print("logic you just wrote is the same logic running underneath.")

# %% [markdown]
# ---
#
# **Back to** `notebooks/01-tensors-and-vectors.ipynb` &nbsp;|&nbsp;
# **Next:** Lesson 02 — Matrix Multiplication by Hand
