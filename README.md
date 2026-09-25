# Deep Learning From Scratch

**45 notebooks that build deep learning from first principles.** Every lesson
derives the maths, works a tiny example *by hand*, visualizes it, then
implements it three times — pure Python, NumPy, TensorFlow — and asserts that
all three agree with the hand calculation. If they disagree, the lesson does
not ship.

Runs on **Kaggle**, Colab, or locally. Nothing to download inside a lesson.

- **Notebooks:** [`notebooks/`](notebooks/) · **Solutions:** [`solutions/`](solutions/)
- **Progress:** [PROGRESS.md](PROGRESS.md)
- **Site & interactive demos:** **https://utathyaworks.github.io/dl-from-scratch**

---

## Install and run it locally

```bash
pip install "dl-from-scratch[notebooks] @ git+https://github.com/Utathyaworks/dl-from-scratch"
```

Then:

```bash
dlscratch list                 # the curriculum, and what is finished
dlscratch open 1               # copy lesson 1 into ./dlscratch-lessons and launch Jupyter
dlscratch site                 # serve the interactive site at localhost:8000
dlscratch info backprop        # look a lesson up by name
dlscratch verify 7             # execute a lesson headless and report any failure
```

`dlscratch open` copies the notebook out of the package into your working directory,
so you can scribble in it freely — your copy is yours, and the pristine one
stays in the package.

### Extras

| Install | Gives you |
|---|---|
| `pip install dl-from-scratch` | the CLI, the notebooks, and the library. NumPy + matplotlib only |
| `...[notebooks]` | JupyterLab, so `dlscratch open` can launch straight into a lesson |
| `...[tf]` | TensorFlow, enabling section 7 of every lesson |
| `...[dev]` | `nbclient` etc. for `dlscratch verify` and for contributing |
| `...[all]` | everything |

TensorFlow is **optional by design** — section 7 of every notebook detects its
absence and skips itself with a message, so nothing breaks without it.

### Using the library directly

The notebooks are deliberately self-contained (the same file runs unchanged on
Kaggle), but the gradient checker is worth having outside one:

```python
from dlscratch import grad_check, numeric_gradient

loss = lambda p: sum((p[0] * x + p[1] - y) ** 2 for x, y in zip(X, Y)) / len(X)
check = grad_check(my_hand_derived_gradient(p), numeric_gradient(loss, p))
print(check)          # rel 6.99e-11  abs 9.32e-10  -> excellent
assert check.passed
```

It applies the two-criterion test derived in Lesson 06 — relative error, with
an absolute escape hatch so a genuinely-zero gradient at a converged parameter
is not reported as a failure.

---

## The structure — every lesson, every time

| # | Section | What it is |
|---|---|---|
| 1 | **Plan** | What we build, why it exists, what breaks without it |
| 2 | **From scratch — the maths** | Full derivation. Every partial derivative written out, not cited |
| 3 | **Numerical** | Two or three numbers worked by hand, every intermediate shown |
| 4 | **Visualization** | Plots of the actual mechanism |
| 5 | **Scratch code** | Pure Python, loops and lists, zero imports. Reproduces §3 exactly |
| 6 | **NumPy** | The same maths vectorized |
| 7 | **TensorFlow** | Idiomatic Keras — what you would actually ship |
| 8 | **Agreement check** | hand ≡ scratch ≡ NumPy ≡ TF to 1e−6, asserted |
| 9 | **The silent failure** | The bug this topic causes, triggered on purpose |
| 10 | **Exercises** | Self-checking `assert` tasks; solutions in a separate notebook |

Section 8 is the point of the whole project. It is easy to *read* a derivation
and believe you understood it. It is not possible to fake three independent
implementations landing on the same number.

---

## Curriculum

| Phase | Lessons | Topic |
|---|---|---|
| 0 | 01–05 | Math foundations — tensors, matmul, chain rule, gradient descent, entropy |
| 1 | 06–15 | The first networks — regression → perceptron → MLP → **backprop** → optimizers |
| 2 | 16–21 | Making training work — overfitting, regularization, dropout, norm, schedules |
| 3 | 22–28 | Convolutional nets — convolution by hand → LeNet → transfer learning |
| 4 | 29–34 | Sequences — embeddings, RNN, BPTT, LSTM/GRU, seq2seq + attention |
| 5 | 35–41 | Transformers — attention from scratch → **mini-GPT** → train a tiny LM |
| 6 | 42–45 | Generative — autoencoder, VAE, GAN, diffusion |

Full list with goals and status: [PROGRESS.md](PROGRESS.md).

---

## Working on it

Lessons are **authored as plain Python** in [`lessons/`](lessons/) using percent-format
cells, and compiled to `.ipynb`. That is deliberate: a notebook's git diff is
unreadable JSON, and this repo gets a commit every day.

```bash
python tools/track.py status          # the todo board
python tools/track.py next            # what to do next

python tools/new_lesson.py 4          # scaffold lesson 4, mark it in-progress
#   ... fill in lessons/04-gradient-descent-1d.py ...
python tools/build_nb.py 4            # compile  -> notebooks/04-*.ipynb
python tools/verify.py 4              # execute headless; fails on any error
python tools/track.py done 4          # mark complete

./tools/push_today.ps1 4              # verify, commit, push
```

Lessons can be done **in any order** — the tracker records what was finished
and when, not just what is next in sequence.

### Requirements

```bash
pip install -r requirements.txt
```

TensorFlow is optional locally: section 7 of every notebook skips itself if TF
is missing, and runs normally on Kaggle where it is preinstalled.

### The site

```bash
python -m http.server -d docs 8000    # then open http://localhost:8000
```

Static, no build step. `tools/track.py` regenerates `docs/data/progress.json`
on every status change, so the site cannot drift from the repo.

It is published straight from `main` + `/docs` (Settings > Pages > Deploy from
a branch), so a push updates the live site with no workflow involved.

---

## Why build it this way

Most deep learning tutorials show you `model.fit()` and call it understanding.
This one makes you compute a gradient on paper, then proves your code agrees
with your paper. The three implementations exist so that the abstraction is
never load-bearing: when a framework does something surprising, you will have
already written the loop it is hiding.
