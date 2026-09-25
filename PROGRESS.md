# Progress

**7 / 45 lessons complete (15%)**

`[ ]` todo &nbsp;&nbsp; `[~]` in progress &nbsp;&nbsp; `[x]` done

Lessons are done in whatever order makes sense on the day; the
*Done* column records when each one actually landed.

## Phase 0 - Math Foundations (1/5)

| | # | Lesson | Goal | Done |
|---|---|---|---|---|
| `[x]` | 01 | [Tensors, Vectors and Shapes](notebooks/01-tensors-and-vectors.ipynb) | Represent data as numbers a network can eat, and never be confused by a shape again. | 2026-09-25 |
| `[ ]` | 02 | Matrix Multiplication by Hand | Compute a matmul manually and see why it is THE operation of deep learning. |  |
| `[ ]` | 03 | Derivatives and the Chain Rule | Differentiate composed functions by hand: the engine of backprop. |  |
| `[ ]` | 04 | Gradient Descent in 1D | Roll downhill on a curve and watch the learning rate make or break it. |  |
| `[ ]` | 05 | Probability, Likelihood and Entropy | Derive why cross-entropy is the loss, not an arbitrary choice. |  |

## Phase 1 - The First Networks (6/10)

| | # | Lesson | Goal | Done |
|---|---|---|---|---|
| `[x]` | 06 | [Linear Regression from Scratch](notebooks/06-linear-regression.ipynb) | Fit a line with gradient descent: the smallest complete neural network. | 2026-09-25 |
| `[x]` | 07 | [Loss Surfaces and MSE](notebooks/07-loss-surfaces.ipynb) | See the bowl you are descending and what makes it steep or flat. | 2026-09-25 |
| `[x]` | 08 | [Logistic Regression and the Sigmoid](notebooks/08-logistic-regression.ipynb) | Turn a line into a probability and derive the sigmoid gradient. | 2026-09-26 |
| `[x]` | 09 | [The Perceptron and its Limits](notebooks/09-the-perceptron.ipynb) | Build the 1958 neuron and hit the XOR wall that froze the field. | 2026-09-26 |
| `[x]` | 10 | [The MLP Forward Pass](notebooks/10-mlp-forward-pass.ipynb) | Stack layers to break the XOR wall; trace one input to one output. | 2026-09-26 |
| `[x]` | 11 | [Backpropagation, Derived in Full](notebooks/11-backpropagation.ipynb) | Derive every partial derivative of a two-layer net and verify numerically. | 2026-09-26 |
| `[ ]` | 12 | Activations: Sigmoid, Tanh, ReLU, GELU | Compare gradients, saturation and dead units across activations. |  |
| `[ ]` | 13 | Softmax and Cross-Entropy | Multi-class outputs and the famous (p - y) gradient simplification. |  |
| `[ ]` | 14 | Weight Initialization: Xavier and He | Derive the variance rules that stop signals exploding or dying. |  |
| `[ ]` | 15 | Optimizers: SGD, Momentum, RMSProp, Adam | Build each update rule from scratch and race them on one surface. |  |

## Phase 2 - Making Training Work (0/6)

| | # | Lesson | Goal | Done |
|---|---|---|---|---|
| `[ ]` | 16 | Bias, Variance and Overfitting | Watch a model memorize noise and learn to read training curves. |  |
| `[ ]` | 17 | L1 and L2 Regularization | Add a penalty term, derive its gradient, see the weights shrink. |  |
| `[ ]` | 18 | Dropout | Train an ensemble for free and get the train/test scaling right. |  |
| `[ ]` | 19 | Batch Norm and Layer Norm | Normalize activations and backprop through the statistics. |  |
| `[ ]` | 20 | Learning Rate Schedules and Warmup | Step, cosine and warmup schedules, and when each one wins. |  |
| `[ ]` | 21 | Batching, Shuffling and Data Pipelines | Mini-batch gradient descent and why batch size changes the path. |  |

## Phase 3 - Convolutional Nets (0/7)

| | # | Lesson | Goal | Done |
|---|---|---|---|---|
| `[ ]` | 22 | Convolution by Hand | Slide a kernel over a grid manually and see what each filter detects. |  |
| `[ ]` | 23 | Stride, Padding and Pooling | Control output shapes exactly and derive the size formula. |  |
| `[ ]` | 24 | Backprop Through a Convolution | Derive the gradient of a convolution: it is another convolution. |  |
| `[ ]` | 25 | LeNet on MNIST | Assemble a real CNN and train it to recognise handwritten digits. |  |
| `[ ]` | 26 | Receptive Fields and Depth | Compute what each deep neuron actually sees in the input image. |  |
| `[ ]` | 27 | Data Augmentation | Manufacture training data and measure the accuracy it buys. |  |
| `[ ]` | 28 | Transfer Learning and Filter Visualization | Reuse pretrained features and look at what the filters learned. |  |

## Phase 4 - Sequences (0/6)

| | # | Lesson | Goal | Done |
|---|---|---|---|---|
| `[ ]` | 29 | Embeddings: Words as Vectors | Learn a lookup table where geometry encodes meaning. |  |
| `[ ]` | 30 | The Vanilla RNN | Add a hidden-state loop and process a sequence step by step. |  |
| `[ ]` | 31 | Backpropagation Through Time | Unroll the loop and sum gradients across every timestep. |  |
| `[ ]` | 32 | Vanishing and Exploding Gradients | Multiply Jacobians repeatedly and watch the learning signal die. |  |
| `[ ]` | 33 | LSTM and GRU Cell Math | Build every gate by hand and see how the cell state survives. |  |
| `[ ]` | 34 | Seq2Seq and the First Attention | Encoder-decoder, the bottleneck problem, and attention as the fix. |  |

## Phase 5 - Transformers (0/7)

| | # | Lesson | Goal | Done |
|---|---|---|---|---|
| `[ ]` | 35 | Scaled Dot-Product Attention | Derive Q, K, V from scratch and justify the sqrt(d) scaling. |  |
| `[ ]` | 36 | Multi-Head Attention | Run several attention subspaces in parallel and concatenate them. |  |
| `[ ]` | 37 | Positional Encoding | Inject order into a model that is otherwise permutation invariant. |  |
| `[ ]` | 38 | The Full Transformer Block | Residuals, layer norm and the feed-forward network, assembled. |  |
| `[ ]` | 39 | Tokenization and Byte-Pair Encoding | Turn raw text into integers and implement BPE from scratch. |  |
| `[ ]` | 40 | Mini-GPT from Scratch | Causal masking and a decoder-only stack, built end to end. |  |
| `[ ]` | 41 | Training a Tiny Language Model | Train the mini-GPT on real text and generate samples from it. |  |

## Phase 6 - Generative Models (0/4)

| | # | Lesson | Goal | Done |
|---|---|---|---|---|
| `[ ]` | 42 | Autoencoders | Compress to a bottleneck and reconstruct; learn representations. |  |
| `[ ]` | 43 | Variational Autoencoders | Derive the ELBO and the KL term; use the reparameterization trick. |  |
| `[ ]` | 44 | Generative Adversarial Networks | Two networks in a minimax game, trained against each other. |  |
| `[ ]` | 45 | Diffusion Models | Add noise, learn to remove it: the idea behind modern image models. |  |

## Order completed

01 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11

---

*Generated by `tools/track.py` - do not edit by hand.*
