
DATE : 2026/06/19 
Version : 1.0.0

# Digit Recognizer — Neural Network from Scratch (NumPy)

A 2-layer neural network built **from scratch using only NumPy** (no TensorFlow, no PyTorch) to classify handwritten digits from the classic [Kaggle Digit Recognizer / MNIST dataset](https://www.kaggle.com/competitions/digit-recognizer). 

#-------------The Pipe Line---------------
inputlayer---forwardpass---hiddenlayers---outputlayer---calculateloss-backpropagation(BGD)---updatingparms---repeat. 

## Overview

| **Dataset** | [MNIST](https://www.kaggle.com/competitions/digit-recognizer) — 28×28 grayscale handwritten digit images (flattened to 784 pixels) |
| **Task** | Multi-class classification (digits 0–9) |
| **Framework** | Pure NumPy — no autograd, every gradient derived and coded by hand |
| **Architecture** | Input (784) → Hidden (10, sigmoid) → Output (10, softmax) |
| **Training method** | Batch gradient descent |
| **Loss** | Cross-entropy (implicitly, via softmax + one-hot targets) |

The goal of this project isn't to get state-of-the-art accuracy (a CNN would crush this) — it's to implement the **fundamentals of a neural network by hand**: matrix math, activation functions, backprop derivatives, and the training loop, with zero high-level ML libraries.


## Architecture diagram
                                               └─────────────────────────┘
```

                 W1, b1                      W2, b2
 ┌──────────────┐        ┌───────────────┐        ┌───────────────┐        ┌─────────────────┐
 │ Input layer  │ ─────▶ │ Hidden layer  │ ─────▶ │ Output layer  │ ─────▶ │ Predicted digit │
 │ 784 pixels   │        │ 10 neurons    │        │ 10 neurons    │        │      0–9        │
 │ (28×28 image)│        │ sigmoid       │        │ softmax       │        │                 │
 └──────────────┘        └───────────────┘        └───────────────┘        └─────────────────┘
                                ▲                          │
                                │                          ▼
                                │             ┌─────────────────────────┐
                                └──────────────│      Backpropagation     │
                                  error signal │ compares prediction vs.  │
                                               │ one-hot true label,      │
                                               │ computes dW1,db1,dW2,db2 │
                                               └─────────────────────────┘
```


**Forward pass equations:**

```
Z1 = W1 · X + b1        # weighted sum into hidden layer
A1 = sigmoid(Z1)        # hidden layer activation
Z2 = W2 · A1 + b2       # weighted sum into output layer
A2 = softmax(Z2)        # output layer activation (probability distribution over 10 digits)
```

**Shapes** (with `m` = number of training examples):

| Tensor | Shape | Meaning |
|---|---|---|
| `X` | `(784, m)` | input pixels, normalized to `[0, 1]` |
| `W1` | `(10, 784)` | weights from input → hidden |
| `b1` | `(10, 1)` | hidden layer biases |
| `Z1`, `A1` | `(10, m)` | hidden layer pre/post activation |
| `W2` | `(10, 10)` | weights from hidden → output |
| `b2` | `(10, 1)` | output layer biases |
| `Z2`, `A2` | `(10, m)` | output layer pre/post activation (final predictions) |

---

## How it works

### 1. Data preparation

```python
data = pd.read_csv("/kaggle/input/competitions/digit-recognizer/train.csv")
data = np.array(data)
r, c = data.shape
np.random.shuffle(data)
```

- The raw CSV has one row per image: the **first column is the label** (0–9), the remaining 784 columns are pixel intensities (0–255).
- Data is converted to a NumPy array and **shuffled** so the dev/train split isn't biased toward any particular digit ordering, and so mini-batches (if used later) aren't correlated.

The shuffled data is then split into a small **dev/validation set** and a larger **training set**:

```python
data_ = data[0:1000].T          # dev set, transposed so each column = one example
Y_dev = data_[0]                 # labels
X_dev = data_[1:c] / 255.0       # pixels, normalized to [0, 1]

data_Train = data[1000:r].T      # training set
Y_Train = data_Train[0]
X_Train = data_Train[1:c] / 255.0
```

- **Transposing** turns the data into `(features, examples)` shape — the convention this network's forward/backward math expects, so a full batch can be processed as one matrix multiplication.
- **Normalization** (`/ 255.0`) scales pixel values from `[0, 255]` to `[0, 1]`, which keeps the activations and gradients in a numerically stable range — sigmoid in particular saturates and its gradient vanishes when inputs are large.

---

### 2. Parameter initialization

```python
def init_Para():
    W1 = np.random.rand(10, 784)   # 10 neurons, each with 784 weights (one per pixel)
    b1 = np.random.rand(10, 1)     # 1 bias per hidden neuron
    W2 = np.random.rand(10, 10)    # 10 output neurons, each with 10 weights (one per hidden neuron)
    b2 = np.random.rand(10, 1)     # 1 bias per output neuron
    return W1, b1, W2, b2
```

Weights and biases are initialized with `np.random.rand`, which draws from a **uniform distribution in `[0, 1)`**.

> **Note:** this is a simple initialization scheme. It works for a shallow 2-layer network like this one, but see [Known issues](#known-issues--improvement-ideas) for why it isn't ideal at scale.

---

### 3. Forward propagation

```python
def sigmoid(Zi):
    return 1 / (1 + np.exp(-Zi))

def softmax(Zi):
    exp_Z = np.exp(Zi - np.max(Zi, axis=0, keepdims=True))  # prevents overflow
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_pass(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
```

- **Sigmoid** squashes the hidden layer's weighted sums into `(0, 1)` — used here as the hidden-layer nonlinearity.
- **Softmax** converts the output layer's 10 raw scores into a probability distribution that sums to 1 across the 10 digit classes. The `max` subtraction is a standard numerical-stability trick that prevents `np.exp` from overflowing on large inputs, without changing the result (softmax is shift-invariant).

---

### 4. One-hot encoding

```python
def onehot(number_of_lables, number_of_neurons=10):
    n = number_of_lables.shape[0]
    Output_matrix = np.zeros((n, number_of_neurons))
    for x, i in enumerate(number_of_lables):
        Output_matrix[x][i] = 1
    return Output_matrix.astype(int).T
```

Since the output layer produces a 10-dimensional probability vector, the true label (a single integer 0–9) needs to be converted into the same shape to compute an error signal. A label of `4`, for example, becomes:

```
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
```

The function builds this for every example in the batch and transposes the result to match the `(10, m)` shape used everywhere else in the network.

---

### 5. Backpropagation

```python
def Backprop_single(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hotY = onehot(Y)

    dz2 = A2 - one_hotY
    dw2 = 1 / m * dz2.dot(A1.T)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)

    dz1 = W2.T.dot(dz2) * derv_sigmoid(Z1)
    dw1 = 1 / m * dz1.dot(X.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2
```

This is the chain rule applied layer by layer, back-to-front:

1. **Output layer error** — `dz2 = A2 - one_hotY`. This clean form is a result of pairing softmax with cross-entropy loss: the gradient of cross-entropy loss with respect to the pre-softmax logits simplifies to "prediction minus target."
2. **Output layer gradients** — `dw2` and `db2` are the average (over all `m` examples) contribution of each hidden activation / bias to the output error.
3. **Propagate error backward** — `dz1 = W2.T.dot(dz2) * derv_sigmoid(Z1)`. The error is projected back through `W2` into the hidden layer, then multiplied by the **derivative of sigmoid** (chain rule) to get the hidden layer's own error signal.
4. **Hidden layer gradients** — `dw1` and `db1` follow the same averaging pattern as the output layer, but using the input `X` instead of `A1`.

```python
def derv_sigmoid(Z):
    s = sigmoid(Z)
    return s * (1 - s)
```

---

### 6. Parameter updates (gradient descent)

```python
def Updating_Param(w1, b1, w2, b2, dw1, db1, dw2, db2, omega):
    w1 = w1 - omega * dw1
    b1 = b1 - omega * db1
    w2 = w2 - omega * dw2
    b2 = b2 - omega * db2
    return w1, b1, w2, b2
```

Standard gradient descent step: each parameter moves a small step (`omega`, the learning rate) in the direction that **reduces** the loss — i.e. opposite to the gradient.

---

### 7. Training loop

```python
def Batch_GD(X, Y, itter, omega):
    w1, b1, w2, b2 = init_Para()
    for i in range(itter):
        z1, a1, z2, a2 = forward_pass(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = Backprop_single(z1, a1, z2, a2, w2, X, Y)
        w1, b1, w2, b2 = Updating_Param(w1, b1, w2, b2, dw1, db1, dw2, db2, omega)
        if i % 50 == 0:
            print("Total iterations:", i)
            print("Accuracy:", get_accuracy(get_predections(a2), Y))
    return w1, b1, w2, b2
```

Each iteration:
1. Runs a **full batch** forward pass over the entire training set (no mini-batching).
2. Computes gradients via backprop.
3. Updates all four parameters.
4. Every 50 iterations, prints predictions vs. true labels and the running accuracy.

```python
def get_predections(A2):
    return np.argmax(A2, 0)

def get_accuracy(predections, Y):
    return np.sum(predections == Y) / Y.size
```

`get_predections` simply picks the digit with the highest predicted probability for each example. `get_accuracy` is the fraction of correct predictions.

---

## Math reference

For anyone newer to backprop, here's the chain of reasoning condensed into one table:

| Quantity | Formula | Intuition |
|---|---|---|
| `Z1` | `W1·X + b1` | raw signal into hidden neurons |
| `A1` | `sigmoid(Z1)` | squashed hidden activation, range (0,1) |
| `Z2` | `W2·A1 + b2` | raw signal into output neurons |
| `A2` | `softmax(Z2)` | predicted probability distribution over digits 0–9 |
| `dZ2` | `A2 - Y_onehot` | how wrong each output neuron is |
| `dW2` | `(1/m)·dZ2·A1ᵗ` | how much to adjust hidden→output weights |
| `dZ1` | `(W2ᵗ·dZ2) ⊙ sigmoid'(Z1)` | how much the hidden layer contributed to the output's error |
| `dW1` | `(1/m)·dZ1·Xᵗ` | how much to adjust input→hidden weights |

---

## Project structure

```
.
├── README.md                 # you are here
└── notebook.ipynb            # full notebook: data loading, training, evaluation
```

---

## Getting started

1. **Get the data.** Download `train.csv` from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/data), or run this notebook directly on Kaggle where the dataset is already mounted at `/kaggle/input/competitions/digit-recognizer/train.csv`.

2. **Install dependencies:**

   ```bash
   pip install numpy pandas matplotlib
   ```

3. **Run the notebook** cell by cell, or convert it to a script:

   ```bash
   jupyter nbconvert --to script notebook.ipynb
   python notebook.py
   ```

4. **Train the model:**

   ```python
   w1, b1, w2, b2 = Batch_GD(X_Train, Y_Train, itter=500, omega=0.1)
   ```

   Increase `itter` (iterations) for more training, and tune `omega` (the learning rate) if accuracy plateaus or diverges.

5. **Evaluate on the dev set:**

   ```python
   _, _, _, A2_dev = forward_pass(w1, b1, w2, b2, X_dev)
   print(get_accuracy(get_predections(A2_dev), Y_dev))
   ```

---

## Current results

In the notebook as currently run, training for 100 iterations at `omega = 0.1` gets accuracy from **~9.7%** (essentially random — there are 10 classes) up to only **~11.1%** after 50 iterations. This is **far below what this architecture is capable of** — see below for why.

---

## Known issues / improvement ideas

A few things in the current implementation are limiting accuracy and worth fixing if you want to push this further:

1. **Weight initialization range.** `np.random.rand(10, 784)` draws from `[0, 1)` — these are large, all-positive initial weights for a 784-input layer. This tends to push `Z1` values to extremes, saturating the sigmoid (gradients near 0) and slowing — or stalling — learning. Try centering and scaling, e.g. `np.random.randn(10, 784) * np.sqrt(1 / 784)` (a basic form of He/Xavier-style initialization).
2. **Learning rate / iteration count.** 100 iterations at `omega = 0.1` is very little training for full-batch gradient descent on 41,000 examples. Try more iterations (1,000–5,000) and experiment with the learning rate.
3. **Sigmoid in the hidden layer.** ReLU (`max(0, Z)`) is the more common choice for hidden layers in modern networks — it avoids the vanishing-gradient problem that sigmoid has for large `|Z|`, and is cheaper to compute.
4. **Batch gradient descent only.** Every iteration processes all 41,000 training examples at once. Mini-batch or stochastic gradient descent usually converges faster and can escape poor regions of the loss surface more easily.
5. **No bias correction for shape.** `b1`/`b2` are `(10, 1)` and broadcast correctly against `(10, m)` matrices — this part is fine, just noting it as a common source of bugs in from-scratch implementations.

---
## NOTE -- SOMEPARTS OF THIS README WERE WRITTEN BY CLAUDE ( IM LAZY)
## License

This project is for educational purposes, built around the publicly available [MNIST-derived Kaggle Digit Recognizer dataset](https://www.kaggle.com/competitions/digit-recognizer). Feel free to fork, modify, and experiment.
