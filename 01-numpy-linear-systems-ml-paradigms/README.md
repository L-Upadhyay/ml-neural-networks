# NumPy, Linear Systems, and ML Foundations

A single notebook covering the mathematical and computational bedrock that most machine learning methods build on: array operations in NumPy, solving linear systems (including the overdetermined case via the Moore-Penrose pseudoinverse), ML paradigm taxonomy, and the theory behind regularisation and cross-validation.

---

## Topics Covered

**1. NumPy Array Fundamentals**
- Integer sequence creation and 2-D reshaping with `arange` + `reshape`
- Evenly-spaced arrays with `linspace`
- A reusable `matrix_round(arr, precision)` utility built on `np.round`

**2. Matrix Inversion — Vandermonde Example**
- Computing $V^{-1}$ for a 5×5 Vandermonde matrix via `np.linalg.inv`
- Verifying $V^{-1}V = VV^{-1} = I$ (identity matrix, up to floating-point noise)
- NumPy commands for identity generation: `np.eye(n)`, `np.identity(n)`

**3. Solving Square Linear Systems**
- Expressing $3x - y + 2z = 3,\ -y + z = 5,\ 2x - 3z = -4$ in matrix form $A\mathbf{b} = \mathbf{c}$
- Solving via $\mathbf{b} = A^{-1}\mathbf{c}$

**4. Overdetermined Systems and the Pseudoinverse**
- 5-equation, 3-unknown system where no exact solution exists
- Why `np.linalg.inv` raises an error on non-square matrices
- Moore-Penrose pseudoinverse via `np.linalg.pinv`
- Verifying $A^+A = I$ and computing the least-squares solution
- Residual analysis confirming the solution is approximate

**5. ML Paradigm Classification (20 tasks)**
- Supervised, Unsupervised, Reinforcement Learning, and No Learning
- Each classification accompanied by a one-line justification

**6. Regularisation and Cross-Validation**
- Why $\lambda^*$ cannot be selected from training data (bias argument using the regularised MSE loss)
- Polynomial feature expansion: term count via $\binom{K+d}{d}$, full predictor for $K=2,\ d=3$
- Cross-validation workflow: how $\lambda^*$ transfers to the final retraining step

---

## How to Run

```bash
pip install numpy
jupyter notebook numpy-linear-algebra-ml-foundations.ipynb
```

No external data files required — all arrays are constructed inline.

---

## Key Results

| Task | Result |
|------|--------|
| Vandermonde inverse check | $V^{-1}V \approx I$ ✓ |
| Square system solution | $\mathbf{b} \approx [-0.909,\ -4.273,\ 0.727]^T$ |
| Pseudoinverse check | $A^+A \approx I$ ✓ |
| Overdetermined LS solution | $\hat{\mathbf{b}} \approx [0.168,\ -0.066,\ 0.481]^T$ |
| Polynomial terms ($K=2, d=3$) | 10 terms via $\binom{5}{3}$ |
