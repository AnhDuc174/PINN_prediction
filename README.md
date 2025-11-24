# 🌀 Hybrid Physics-Informed Neural Network (PINN) for Flow Past a Cylinder

**URANS + RNG k-ε turbulence model | Hybrid data + physics constraints | ResNet and Feed-forward architectures**

This repository contains a **Physics-Informed Neural Network (PINN)** designed to simulate **2D unsteady flow past a circular cylinder** at high Reynolds numbers. The model aims to **capture vortex shedding** behavior using a **hybrid approach** combining:

* Governing **URANS PDE residuals**
* **CFD data supervision** (sparse)
* Physical constraints (non-negativity, incompressibility, boundary conditions)

The project includes a comparison between two neural architectures:

1. **Feed-forward PINN (FFNN)**
2. **ResNet-based PINN**

---

## 🎯 Objective

Predict unsteady flow fields governed by **URANS + RNG k-ε** model:

* **Inputs:** (x, y, t)
* **Outputs:**

  * ψ – Streamfunction
  * p – Pressure
  * k – Turbulent kinetic energy
  * ε – Dissipation rate

Velocity is derived from the streamfunction:
[ u = \partial_y \psi, \quad v = -\partial_x \psi ]

The main goal is to model wake dynamics such as **vortex shedding**, which requires capturing complex temporal and spatial patterns.

---

## 📐 PDE Model (URANS + RNG k-ε)

We enforce residuals from:

### Continuity

[ \partial_x u + \partial_y v = 0 ]

### Momentum (URANS)

[ \frac{D u}{D t} = - \frac{1}{\rho}\partial_x p + \nabla\cdot[(\nu + \nu_t)\nabla u] ]
(similar for v)

### Turbulence transport (k, ε)

Transport equations for k and ε with production and dissipation terms; turbulent viscosity (\nu_t) computed from k and ε (RNG/k-ε relations).

Residuals are computed with PyTorch autograd.

---

## 🧠 Model Architectures

### 1. Feed-Forward Neural Network (FFNN)

* Fully connected layers with Tanh activation
* Xavier initialization
* Softplus output for k and ε (to enforce positivity)

**Pros:** simple and fast; works for lower-Re flows.

**Cons:** struggles with deep networks and stiff PDEs; may not capture vortex shedding without extra features.

### 2. ResNet PINN (recommended)

A PINN backbone built from residual blocks (skip connections). Each block learns a correction to its input:
[ h_{i+1} = h_i + f(h_i) ]

**Pros:** improved gradient flow, stable training for deeper nets, better at capturing wake structures.

**Cons:** slightly more compute and memory.

---

## 📊 Dataset & Normalization

* Inputs (x, y, t) are scaled to **[-1, 1]**:

```python
mins = xyt_tensor.min(dim=0)[0]
maxs = xyt_tensor.max(dim=0)[0]
xyt_scaled = 2 * (xyt_tensor - mins) / (maxs - mins) - 1
```

* Non-dimensionalize outputs (recommended):

  * U_ref = inlet velocity, P_ref = rho * U_ref^2
  * k_ref = U_ref^2, eps_ref = U_ref^3 / L_ref
* Predict non-dimensional targets (u/U_ref, p/P_ref, k/k_ref, eps/eps_ref)

Normalization dramatically improves PINN stability.

---

## 🧪 Training (Hybrid PINN)

Total loss is a weighted sum of:

* PDE residual loss (momentum, continuity, k-ε)
* Data loss (sparse CFD points)
* Boundary condition loss (inlet, cylinder no-slip, outlet)
* Initial condition loss
* Regularization (optional)

**Key training practices:**

* Start with lower PDE weight and gradually increase (curriculum).
* Use small learning rate (e.g., 5e-5).
* Use gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.
* Fix collocation points (don’t resample every step); optionally adaptively add points in high-residual regions.
* Disable mixed precision (AMP) while debugging; enable later if stable.

---

## ⚠️ Common issues & fixes

**Loss decreases then explodes** — Typical causes and fixes:

* Too-large learning rate ⇒ reduce to 1e-4..1e-5.
* Unbalanced loss terms ⇒ rescale residuals or adjust weights.
* Resampling collocation each iteration ⇒ fix the sample or use adaptive sampling.
* Large `k`/`ε` values due to bad output mapping ⇒ use `Softplus` and non-dimensionalization.
* High-order autograd gradients explode ⇒ use gradient clipping.

**Numerical stability tips:**

* Normalize PDE residuals by characteristic scales (U_ref, L_ref).
* Monitor per-term loss and gradient norms.
* Save best model by validation metric and use early stopping.

---

## 📈 Experiments & Observations

* **FFNN**: learned smoother flows, but often failed to capture coherent vortex shedding for higher Re without careful tuning.
* **ResNet PINN**: more stable, better at representing wake; required fewer manual tricks to converge.

---

## ⚙️ How to run

1. Place CFD data in `data/` (or update `data_dir` in the notebook to the actual path).
2. Instantiate model:

```python
# Feedforward
model = PINN(layers=[3] + [64]*10 + [4])
# or ResNet
model = ResNetPINN(in_dim=3, hidden_dim=64, n_blocks=8, out_dim=4)
```

3. Train with `train_adam(...)` provided in the notebook, passing `device=torch.device(...)`.

**Notebook path:** `/mnt/data/pinn-pde-rans.ipynb`

---

## 📁 Repository structure (suggested)

```
├── data/                     # CFD training points
├── models/                   # Saved checkpoints
├── pinn-pde-rans.ipynb       # Main notebook (training & experiments)
├── model_resnet.py           # ResNet PINN implementation
├── model_ffnn.py             # Feedforward PINN implementation
├── utils/                    # normalization, pde residuals, samplers
└── README.md                 # this file
```

---

## 📮 Next steps

* Implement adaptive collocation focusing on wake residuals.
* Try Fourier features or SIREN if ResNet cannot fully capture wake high-frequencies.
* Add grad-norm balancing for automatic loss weighting.
* Build validation plots: probe time series, residual maps, and velocity profiles.

---

If you want, I can also produce:

* A short presentation summarizing these experiments,
* A diagram of the ResNet architecture, or
* An updated training loop implementing curriculum + grad clipping.

Contact: Duc
