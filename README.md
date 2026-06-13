# Predicting Turbofan Failure via Custom NumPy Recurrent Neural Networks

## 1. Project Overview & Constraints

The goal of this system is to ingest continuous, multi-channel telemetry streams from aircraft turbofan engines (NASA's CMAPSS dataset) and predict their **Remaining Useful Life (RUL)**—the precise number of flight cycles an engine can sustain before mechanical failure.

### Why Standard Neural Networks Fail Here

A standard Feedforward Neural Network (FFNN) maps an isolated input vector $x$ directly to an output $\hat{y}$, assuming total statistical independence between samples. In complex mechanical and thermodynamic systems, a static snapshot of a sensor is meaningless without context:

* If an engine core temperature sensor reads **180°C** and has been steady for 50 cycles, the thermal state is stable.
* If it reads **180°C** but was at **120°C** just two cycles ago, the engine is experiencing critical thermal runaway.

An FFNN is completely blind to this temporal trend. To accurately model structural wear, the system must process the degradation trajectory across time.

### Why Pure NumPy?

This project entirely bans deep learning frameworks like PyTorch or TensorFlow. The complete data windowing pipeline, forward activation graph, recursive backward matrix calculus, and gradient clipping constraints are written from scratch in pure NumPy. This approach forces total transparency over tensor alignments and the mathematical realities of sequential backpropagation.

---

## 2. Data Pipeline & Sliding Windows

The raw NASA CMAPSS (FD001) dataset enters as a flat 2D matrix of shape $(M, N)$, where $M$ is the total aggregated cycles across the entire fleet and $N$ represents the operational settings and 21 distinct sensors.

### Engine-Bounded Windowing

To keep the temporal sequence intact, the pipeline groups data rows strictly by their unique `Engine_ID`. For an engine that operated for 200 cycles, a historical window of length $T$ (set to 30 flights) slides across its timeline.

Cross-contamination between engines is strictly forbidden. If a window were allowed to overlap from the final, failed state of Engine #1 into the initial, pristine state of Engine #2, it would introduce artificial noise and corrupt the training state. The bounded slicing generates a clean 3D temporal tensor:

$$\mathbf{X} \in \mathbb{R}^{m \times T \times n_x}$$

Where:

* $m$: Total valid overlapping historical windows across the fleet.
* $T$: Continuous sequence history length (30 lookback steps).
* $n_x$: Number of normalized input features.

### Feature Scaling & The Zero-Variance Trap

To prevent future telemetry signatures from leaking into the training state, Z-score normalization constants are calculated exclusively from the training partition:

$$\mu = \frac{1}{M_{\text{train}}} \sum_{i=1}^{M_{\text{train}}} x_i, \quad \sigma = \sqrt{\frac{1}{M_{\text{train}}} \sum_{i=1}^{M_{\text{train}}} (x_i - \mu)^2}$$

If a sensor exhibits zero variance ($\sigma = 0$) across cycles, its divisor is clamped to **1.0** to prevent division-by-zero errors from filling the tensor with `NaN` strings. Standardizing these inputs ensures a symmetrical loss landscape, allowing Gradient Descent to march efficiently toward the global minimum.

---

## 3. RNN Architecture & Vectorized Forward Pass

At each time step $t$ within a sequence, the network updates an internal coordinate space called the **Hidden State** ($h_t$), which acts as the working memory of historical degradation.

### The Structural Equations

For a single sequence sample at time step $t$:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$\hat{y} = W_{hy} h_T + b_y$$

Where the parameter spaces are initialized using Xavier/Glorot scale factors:

* $W_{xh} \in \mathbb{R}^{n_h \times n_x}$: Weights projecting input features to hidden space.
* $W_{hh} \in \mathbb{R}^{n_h \times n_h}$: Recurrent weights transforming the previous memory state.
* $W_{hy} \in \mathbb{R}^{1 \times n_h}$: Output weights mapping the final memory state to the scalar RUL prediction.
* $b_h \in \mathbb{R}^{n_h \times 1}, \quad b_y \in \mathbb{R}^{1 \times 1}$: System biases.

### Batch Vectorization

To eliminate slow Python loops over the batch dimension $m$, the forward pass equations are vectorized. At any given time step $t$, the slice of all batch samples is compiled into a single matrix $\mathbf{X}_t \in \mathbb{R}^{n_x \times m}$. The state update executes via parallel matrix dot products:

$$\mathbf{Z}_t = \mathbf{W}_{xh} \mathbf{X}_t + \mathbf{W}_{hh} \mathbf{H}_{t-1} + \mathbf{b}_h \quad \in \mathbb{R}^{n_h \times m}$$

$$\mathbf{H}_t = \tanh(\mathbf{Z}_t) \quad \in \mathbb{R}^{n_h \times m}$$

> **Note on Parallelization:** While the batch dimension is fully parallelized, the time dimension ($T$) must execute sequentially. Because $\mathbf{H}_t$ has a direct causal dependency on $\mathbf{H}_{t-1}$, recurrent architectures contain an inherent temporal bottleneck.

### Hidden Activation: Why Tanh?

The activation function for the hidden state is strictly bounded to $\tanh$ over Sigmoid for two specific reasons:

1. **Zero-Centered Output:** $\tanh$ outputs range from $[-1, +1]$, ensuring that the hidden state vectors maintain a mean close to zero across training epochs, stabilizing the scale of updates.
2. **Gradient Longevity:** The derivative is $\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$, which yields a maximum gradient value of **1.0** at the origin. Sigmoid maxes out at **0.25**, which accelerates the vanishing gradient problem by a factor of 4 at every single time step.

---

## 4. Backpropagation Through Time (BPTT)

The optimization engine minimizes Mean Squared Error (MSE) loss across all batch samples:

$$J = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

### The Temporal Chain Rule

During the backward pass, the error gradient must flow in reverse through both the parameter layers and sequential time steps. The derivative of the loss with respect to the final prediction output is initialized as:

$$\delta_y = \frac{\partial J}{\partial \hat{y}} = \frac{1}{m} (\hat{y} - y) \quad \in \mathbb{R}^{1 \times m}$$

This error directly determines the parameter adjustments at the final step:

$$\frac{\partial J}{\partial W_{hy}} = \delta_y \mathbf{H}_{T-1}^T, \quad \frac{\partial J}{\partial b_y} = \sum_{\text{batch}} \delta_y$$

The error signal is then injected into the hidden memory chain, generating the initial hidden gradient:

$$\delta_{h, T-1} = W_{hy}^T \delta_y \quad \in \mathbb{R}^{n_h \times m}$$

To update the core recurrent weights ($W_{hh}, W_{xh}$), the gradient steps backward recursively from $t = T-1$ down to $t = 0$. At each step, the error passes through the non-linear activation barrier using an element-wise Hadamard product ($\odot$):

$$\delta_{z, t} = \delta_{h, t} \odot (1 - \mathbf{H}_t^2)$$

The updates are accumulated across the entire timeline:

$$\frac{\partial J}{\partial W_{hh}} += \delta_{z, t} \mathbf{H}_{t-1}^T, \quad \frac{\partial J}{\partial W_{xh}} += \delta_{z, t} \mathbf{X}_t^T, \quad \frac{\partial J}{\partial b_h} += \sum_{\text{batch}} \delta_{z, t}$$

The gradient is then propagated backward to the previous time step via matrix multiplication with the recurrent weight transpose:

$$\delta_{h, t-1} = \mathbf{W}_{hh}^T \delta_{z, t}$$

### Exploding Gradient Protection

Because $\delta_{h, t-1}$ relies on consecutive multiplications of $\mathbf{W}_{hh}^T$ over $T$ steps, sudden sensor anomalies can cause gradients to blow up exponentially. To prevent arithmetic overflow and `NaN` corruptions, Gradient Norm Clipping is implemented right before parameter updates:

$$\text{if } \|g\| > C: \quad g = C \frac{g}{\|g\|} \quad \text{where } C = 5.0$$

---

## 5. Engineering Challenges & Debugging Scars

Building this system in pure NumPy without high-level framework abstractions forced manual resolution of several structural and mathematical hurdles:

* **The 3D Batch Transpose Realignment:** In the forward loop, processing temporal slices required isolating the batch dimension $m$ and the feature dimension $n_x$ smoothly. Correctly mapping `X_batch[:, t, :].T` to yield a clean $(n_x, m)$ matrix without scrambling the sample-to-feature mapping required multiple dry runs on paper.
* **Diagnosing the Vanishing Gradient:** During early test runs, the training loss flatlined completely. By printing the norm of `dh_next` during the temporal loop, the error signal was observed vanishing down to $10^{-7}$ before reaching $t=0$. Tuning the weight initializations and implementing an explicit gradient norm clipping threshold of **5.0** kept the updates stable.
* **BPTT Dimension Matching:** Tracking matrix transpose alignments during the backward loop was highly delicate. Ensuring that `np.dot(dtanh, h_prev.T)` cleanly produced the correct $(n_h, n_h)$ matrix shape required for updating $dW_{hh}$ required strict matching of inner dimensions across the entire historical path.

---

## 6. Experimental Results & Verification

### The Asymmetric NASA Scoring Function

In propulsion systems, prediction errors are not symmetrical. Over-predicting RUL means the model thinks an engine has more life left than it actually does, risking catastrophic mid-flight failure. Under-predicting RUL merely triggers early maintenance, causing a minor economic penalty. To enforce this, the model is evaluated against the official asymmetric NASA scoring law:

$$S = \begin{cases} e^{-\left(\frac{\hat{y} - y}{13}\right)} - 1 & \text{for } (\hat{y} - y) < 0 \quad \text{(Early Prediction - Safe)} \\ e^{\left(\frac{\hat{y} - y}{10}\right)} - 1 & \text{for } (\hat{y} - y) \ge 0 \quad \text{(Late Prediction - Critical)} \end{cases}$$

$$\text{Total Fleet Penalty} = \sum_{i=1}^{M_{\text{test}}} S_i$$

### Verified Baseline Metrics (FD001 Test Set)

After training convergence, the custom NumPy model achieved the following deterministic scores on the unseen verification set:

* **Mean Absolute Error (MAE):** 14.60 Flight Cycles
* **Root Mean Squared Error (RMSE):** 19.83 Flight Cycles
* **Official NASA Penalty Score:** 1682.27

The RMSE of 19.83 confirms that a standard recurrent architecture built entirely out of raw array operations can successfully track high-dimensional telemetry trends without relying on pre-built black-box framework classes.

---

## Collaborators & Tools

* **Developer:** Sanjay Kushwaha
* **Tools:** Mathematical logic and architecture designed from first principles; syntax optimization and markdown formatting accelerated with the assistance of LLM code generation tools.