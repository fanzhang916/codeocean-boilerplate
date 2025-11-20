# Iris Classification
<img src="./irir_flowers.png" alt="Iris">

  The project demonstrates a complete end-to-end pipeline for training and evaluating a feedforward neural network on the classic Iris dataset, a multiclass classification problem containing 150 samples across three species of iris flowers (Setosa, Versicolor, and Virginica).

  The implementation showcases best practices in reproducible research, including containerization, dependency management, comprehensive logging, and artifact preservation. This template serves as a starting point for packaging your own machine learning experiments with reproducibility for code ocean deployment.

  **Project Structure (template)**

  - `code/`: Core Python source code
    - `nn_example.py`: Neural network training and evaluation driver.
    - `analyze.py`: Analysis and reporting utilities that consume model outputs.
    - `data_loader/data_module.py`: Dataset and data-loading logic.
    - `models/model.py`: Model architecture used in the example.
    - `utils/metrics_reporter.py`: Helpers to compute and report metrics.
  - `data/`:
    - `Iris.csv`: Example dataset used by the example model.
  - `results/`:
    - `analysis_report.txt`: Sample analysis output.
    - `iris_nn_model.pth`: Trained model weights saved from example training.

  ## Model Architecture and Mathematical Description

### Network Architecture

The model implements a fully-connected feedforward neural network (Multi-Layer Perceptron) with the following architecture:

```
Input Layer (4 features) → Hidden Layer 1 (64 units) → Hidden Layer 2 (32 units) → Output Layer (3 classes)
```

**Layer Specifications:**
- **Input Layer**: 4 features (sepal length, sepal width, petal length, petal width)
- **Hidden Layer 1**: 64 neurons with ReLU activation
- **Hidden Layer 2**: 32 neurons with ReLU activation
- **Output Layer**: 3 neurons (one per class) with no activation (logits)

### Technical Details

#### Forward Pass

Given an input vector **x** ∈ ℝ⁴, the forward propagation through the network is defined as:

**Layer 1 (Input → Hidden1):**

$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$

where:
- $\mathbf{W}_1 \in \mathbb{R}^{64 \times 4}$ is the weight matrix
- $\mathbf{b}_1 \in \mathbb{R}^{64}$ is the bias vector
- $\text{ReLU}(z) = \max(0, z)$ is the activation function

**Layer 2 (Hidden1 → Hidden2):**

$$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$$

where:
- $\mathbf{W}_2 \in \mathbb{R}^{32 \times 64}$ is the weight matrix
- $\mathbf{b}_2 \in \mathbb{R}^{32}$ is the bias vector

**Layer 3 (Hidden2 → Output):**

$$\mathbf{z} = \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3$$

where:
- $\mathbf{W}_3 \in \mathbb{R}^{3 \times 32}$ is the weight matrix
- $\mathbf{b}_3 \in \mathbb{R}^{3}$ is the bias vector
- $\mathbf{z} \in \mathbb{R}^{3}$ represents the logits for each class

#### Loss Function

The model uses **Cross-Entropy Loss** for multiclass classification:

$$\mathcal{L}(\mathbf{z}, y) = -\log\left(\frac{e^{z_y}}{\sum_{j=1}^{3} e^{z_j}}\right) = -z_y + \log\left(\sum_{j=1}^{3} e^{z_j}\right)$$

where:
- $y \in \{0, 1, 2\}$ is the true class label
- $z_y$ is the logit corresponding to the true class
- The first term is equivalent to applying softmax followed by negative log-likelihood

#### Optimization

The network parameters $\theta = \{\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2, \mathbf{W}_3, \mathbf{b}_3\}$ are optimized using **Adam optimizer** with learning rate $\alpha = 0.01$:

$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected first and second moment estimates of the gradients.

#### Prediction

For inference, the predicted class is obtained by:

$$\hat{y} = \underset{j \in \{0,1,2\}}{\text{argmax}} \, z_j$$

### Training Configuration
- **Epochs**: 50
- **Learning Rate**: 0.01
- **Optimizer**: Adam
- **Batch Size**: 16 (defined in data module)
- **Loss Function**: Cross-Entropy Loss
- **Device**: CUDA if available, otherwise CPU

  ## Citation

If you use this template in a publication, please cite:

```bibtex
@software{iris_nn_template_2025,
  title = {Reproducibility Capsule Template},
  author = {[Your Name/Organization]},
  year = {2025},
  url = {[Repository URL]},
  doi = {[DOI if available]}
}
```

  **License**

  This repository uses the MIT License by default. Replace or update this license if your institution requires a different license.
