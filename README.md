# 🚀 Self-Pruning Neural Network

## 📌 Overview

This project implements a **Self-Pruning Neural Network** that dynamically removes unnecessary weights during training using learnable gating mechanisms.

Unlike traditional pruning (post-training), this model **learns to prune itself** by optimizing both accuracy and sparsity simultaneously.

---

## 🧠 Key Idea

Each weight in the network is assigned a **learnable gate value (0–1)**:

* Gate ≈ 1 → Weight is active
* Gate ≈ 0 → Weight is pruned

The model uses:

* **Sigmoid function** → Converts gate scores into probabilities
* **L1 regularization** → Encourages gates to become zero

---

## ⚙️ Architecture

* Custom `PrunableLinear` layer
* Fully connected neural network:

  * Input → 3072 (CIFAR-10)
  * Hidden Layers → 256 → 128
  * Output → 10 classes

---

## 📉 Loss Function

Total Loss:

Classification Loss + λ × Sparsity Loss

Where:

* Classification Loss → CrossEntropy
* Sparsity Loss → Sum of all gate values (L1 penalty)

---

## 📊 Dataset

* **CIFAR-10** (via torchvision)
* 10 image classes
* Input size: 32×32 RGB

---

## 🚀 Features

✅ Custom prunable layer
✅ Learnable gating mechanism
✅ Dynamic weight pruning during training
✅ L1 sparsity regularization
✅ Hard pruning (threshold-based)
✅ Sparsity vs Accuracy comparison
✅ Gate distribution visualization

---

## 📈 Results

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 0.001  | High     | Low      |
| 0.01   | Medium   | Medium   |
| 0.1    | Lower    | High     |

👉 Higher λ → More pruning but lower accuracy

---

## 📊 Output Visualization

* Histogram of gate values
* Shows:

  * Spike near 0 → pruned weights
  * Cluster away from 0 → important weights

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python self_pruning_nn_final.py
```

---

## 📁 Project Structure

```
self-pruning-neural-network/
├── src/
├── results/
├── notebook/
├── self_pruning_nn_final.py
├── requirements.txt
└── README.md
```

---

## 🧪 Experimentation

Modify lambda values in the script:

```python
lambda_values = [0.001, 0.01, 0.1]
```

---

## 🎯 Key Insights

* L1 regularization forces many gates → 0
* Enables automatic feature selection
* Achieves model compression without manual pruning

---

## 💬 Interview Summary

“I built a neural network that learns to prune itself during training by introducing learnable gates on each weight and applying L1 regularization to enforce sparsity.”

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Ram
