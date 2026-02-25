
# DyNACO: Beyond Static Priors – Dynamic Neural Guidance for Large-Scale Ant Colony Optimization

[![Conference](https://img.shields.io/badge/KDD'26-Under_Review-blue)](https://kdd.org/kdd2026/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13%2B-blue)](https://www.python.org/)

**DyNACO** is a novel Learning-Guided Optimization (LGO) framework that shifts from *static* heatmaps to **dynamic neural guidance**. By formulating ACO as a semi-MDP, DyNACO trains a meta-policy to observe the evolving pheromone field and incumbent solution, injecting updated guidance throughout the search trajectory.

Paired with a **Scope-Restricted Refinement (SRR)** backend, DyNACO scales efficiently to **100,000-node** instances, outperforming state-of-the-art neural baselines while often running *faster* than the unguided solver.

---

## 🚀 Key Features

*   **Dynamic Guidance:** Unlike prior works (DeepACO, GFACS) that predict a static heatmap once, DyNACO adapts its guidance based on the search phase (exploration vs. exploitation).
*   **Trajectory-Aware Training:** Optimizes expected cost across the full optimization history, aligning training with iterative search dynamics.
*   **Scalability:** Scales to **100K nodes** using a perturbation-based backend and $O(1)$ per-ant refinement costs.
*   **Efficiency:** Reduces total TSP runtime by **20–33%** compared to unguided ACO by accelerating convergence.
*   **Zero-Shot Transfer:** A single model trained on random 1K instances generalizes to real-world TSPLIB and CVRPlib instances up to 85K nodes.

---

## 📊 Experimental Results

DyNACO achieves state-of-the-art performance among neural methods on TSP and CVRP (1K–100K nodes).

### 1. Large-Scale Synthetic Instances (TSP & CVRP)

Comparison of Optimality Gap (%) and Runtime on 100K-node instances. DyNACO outperforms constructive baselines (SIL) and unguided ACO.

| Problem | Method | Gap (%) | Time | Scalability |
| :--- | :--- | :---: | :---: | :---: |
| **TSP-100K** | LKH-3 | 0.00% | 25h | Low |
| | SIL (PRC1000) | 2.45% | 2.6h | Med |
| | ACO (Unguided) | 3.12% | 3.7m | High |
| | **DyNACO (Ours)** | **1.90%** | **2.8m** | **High** |
| **CVRP-100K** | HGS | 0.00% | 24h | Low |
| | SIL (PRC1000) | -2.55% | 2.2h | Med |
| | ACO (Unguided) | 7.26% | 11.5m | High |
| | **DyNACO (Ours)** | **6.75%** | **11.7m** | **High** |

> **Note:** On TSP, DyNACO is **faster** than the unguided baseline because targeted neural perturbations lead to faster local search convergence.

### 2. Real-World Zero-Shot Generalization

Evaluation of the model trained on **random 1K instances** directly on TSPLIB (up to 85K nodes) and CVRPlib (up to 30K nodes) without fine-tuning.

| Benchmark | Method | Gap (%) | Time |
| :--- | :--- | :---: | :---: |
| **TSPLIB** | LEHD | 13.2% | 38m |
| (33 instances) | SIL | 3.03% | 45m |
| | ACO (Unguided) | 1.29% | 13s |
| | **DyNACO** | **0.89%** | **11s** | 
| **CVRPlib** | SIL | 7.69% | 54m 
| (14 instances) | ACO (Unguided) | 4.26% | 50s | 
| | **DyNACO** | **3.66%** | **51s** | 

---

## 🛠️ Installation

### Prerequisites
*   **OS:** Linux (Tested on Ubuntu 22.04)
*   **Python:** ≥ 3.13
*   **Hardware:** CUDA-capable GPU
*   **Compiler:** C++17 compatible compiler with OpenMP support (e.g., `g++`, `clang`)
*   **Package Manager:** [uv](https://github.com/astral-sh/uv) (recommended for fast environment management)

### Setup Steps

1.  **Clone and Sync Environment**
    ```bash
    git clone https://github.com/anonymous/DyNACO.git
    cd DyNACO
    uv sync
    ```

2.  **Build C++ Backend**
    DyNACO relies on a high-performance C++ backend for perturbation-based ACO and Scope-Restricted Refinement (SRR).
    ```bash
    cd src
    uv run python setup.py build_ext --inplace
    cd ..
    ```

3.  **Verify Installation**
    ```bash
    uv run python -c "import faco_opt; print('✅ C++ backend loaded successfully')"
    uv run python -c "import torch; print(f'✅ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    ```

---

## 🏃 Usage

### Training

DyNACO uses PPO with trajectory replay. Training a model on TSP-1K takes approximately **30 minutes** on a single RTX 5090.

**Standard Training (TSP-1K):**
```bash
uv run python train.py --problem tsp --n_node 1000 --save_dir pretrained/tsp_1k
```

**Cross-Problem Training (CVRP-1K):**
```bash
uv run python train.py --problem cvrp --n_node 1000 --save_dir pretrained/cvrp_1k
```

**Scaling Up (TSP-10K / 100K):**
```bash
# TSP-10K (~2 hours)
uv run python train.py --problem tsp --n_node 10000

# TSP-100K (~4 hours)
uv run python train.py --problem tsp --n_node 100000
```

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--problem` | - | Problem type: `tsp` or `cvrp` |
| `--n_node` | 1000 | Problem size ($N$) |
| `--k_sparse` | 32 | Size of K-NN candidate graph |
| `--H` | 10 | Number of outer guidance updates |
| `--mini_H` | 100 | Inner ACO sampling steps per update |
| `--rho` | 0.5/0.1 | Pheromone evaporation rate (CVRP/TSP) |

### Evaluation

Evaluate trained checkpoints on synthetic data or real-world benchmarks.

**Evaluate on Synthetic Data (TSP-1K):**
```bash
uv run python test.py \
    --problem tsp --n_node 1000 \
    --checkpoint pretrained/tsp/n1000/best.pt \
    --H 10 --mini_H 100
```

**Evaluate Zero-Shot on TSPLIB/CVRPlib:**
To reproduce the real-world benchmark results (using the `--rl_data` flag):
```bash
uv run python test.py \
    --problem tsp \
    --checkpoint pretrained/tsp/n1000/best.pt \
    --rl_data \
    --H 10 --mini_H 100
```

**Run Unguided Baseline:**
To verify the contribution of the neural component:
```bash
uv run python test.py --problem tsp --n_node 1000 --no_model
```

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--checkpoint` | None | Path to `.pt` model file |
| `--rl_data` | `False` | Use real-world instances (TSPLIB/CVRPlib) |
| `--no_model` | `False` | Run pure ACO without neural guidance |
| `--warmup` | `True` | Enable phased injection (inference strategy) |
| `--no_anneal` | `False` | Disable guidance annealing (inference strategy) |

---

## 📂 Project Structure

```
├── train.py          # Main training loop (PPO + Trajectory Replay)
├── test.py           # Evaluation pipeline
├── net.py            # Model Architecture (GNN Encoder + MLP Decoder)
├── faco.py           # Python wrapper for the ACO environment
├── utils.py          # Data loading and metric logging
├── src/              # C++ Backend
│   ├── mfaco_train.cpp   # Perturbation-based ACO + SRR implementation
│   ├── binding.cpp       # PyBind11 bindings
│   └── setup.py          # Build configuration
├── data/             # Datasets (TSPLIB, CVRPlib, Synthetic)
└── pretrained/       # Directory for model checkpoints
```

## 📜 Citation

If you find our work useful, please cite our paper (BibTeX will be updated upon publication):

```bibtex
@inproceedings{dynaco2026,
  title={Beyond Static Priors: Dynamic Neural Guidance for Large-Scale Ant Colony Optimization},
  author={Anonymous Authors},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year={2026}
}
```
