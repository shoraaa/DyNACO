# DyNACO: Beyond Static Priors: Dynamic Neural Guidance for Large-Scale Ant Colony Optimization

[![Conference](https://img.shields.io/badge/KDD-2026-blue)](https://kdd.org) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

This is the official anonymous repository for the KDD 2026 submission: **"Beyond Static Priors: Dynamic Neural Guidance for Large-Scale Ant Colony Optimization."**

## 📖 Overview

Neural-guided Ant Colony Optimization (ACO) currently suffers from a fundamental training-inference misalignment: policies are typically trained to generate *static* priors (e.g., heatmaps) from instance geometry only once, yet are deployed to guide iterative, long-horizon search processes where pheromones dynamically evolve. 

**DyNACO** shifts from static to **dynamic neural guidance**. Formulated as a semi-Markov Decision Process (semi-MDP), DyNACO employs a state-aware meta-policy that periodically observes the evolving pheromone distribution and incumbent solution. To make this tractable at massive scales, we pair the policy with a perturbation-based ACO backend and a **Scope-Restricted Refinement (SRR)** mechanism.

### ✨ Key Features
* **Dynamic Guidance:** The neural policy adapts its strategy based on the current search phase, actively counteracting ACO stagnation by suppressing over-reinforced edges.
* **Scope-Restricted Refinement:** Confines local search to a perturbation neighborhood, preserving gradient fidelity and achieving 2-opt optimality at $O(M \cdot K)$ cost rather than $O(N^2)$.
* **Extreme Scalability:** Scales efficiently up to **100,000 nodes** on a single GPU.
* **Lightning Fast Training:** The ~50K parameter policy trains in just ~30 minutes for TSP-1K on a single RTX 5090.

---

## 🚀 Experimental Highlights

Extensive evaluations across synthetic and real-world instances demonstrate that DyNACO achieves state-of-the-art performance among learning-guided solvers. 

### 1. Superior Scalability and Quality (Up to 100K Nodes)
DyNACO surpasses all existing neural baselines across all scales. Remarkably, on TSP, DyNACO **reduces total runtime by 20–33%** compared to the unguided solver because better-targeted neural perturbations lead to faster local-search convergence.

| Problem | Method | Gap to Reference | Total Time | 
| :--- | :--- | :--- | :--- | 
| **TSP-10K** | Unguided ACO | 2.02% | 26.81s |
| | **DyNACO (Ours)** | **0.82%** | **17.89s** |
| **TSP-100K** | Unguided ACO | 3.12% | 223.32s |
| | **DyNACO (Ours)** | **1.90%** | **171.69s** |

*(Results based on $I_{10000}$ iteration budget. Reference: LKH-3)*

### 2. Zero-Shot Generalization to Real-World Benchmarks
Models trained *only* on uniform synthetic 1K instances generalize zero-shot to non-uniform, real-world topologies and much larger scales (up to $86\times$ larger than training), outperforming classical and neural baselines.

| Benchmark Dataset | Opt/BKS Gap | Win Rate vs. ACO | Highlights |
| :--- | :--- | :--- | :--- |
| **TSPLIB** (33 instances, 1K–86K) | **0.89%** | 29 / 33 | 31% relative improvement over unguided ACO. |
| **CVRPlib** (14 instances, 1K–30K) | **3.66%** | 14 / 14 | Outperforms the classical state-of-the-art (HGS). |

### 3. Cross-Problem Adaptability (CVRP)
Dynamic neural guidance transfers directly to the Capacitated Vehicle Routing Problem (CVRP) without any architectural modification. It consistently improves upon the unguided baseline at every iteration budget, adding **<1% computational overhead** at large scales.

---

## ⚙️ Installation

**Prerequisites**
* Linux (tested on Ubuntu)
* Python ≥ 3.13
* CUDA-capable GPU
* C++17 compiler with OpenMP support
* `uv` package manager

**Setup Environment & Build Backend**
```bash
# Clone the repository
git clone https://anonymous.4open.science/r/DyNACO/
cd DyNACO

# Create environment and install dependencies
uv sync

# Build the C++ backend (perturbation-based ACO + SRR)
cd src
uv run python setup.py build_ext --inplace
cd ..
```

**Verify Installation**
```bash
uv run python -c "import faco_opt; print('C++ backend OK')"
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
```

---

## 🏃 Usage

### Training

Train DyNACO on TSP-1K (default configuration, ~30 min on an RTX 5090):
```bash
uv run python train.py --problem tsp --n_node 1000
```

Train on CVRP-1K:
```bash
uv run python train.py --problem cvrp --n_node 1000
```

Scale to larger instances:
```bash
# TSP-10K (~2 hours)
uv run python train.py --problem tsp --n_node 10000

# TSP-100K (~4 hours)
uv run python train.py --problem tsp --n_node 100000
```

**Key Training Arguments:**

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--problem` | *(required)* | `tsp` or `cvrp` |
| `--n_node` | `1000` | Problem size (number of nodes) |
| `--k_sparse` | `32` | K-NN candidate graph size |
| `--n_ants` | `100` | Number of ants |
| `--H` | `10` | Outer steps (guidance updates) |
| `--mini_H` | `100` | Inner steps per guidance update |
| `--epochs` | `10` | Training epochs |
| `--algo` | `ppo` | `ppo` or `reinforce` |
| `--rho` | `0.5` | Pheromone evaporation rate |
| `--lr` / `--ppo_lr` | `5e-6` | Learning rate |
| `--device` | `cuda:0` | Compute device |
| `--save_dir` | `pretrained`| Directory to save checkpoints |

### Evaluation

Evaluate a trained checkpoint:

```bash
# TSP-1K with 1K iterations
uv run python test.py \
    --problem tsp --n_node 1000 \
    --checkpoint pretrained/tsp/n1000/best.pt \
    --H 10 --mini_H 100

# CVRP-1K
uv run python test.py \
    --problem cvrp --n_node 1000 \
    --checkpoint pretrained/cvrp/n1000/best.pt \
    --H 10 --mini_H 100
```

Evaluate on **TSPLIB/CVRPlib** real-world instances:
```bash
uv run python test.py \
    --problem tsp \
    --checkpoint pretrained/tsp/n1000/best.pt \
    --rl_data --H 10 --mini_H 100
```

Run **unguided ACO baseline** (no neural guidance):
```bash
uv run python test.py --problem tsp --n_node 1000 --no_model
```

**Key Evaluation Arguments:**

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--checkpoint` | `none` | Path to trained model weights |
| `--H` | `10` | Outer steps (inference iterations) |
| `--mini_H` | `100` | Inner steps per outer step |
| `--rl_data` | `false` | Evaluate on TSPLIB/CVRPlib real-world instances |
| `--dataset` | `none` | Custom dataset path |
| `--no_model` | `false` | Run unguided ACO baseline only |
| `--timed` | `false` | Enable detailed CPU/GPU timing breakdowns |
| `--warmup` | `true` | Apply Phased Injection strategy |
| `--no_anneal` | `false` | Disable guidance annealing |

---

## 📦 Pretrained Checkpoints

Pretrained models are available in the `pretrained/` directory. 
*💡 Note: A model trained on 1K instances transfers zero-shot to larger scales (5K, 10K, 100K) and real-world datasets with minimal degradation.*

---

## 📁 Project Structure

```text
├── train.py              # Training script (PPO with trajectory replay)
├── test.py               # Evaluation and benchmarking script
├── net.py                # GNN encoder + MLP decoder (~50K params)
├── faco.py               # ACO environment wrapper and state formulation
├── utils.py              # Utilities, metrics, and analysis tools
├── src/                  # C++ Backend
│   ├── mfaco_train.cpp   # Perturbation-based ACO + SRR mechanics
│   ├── binding.cpp       # pybind11 integration
│   └── setup.py          # C++ extension build script
├── data/                 # Benchmark datasets (Synthetic, TSPLIB, CVRPlib)
└── pretrained/           # Pretrained checkpoints
```

---

## 📑 Citation
*(Placeholder - Currently under anonymous review for KDD 2026)*
```bibtex
@inproceedings{dynaco2026,
  title={Beyond Static Priors: Dynamic Neural Guidance for Large-Scale Ant Colony Optimization},
  author={Anonymous Authors},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```
