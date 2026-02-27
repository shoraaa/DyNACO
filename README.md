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
Most end-to-end NCO methods face severe Out-Of-Memory (OOM) limitations beyond 10K nodes due to quadratic attention complexity. Hierarchical/Decomposition methods scale further but incur substantial runtime costs. 

**DyNACO** easily scales to **100,000 nodes** while maintaining sub-linear runtime growth. Remarkably, on TSP, DyNACO **reduces total runtime by 20–33%** compared to the unguided solver because better-targeted neural perturbations lead to faster local-search convergence.

#### 🏆 Comprehensive Benchmark: Traveling Salesman Problem (TSP)
*Metrics reported as: **Gap to Reference (%) / Total Time**. Reference: LKH-3. OOM = Out of Memory.*

| Category | Method | TSP-1K | TSP-10K | TSP-100K |
| :--- | :--- | :--- | :--- | :--- |
| **Classical** | LKH-3 `[1]` | 0.00% / 1.70m | 0.00% / 33.00m | 0.00% / 25.00h |
| **End-to-End NCO** | POMO `[2]` | 40.61% / 4.10s | OOM | OOM |
| | BQ `[3]` | 1.34% / 13.00s | OOM | OOM |
| | SIGD `[4]` | 1.04% / 17.30s | OOM | OOM |
| **Hierarchical NCO** | LEHD (RRC 1000) `[5]` | 0.74% / 3.30m | 12.71% / 18.60m | OOM |
| | L2C-Insert (I 1000) `[6]` | 0.48% / 21.75s | 2.08% / 1.04m | 4.92% / 1.63m |
| | SIL (PRC 1000) `[7]` | 0.38% / 1.50m | 1.81% / 17.00m | 2.45% / 2.60h |
| **Neural-Guided ACO**| DeepACO `[8]` | 2.87% / 1.10m | — | — |
| | GFACS `[9]` | 2.63% / 1.10m | — | — |
| | HeatACO (DIFUSCO) `[10]`| 0.23% / 10.00s | 1.19% / 2.89m | — |
| **Ours** | **ACO** *(Unguided, I 10000)* | 0.48% / 5.44s | 2.02% / 26.81s | 3.12% / 3.72m |
| | **DyNACO** *(I 10000)* | **0.20% / 3.68s** | **0.82% / 17.89s** | **1.90% / 2.86m** |

#### 🏆 Comprehensive Benchmark: Capacitated Vehicle Routing Problem (CVRP)
*Metrics reported as: **Gap to Reference (%) / Total Time**. Reference: HGS. Negative gaps indicate performance better than the baseline reference.*

| Category | Method | CVRP-1K | CVRP-10K | CVRP-100K |
| :--- | :--- | :--- | :--- | :--- |
| **Classical** | HGS `[11]` | 0.00% / 2.50m | 0.00% / 5.00h | 0.00% / 24.00h |
| **End-to-End NCO** | POMO `[2]` | 133.92% / 4.80s | OOM | OOM |
| | BQ `[3]` | 5.18% / 14.00s | OOM | OOM |
| | SIGD `[4]` | 7.88% / 17.30s | 22.42% / 3.97m | OOM |
| **Hierarchical NCO** | LEHD (RRC 1000) `[5]` | 3.14% / 3.40m | 29.17% / 41.00m | OOM |
| | L2C-Insert (I 1000) `[6]` | 5.63% / 43.59s | 26.42% / 1.27m | 45.54% / 2.38m |
| | SIL (PRC 1000) `[7]` | 2.73% / 1.50m | -0.66% / 15.20m | -2.55% / 2.17h |
| **Neural-Guided ACO**| DeepACO `[8]` | 2.40% / 1.30m | — | — |
| **Ours** | **ACO** *(Unguided, I 10000)* | 1.85% / 20.17s | 6.76% / 1.24m | 7.26% / 11.55m |
| | **DyNACO** *(I 10000)* | **1.04% / 19.05s** | **6.04% / 1.28m** | **6.75% / 11.70m** |


### 2. Zero-Shot Generalization to Real-World Benchmarks
Models trained *only* on uniform synthetic 1K instances generalize zero-shot to non-uniform, real-world topologies and much larger scales (up to $86\times$ larger than training). DyNACO solves *all* instances, massively outperforming all NCO baselines on both TSPLIB and CVRPlib, and even outperforming the classical state-of-the-art solver (HGS) on CVRPlib.

*Metrics reported as: **Gap to Reference (%) / Total Time**. All NCO models use their respective zero-shot/transfer evaluation methods.*

| Category | Method | TSPLIB (33 instances, 1K–86K) | CVRPlib (14 instances, 1K–30K) |
| :--- | :--- | :--- | :--- |
| **Classical Solvers** | LKH-3 `[1]` | 0.07% / 35.00m | 13.60% / 2.10h |
| | HGS `[11]` | — | 5.15% / 5.00h |
| **NCO Baselines** | LEHD `[5]` | 13.20% / 38.00m | 16.90% / 40.00m |
| | GLOP `[12]` | 6.99% / 34.00s | 20.60% / 39.00s |
| | SIL `[7]` | 3.03% / 45.00m | 7.69% / 54.00m |
| **Ours** | **DyNACO** *(I 10000)* | **0.89% / 11.57s** | **3.66% / 51.07s** |

* **TSPLIB Highlights**: DyNACO achieves a 31% relative improvement over unguided ACO, winning on 29 out of 33 instances.
* **CVRPlib Highlights**: DyNACO improves upon the unguided solver on every single instance (14/14) with a 14% relative gap reduction, solidly outperforming the highly engineered HGS solver.

### 3. Cross-Problem Adaptability
Dynamic neural guidance transfers directly to CVRP without any architectural modification. It consistently improves upon the unguided baseline at every iteration budget, adding **<1% computational overhead** on GPU/CPU at large scales dataset.

<details>
<summary><b>📚 References for Evaluated Baselines</b> (Click to expand)</summary>

* `[1]` **LKH-3**: *An effective implementation of the Lin-Kernighan traveling salesman heuristic* (Helsgaun, EJOR 2000)
* `[2]` **POMO**: *POMO: policy optimization with multiple optima for reinforcement learning* (Kwon et al., NeurIPS 2020)
* `[3]` **BQ**: *BQ-NCO: bisimulation quotienting for efficient neural combinatorial optimization* (Drakulic et al., NeurIPS 2023)
* `[4]` **SIGD**: *Self-Improvement for Neural Combinatorial Optimization: Sample Without Replacement, but Improvement* (Pirnay & Grimm, TMLR 2024)
* `[5]` **LEHD**: *Neural combinatorial optimization with heavy decoder: toward large scale generalization* (Luo et al., NeurIPS 2023)
* `[6]` **L2C-Insert**: *Learning to Insert for Constructive Neural Vehicle Routing Solver* (Luo et al., NeurIPS 2025)
* `[7]` **SIL**: *Boosting Neural Combinatorial Optimization for Large-Scale Vehicle Routing Problems* (Luo et al., ICLR 2025)
* `[8]` **DeepACO**: *DeepACO: neural-enhanced ant systems for combinatorial optimization* (Ye et al., NeurIPS 2023)
* `[9]` **GFACS**: *Ant Colony Sampling with GFlowNets for Combinatorial Optimization* (Kim et al., AISTATS 2025)
* `[10]` **HeatACO**: *HEATACO: Heatmap-Guided Ant Colony Decoding for Large-Scale Travelling Salesman Problems* (Lin et al., 2026)
* `[11]` **HGS**: *Hybrid genetic search for the CVRP: Open-source implementation and SWAP\* neighborhood* (Vidal, COR 2022)
* `[12]` **GLOP**: *GLOP: learning global partition and local construction for solving large-scale routing problems in real-time* (Ye et al., AAAI 2024)
</details>

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
