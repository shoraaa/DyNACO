# DyNACO: Beyond Static Priors — Dynamic Neural Guidance for Large-Scale Ant Colony Optimization

> **KDD '25** | [Paper (coming soon)](#) | [Anonymous Code](https://anonymous.4open.science/r/DyNACO/)

## Overview

Neural-guided Ant Colony Optimization (ACO) suffers from a fundamental **training–inference misalignment**: policies are trained to produce static heatmaps from instance geometry alone, yet deployed inside iterative search loops where pheromone fields evolve over hundreds of iterations. The model never observes search progress and cannot adapt.

**DyNACO** shifts from *static* to *dynamic* neural guidance. Formulated as a semi-MDP, a lightweight meta-policy (~50K parameters) periodically observes the evolving pheromone distribution and incumbent solution, emitting updated edge-level guidance throughout the search trajectory. Two complementary mechanisms enable this:

- **State-Aware Representation** — conditions the policy on per-edge pheromone statistics and incumbent topology, enabling search-phase-dependent guidance.
- **Trajectory-Aware Training** — optimizes expected cost across the full search history via PPO, aligning training with iterative search dynamics.

To scale to 100K nodes, DyNACO pairs the policy with a perturbation-based ACO backend and **Scope-Restricted Refinement (SRR)**, which preserves gradient fidelity at $O(M \cdot K)$ cost per ant, independent of instance size.

### Key Results

| | TSP-1K | TSP-10K | TSP-100K | CVRP-1K | CVRP-100K |
|---|---|---|---|---|---|
| **Gap** | 0.20% | 0.82% | 1.90% | 1.04% | 6.75% |
| **vs. ACO** | −58% | −59% | −39% | −44% | −7% |

- Outperforms all neural baselines across all scales (1K–100K)
- Reduces runtime by 20–33% on TSP compared to unguided ACO
- <1% overhead on CVRP
- Zero-shot transfer from 1K training to 86K-node TSPLIB instances

## Installation

### Prerequisites

- Linux (tested on Ubuntu)
- Python ≥ 3.13
- CUDA-capable GPU
- C++17 compiler with OpenMP support
- [`uv`](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Create environment and install dependencies
uv sync

# Build the C++ backend (perturbation-based ACO + SRR)
cd src
uv run python setup.py build_ext --inplace
cd ..
```

### Verify installation

```bash
uv run python -c "import faco_opt; print('C++ backend OK')"
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
```

## Usage

### Training

Train DyNACO on TSP-1K (default configuration, ~30 min on RTX 5090):

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

Key training arguments:

| Argument | Default | Description |
|---|---|---|
| `--problem` | (required) | `tsp` or `cvrp` |
| `--n_node` | 1000 | Problem size |
| `--k_sparse` | 32 | K-NN candidate graph size |
| `--n_ants` | 100 | Number of ants |
| `--H` | 10 | Outer steps (guidance updates) |
| `--mini_H` | 100 | Inner steps per guidance update |
| `--epochs` | 10 | Training epochs |
| `--algo` | ppo | `ppo` or `reinforce` |
| `--rho` | 0.5 | Pheromone evaporation rate |
| `--lr` / `--ppo_lr` | 5e-6 | Learning rate |
| `--device` | cuda:0 | Device |
| `--save_dir` | pretrained | Checkpoint directory |

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

Evaluate on TSPLIB/CVRPlib real-world instances:

```bash
uv run python test.py \
    --problem tsp \
    --checkpoint pretrained/tsp/n1000/best.pt \
    --rl_data --H 10 --mini_H 100
```

Run unguided ACO baseline (no neural guidance):

```bash
uv run python test.py --problem tsp --n_node 1000 --no_model
```

Key evaluation arguments:

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | none | Path to trained model |
| `--H` | 10 | Outer steps |
| `--mini_H` | 100 | Inner steps per outer step |
| `--rl_data` | false | Use TSPLIB/CVRPlib instances |
| `--dataset` | none | Custom dataset path |
| `--no_model` | false | Run unguided ACO only |
| `--timed` | false | Enable detailed timing |
| `--warmup` | true | Phased injection |
| `--no_anneal` | false | Disable guidance annealing |

### Pretrained Checkpoints

Pretrained models are available in `checkpoints/`.

The 1K-trained model transfers zero-shot to larger scales with minimal degradation.

## Project Structure

```
├── train.py          # Training script (PPO with trajectory replay)
├── test.py           # Evaluation script
├── net.py            # GNN encoder + MLP decoder (~50K params)
├── faco.py           # ACO environment wrapper
├── utils.py          # Utilities, metrics, analysis tools
├── src/
│   ├── mfaco_train.cpp   # C++ perturbation-based ACO backend
│   ├── binding.cpp       # pybind11 bindings
│   └── setup.py          # C++ extension build script
├── data/             # Benchmark datasets
├── pretrained/       # Trained checkpoints
```
The code was build partly on top of [DeepACO](github.com/henry-yeh/DeepACO) structures.
