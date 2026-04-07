# PLAN

## Goal

Produce the smallest set of experiments, implementation changes, and paper updates needed to address reviewer concerns about the experimental results in `paper.tex`, while preserving a clear execution order and avoiding scattered one-off runs.

This plan is organized by:

- `Phase`: execution order and dependency.
- `Relevance`: how directly the work addresses reviewer decisions.
- `Outcome`: what artifact should exist after the phase completes.

## Reviewer Concern Map

### Highest-pressure concerns

- `C1` SRR necessity and isolation.
- `C2` Stabilized pheromone dynamics need clearer isolation and broader coverage.
- `C3` Hyperparameter sensitivity for `M`, `K`, and temporal abstraction `(H, S)`.
- `C4` Efficiency, memory, and OOM explanation.

### Medium-pressure concerns

- `C5` Cross-scale and cross-distribution generalization explanation.
- `C6` CVRP-specific behavior, especially polarity reversal across scales.
- `C7` Mechanistic and theoretical explanation of enhancement-to-suppression behavior.

### Lower-pressure but important scope concerns

- `C8` Applicability beyond Euclidean routing and beyond TSP/CVRP.

## Phase 0: Reproducibility Baseline

**Relevance:** Critical. Everything later depends on trustworthy and repeatable runs.

### Objectives

- Freeze the exact training and evaluation commands that generate the current paper tables and figures.
- Standardize output locations and metric schemas.
- Ensure every experiment records seed, wall-clock time, GPU memory, and config.

### Implementation

- Create a unified experiment driver for train/eval/ablation sweeps.
- Add structured output files for:
  - config
  - aggregate metrics
  - per-seed metrics
  - runtime breakdown
  - memory usage
- Add a manifest file that maps paper tables/figures to command lines.
- Add deterministic naming for runs so comparisons can be joined automatically.
- Enable W\&B logging for all training runs that are meant to compare learning behavior.
  Training-curve comparisons should not rely only on local logs.

### Deliverables

- `experiments/manifest.md` or JSON equivalent.
- `scripts/` or `tune.py` updates for reproducible sweeps.
- One baseline result bundle reproducing the current main results.

### Success Criteria

- One command per table/figure family.
- Every future run can be compared against the same baseline metrics.

## Phase 1: Must-Have Rebuttal Experiments

**Relevance:** Highest. These directly answer the most dangerous reviewer objections.

### 1.1 SRR necessity and isolation

**Addresses:** `C1`

#### Questions to answer

- Is SRR/localized LS necessary, or just helpful?
- Is the gain due to SRR itself or confounded with perturbation-based search and policy learning?
- How does SRR compare against truncated LS and full LS?
- Is SRR’s main benefit better final inference quality, or more stable and learnable training dynamics?

#### Core framing

- State this explicitly in the study design:
  - `SRR/localized LS is hypothesized to matter mainly because it stabilizes policy-gradient training by preserving action-outcome fidelity.`
- Therefore the ablation must compare:
  - `training behavior`
  - and `final inference behavior`
- Do not evaluate SRR only by final test-time gap.
  A stronger full/global LS may still outperform SRR at inference after training, while SRR can remain the better design for learning because it yields lower variance, higher edge survival, and more reliable optimization.
- Because this is a training-stability claim, all LS-comparison training runs should be logged to W\&B and grouped consistently so learning curves can be compared directly across seeds and modes.

#### Experiment set

- Compare:
  - `No LS`
  - `Truncated LS`
  - `Global/full 2-opt`
  - `Localized LS / SRR`
- Primary study should run on small scales first, because LS-scope ablations are expensive and this comparison is about learning stability and action-outcome fidelity, not large-scale leaderboard numbers.
- Run first on:
  - TSP-100
  - TSP-200
  - TSP-500
  - CVRP-100
  - CVRP-200
  - CVRP-500
- Only after a clear trend is established, run a lightweight confirmation subset on:
  - TSP-1K
  - CVRP-1K
- Do not start with 5K or larger for LS-mode ablations unless small-scale evidence is already complete and runtime is acceptable.
- Record:
  - training curves
  - convergence speed
  - variance across seeds
  - training collapse / instability count
  - edge survival during training
  - gradient variance / KL instability
  - final gap
  - final inference gap under the trained policy
  - edge survival
  - train instability events
  - runtime impact

#### Fixed defaults for LS-comparison runs

- Unless a run is explicitly part of a sensitivity study, keep solver hyperparameters fixed at:
  - `K = 32`
  - `M = 12`
- Do not retune hyperparameters separately for each LS mode during the first comparison pass.
  The goal is a controlled comparison of LS behavior, not per-mode best-case tuning.
- Hyperparameter sweeps for `K` and `M` belong to the later robustness phase, not the initial SRR-isolation phase.

#### Two-level evaluation protocol

- For each LS mode, report two result families:
  - `Training results`
    - learning curves
    - variance across seeds
    - stability metrics
    - best validation gap during training
  - `Inference/final results`
    - final evaluation gap
    - time-quality tradeoff if relevant
- Inference comparisons must be runtime-matched, because SRR/localized LS is partly justified by efficiency.
- Do not compare inference quality at a fixed iteration count when LS modes have different per-iteration costs.
- Use the same number of ants across compared methods, and stop search when the wall-clock time budget is exhausted.
- In other words, vary the number of completed ACO iterations under a fixed time limit rather than forcing equal iteration counts.
- The paper discussion must explicitly allow the following outcome:
  - `full/global LS may be stronger as a standalone inference heuristic`
  - while `SRR/localized LS is better for training neural control`
- If that happens, present it as evidence for the paper’s actual claim rather than as a contradiction.

#### Runtime-matched inference protocol

- Fix for each comparison:
  - same instance set
  - same seed set
  - same ant count
  - same hardware
  - same outer/controller settings when neural guidance is used
- For each method:
  - run until a per-instance or per-batch wall-clock budget is reached
  - keep the incumbent best solution found before timeout
- Report:
  - final gap at timeout
  - completed inner/outer iterations at timeout
  - average time spent in:
    - sampling
    - LS
    - pheromone update
    - neural forward if applicable
- If useful, also include a small time-quality curve, but the primary comparison should be at matched runtime budgets.

#### LS-factorization requirement

- The LS comparison must separate:
  - `scope`: `localized` vs `global`
  - `budget`: `truncated` vs `full`
- Define `normal LS` in the codebase and experiments as the `global` scope variant.
- Recommended comparison grid:
  - `No LS`
  - `Localized + truncated`
  - `Localized + full`
  - `Global + truncated`
  - `Global + full`
- If compute is too limited, use this fallback order:
  1. `No LS`
  2. `Localized + truncated`
  3. `Global + truncated`
  4. `Global + full`

#### Truncated-LS definition

- Define `truncated` LS explicitly and consistently across runs.
- The default truncated budget for LS-comparison experiments should be:
  - `max_opt = N / 4`
- Here `N` means:
  - TSP: number of nodes
  - CVRP: number of customers in the giant-tour representation
- Use integer rounding consistently, preferably:
  - `max_opt = floor(N / 4)` with a minimum of `1`
- This budget should apply to both localized and global LS when `ls_budget = truncated`.
- If the implementation uses pass-based logic internally, expose an equivalent operator-count cap so the experimental definition remains unambiguous.

#### Runtime-control rules

- Use fewer seeds for exploratory LS runs:
  - `3` seeds for the first pass
  - `5` seeds only for the final table once the configuration is stable
- Cap training duration for LS ablations:
  - fewer epochs than the headline experiments if trends appear early
  - fixed wall-clock budget per setting
- Prefer collecting dense diagnostics on small `N` rather than sparse diagnostics on large `N`.
- All training runs intended for curve comparison should:
  - use W\&B
  - include a shared group/job-type naming scheme
  - log LS mode in the config (`ls_scope`, `ls_budget`, and truncated budget)

#### Implementation needs

- Expose LS mode as a clean config axis instead of one-off flags.
- Log edge survival and gradient statistics consistently in train and eval.
- Implement LS as two independent axes:
  - `ls_scope in {none, localized, global}`
  - `ls_budget in {truncated, full}`
- If “truncated LS” does not exist cleanly, implement a bounded version of 2-opt via:
  - `max_opt = N/4` as the primary experimental definition
  - then map that to:
    - `max_changes`
    - or `max_passes`
    - or `max_nodes_scanned`
    in the implementation
- Keep the first implementation simple:
  - `localized + truncated`: current checklist-based mode with explicit cap
  - `localized + full`: checklist-based until no improvement
  - `global + truncated`: same cap but scanning all nodes
  - `global + full`: run to local optimum
- Add `normal LS` to both TSP and CVRP using the same local-search operators already used by the SRR/localized path, but changing only the scope:
  - `TSP normal LS`: use the same 2-opt move logic as `two_opt_nn(...)`, but scan globally instead of only over the checklist induced by changed nodes
  - `CVRP normal LS`: use the same giant-tour 2-opt style operators as the current localized CVRP LS, but apply them globally over the whole permutation instead of restricting to the changed set
- For CVRP specifically, do not introduce a separate HGS-style route local search in this phase.
  The first “normal LS” implementation should preserve operator parity with SRR:
  - same move family
  - same giant-tour representation
  - same post-LS Split evaluation
  - only the search scope changes from local checklist to global scan

#### Concrete implementation target

- TSP:
  - keep current checklist-restricted `two_opt_nn(...)` as `localized`
  - add a global counterpart that iterates over all nodes as candidate anchors
- CVRP:
  - keep current checklist-restricted giant-tour `two_opt_nn(...)` as `localized`
  - add a global counterpart over all customers in the permutation
- The code should make it impossible to conflate:
  - `SRR/localized LS`
  - `normal/global LS`
  - `no LS`
- Add explicit config/log fields for:
  - `ls_scope`
  - `ls_budget`
  - `ls_max_opt`
- Ensure W\&B run names or groups encode the LS mode cleanly enough that training curves can be overlaid without manual bookkeeping.

#### Expected outputs

- Main table in paper:
  - small-scale averaged results over `N in {100, 200, 500}`
- Main figure in paper:
  - training curves for at least `Localized + truncated` vs `Global + full`
- Appendix figure/table:
  - per-seed stability and variance metrics
- Appendix table:
  - per-scale breakdown
- Optional confirmation line:
  - one 1K result for localized vs global if budget permits

#### Paper outputs

- Expand the localized-vs-global LS ablation into a fuller SRR necessity table.
- Add one short mechanism paragraph connecting survival to policy credit fidelity.

### 1.2 Stabilized pheromone bounds: broader isolation

**Addresses:** `C2`

#### Questions to answer

- Do fixed bounds help beyond TSP-5K?
- Are the gains due to stabilized pheromones themselves, or entangled with SRR and other design choices?
- Does the anti-stagnation behavior persist on CVRP and benchmark instances?

#### Experiment set

- Compare:
  - standard MMAS bounds
  - fixed stabilized bounds
  - optionally adaptive-but-normalized bounds if cheap to add
- Run on:
  - TSP-1K
  - TSP-5K
  - CVRP-1K
  - TSPLIB / CVRPLIB subset
- Record:
  - final gap
  - training loss variance
  - pheromone CV / entropy
  - instability frequency across seeds
  - runtime overhead

#### Implementation needs

- Add bound-strategy switch as a first-class solver config.
- Log pheromone summary statistics at outer steps.
- Add a compact stagnation proxy metric if not already available.

#### Paper outputs

- Replace the single narrow pheromone ablation with a multi-domain table.
- Add one figure showing learning stability with and without stabilization.

### 1.3 Efficiency, memory, and OOM accounting

**Addresses:** `C4`

#### Questions to answer

- Why do baselines or alternatives OOM?
- What is LGA’s real training/inference overhead?
- What are parameter count and memory scaling at larger `N`?

#### Experiment set

- Profile:
  - peak GPU memory
  - host memory if relevant
  - runtime decomposition
  - model parameter count
  - activation/snapshot memory estimate
- Compare:
  - vanilla backend
  - open-loop neural guidance
  - closed-loop LGA
- Evaluate at:
  - 1K
  - 5K
  - 10K
  - largest feasible scale

#### Implementation needs

- Add peak-memory logging around:
  - GNN forward
  - PPO snapshot storage
  - solver sampling
- Add a utility that computes memory analytically from `N, K, H, S`.
- Capture failures cleanly when a method exceeds memory.

#### Paper outputs

- Strengthen the profiling table.
- Add a short explicit OOM explanation in the experimental section.
- Add a parameter-count sentence for the network.

## Phase 2: Robustness and Generalization

**Relevance:** High. These concerns may not flip acceptance alone, but they materially improve the rebuttal.

### 2.1 Hyperparameter sensitivity and scale robustness

**Addresses:** `C3`

#### Questions to answer

- How sensitive is performance to `M`, `K`, `H`, and `S`?
- Are the defaults robust across scales and across TSP/CVRP?
- Is there a clear quality-time tradeoff?

#### Experiment set

- One-factor sweeps for:
  - `K in {8,16,32,64}`
  - `M in {4,8,12,16,24}`
  - `H,S` under fixed `T = H*S`
- Run on:
  - TSP-1K
  - TSP-5K
  - CVRP-1K
- Record:
  - gap
  - time per iteration
  - total runtime
  - training stability
  - neural improvement over vanilla

#### Implementation needs

- Central sweep runner with Cartesian-product configs.
- Plot generation for robustness curves.

#### Paper outputs

- Replace placeholder candidate-size and refresh ablations with complete sweeps.
- Add a brief statement on whether defaults are reused across scales.

### 2.2 Cross-scale and distribution-shift generalization

**Addresses:** `C5`

#### Questions to answer

- Why does 1K-trained guidance transfer to larger scales?
- Does transfer hold under clustered, mixed, and benchmark distributions?

#### Experiment set

- Train at:
  - TSP-1K
  - optionally CVRP-1K
- Test at:
  - 5K
  - 10K
  - 50K if feasible
  - clustered
  - mixed
  - TSPLIB / CVRPLIB subset
- Record:
  - zero-shot gap
  - degradation relative to in-scale training
  - feature-statistics shift between train and test

#### Implementation needs

- Add data generators for clustered and mixed distributions if not already present.
- Add feature-statistics dumps for dynamic inputs:
  - relative pheromone
  - pheromone CV
  - incumbent-edge indicators

#### Paper outputs

- Populate the distribution-shift and size-generalization tables.
- Add a short explanation grounded in normalized local features and candidate-graph locality.

## Phase 3: Mechanism and CVRP-Specific Analysis

**Relevance:** Medium-high. Best used to turn partial answers into convincing ones.

### 3.1 CVRP polarity reversal analysis

**Addresses:** `C6`

#### Questions to answer

- Why does mean guidance behavior change with scale in CVRP?
- Is the cause route-count growth, capacity tightness, or feasible-geometry changes?

#### Experiment set

- Controlled CVRP sweeps varying:
  - capacity `Q`
  - demand scale/distribution
  - number of customers
- Record:
  - route count
  - average load ratio
  - guidance mean/sign statistics
  - improvement over vanilla

#### Implementation needs

- Add CVRP dataset variants with controlled capacity regimes.
- Log route-structure statistics from decoded solutions.

#### Paper outputs

- One focused CVRP analysis figure/table.
- A concise explanation in the CVRP subsection or appendix.

### 3.2 Anti-stagnation / enhancement-to-suppression mechanism

**Addresses:** `C7`

#### Questions to answer

- When does the controller help intensify search versus suppress overconfident pheromones?
- Is this behavior linked to pheromone concentration, incumbent stability, or stagnation proxies?

#### Experiment set

- Track over outer steps:
  - pheromone CV
  - top-edge overlap between prior and pheromone
  - prior-pheromone correlation
  - incumbent improvement frequency
  - entropy / effective branching factor
- Compare with and without stabilized bounds.

#### Implementation needs

- Consolidate these metrics into a single logging path.
- Add analysis notebooks/scripts for trajectory plots over time.

#### Paper outputs

- One mechanism figure plus text interpreting the crossover.
- If theory remains weak, frame it as empirical regularity rather than formal derivation.

## Phase 4: Scope Extension and Future-Work Positioning

**Relevance:** Lower for immediate acceptance, but important for completeness.

### 4.1 Applicability beyond Euclidean routing

**Addresses:** `C8`

#### Options

- Minimal path:
  - add explicit discussion only
  - narrow claims in the paper
- Stronger path:
  - implement one lightweight non-standard setting:
    - non-uniform distributions
    - asymmetric costs if backend supports it
    - one related graph-routing variant

### 4.2 Other ACO problems

**Addresses:** `C8`

#### Practical approach

- Do not overcommit to OP/KP/BPP experimentally unless the backend can be adapted quickly.
- Instead:
  - identify what parts are generic:
    - closed-loop state encoding
    - sample-and-hold control
    - residual-logit shaping
  - identify what parts are problem-specific:
    - perturbation operator
    - feasibility decoder
    - local search

#### Paper outputs

- A scoped future-work paragraph.
- Stronger separation between “general framework” and “demonstrated domains”.

## Phase 5: Paper Integration

**Relevance:** Critical. Experimental work only matters if it lands cleanly in the paper.

### Required writing tasks

- Fill all currently placeholder tables with actual numbers.
- Ensure each reviewer concern maps to:
  - one experiment/table/figure
  - one short interpretation paragraph
- Tighten claims where evidence stays partial.
- Add a short limitations paragraph where scope remains narrow.

### Section-by-section targets in `paper.tex`

- `Ablation Studies`
  - expand SRR/localized-LS ablation
  - broaden pheromone-bound ablation
  - complete refresh and `K` sensitivity tables
- `Runtime and Scalability`
  - add parameter count, memory, and OOM explanation
- `Robustness and Generalization`
  - populate distribution-shift and size-generalization tables
- `CVRP`
  - add polarity-reversal explanation if evidence is strong enough
- `Discussion/Conclusion`
  - clarify demonstrated scope vs broader applicability

## Priority Matrix

### Phase A: Must do before rebuttal freeze

- Phase 0 baseline reproducibility
- Phase 1.1 SRR isolation
- Phase 1.2 pheromone stabilization broadening
- Phase 1.3 efficiency and memory accounting

### Phase B: Strongly recommended

- Phase 2.1 hyperparameter robustness
- Phase 2.2 cross-scale and distribution-shift generalization
- Phase 3.1 CVRP polarity analysis

### Phase C: Nice to have if time remains

- Phase 3.2 deeper anti-stagnation mechanism analysis
- Phase 4 scope-extension experiments

## Suggested Execution Order

1. Build reproducible run harness and logging.
2. Reproduce current headline numbers.
3. Run SRR/localized-LS isolation study.
4. Run broader pheromone-bound study.
5. Run profiling and memory accounting.
6. Run `K`, `M`, and `(H,S)` sweeps.
7. Run size-generalization and distribution-shift evaluations.
8. Run targeted CVRP diagnostic sweeps.
9. Update paper tables/figures.
10. Rewrite claims to match evidence.

## Minimal Acceptable Package

If time becomes constrained, the minimum package that should still materially improve the paper is:

- A clean LS ablation on small scales (`100/200/500`) covering at least `No LS`, `Localized + truncated`, `Global + truncated`, and ideally `Global + full`.
- A broader stabilized-vs-standard pheromone study on both TSP and CVRP.
- A profiling/memory table with explicit OOM explanation.
- One robustness figure covering `K`, `M`, and refresh frequency.
- One filled size-generalization or distribution-shift table.

## Working Rules

- Every experiment must run with at least 3 seeds; 5 seeds for main tables if affordable.
- Every comparison must use matched wall-clock budgets unless a figure explicitly studies time-quality tradeoff.
- No new claim should be added to the paper without a corresponding table, figure, or tightly scoped caveat.
- Prefer adding targeted experiments that isolate a mechanism over broad but shallow benchmark expansion.
