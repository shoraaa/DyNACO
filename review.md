I'll answer as a world-famous KDD rebuttal strategist and PhD in learning-guided optimization with years of senior-PC style reviewing experience in routing and meta-heuristics.

**TL;DR**: Across the 4 reviewers, the concerns cluster into 8 themes: SRR necessity, pheromone-stabilization isolation, hyperparameter sensitivity, training/memory efficiency, generalization explanation, scope beyond Euclidean routing, theoretical depth of the guidance transition, and broader applicability to other problems.

Here are the **weaknesses / questions proposed by each reviewer**.

## Reviewer 1

### Weaknesses

* Need a clearer explanation of why some baseline methods run out of memory, especially in Table 1.
* The ablation study validates state-aware representation and trajectory-aware training, but does **not** isolate the necessity of **SRR**.
* Missing comparison against:

  * traditional truncated local search
  * full local search

### Questions / suggestions

* Please explain the reason behind the GPU memory explosion / OOM in baselines.
* What is the impact on final performance if **SRR is replaced** by:

  * truncated local search?
  * full local search?
* Show why SRR is necessary.

---

## Reviewer 2

### Weaknesses

* The contribution of **stabilized pheromone dynamics** is not fully disentangled from **SRR**.
* The pheromone-bound ablation is too limited, since it is only shown on TSP-5K.
* It is unclear how stabilized pheromone bounds affect:

  * CVRP
  * real-world instances

### Questions / suggestions

* How do **perturbation size (M)** and **K-NN size (K)** affect performance?
* Is there a tradeoff between:

  * larger (M/K) for stronger perturbation / broader search
  * higher computational cost?
* Are the best (M) and (K) values scale-invariant?
* What challenges would arise in **multi-scale training** for DyNACO?
* For CVRP, what causes the **scale-dependent polarity reversal** in mean guidance?

  * looser/tighter capacity constraints?
  * change in number of routes?
  * change in feasible solution-space geometry?

---

## Reviewer 3

### Weaknesses

* There is ambiguity about the best choice of:

  * temporal abstraction granularity
  * candidate graph construction parameters
    when adapting to other domains.
* The paper does not sufficiently explain **why a 1K-trained policy generalizes so strongly** to much larger real-world instances.
* The experimental scope is mainly limited to **Euclidean 2D routing problems**.
* The enhancement-to-suppression transition is qualitatively interesting, but lacks a stronger **theoretical derivation** for when the crossover should happen.

### Questions / suggestions

* Was **adaptive temporal granularity** considered for macro-action duration?
* What specific structural properties of the **normalized state-aware features** enable such strong cross-scale generalization?
* How much does the observed suppressive anti-stagnation behavior depend on the use of **fixed stabilized pheromone bounds** rather than adaptive bounds?

---

## Reviewer 4

### Weaknesses

* Novelty is viewed as only moderate; the reviewer sees “static heatmap → dynamic policy with semi-MDP” as not highly novel.
* No major technical flaws, but they raise several practical concerns.

### Questions / suggestions

* Will DyNACO **reduce training efficiency**?
* Does the network increase:

  * number of parameters?
  * memory usage?
* Can the framework be applied to other ACO problems such as:

  * OP
  * KP
  * BPP?

---

## Aggregated list of all reviewer concerns

### 1. SRR necessity and isolation

* Need a direct SRR ablation.
* Compare SRR against truncated LS and full LS.
* Clarify SRR’s role versus other design choices.

### 2. Stabilized pheromone dynamics

* Need broader ablation beyond TSP-5K.
* Separate its effect from SRR.
* Clarify whether anti-stagnation behavior depends on fixed bounds.

### 3. Hyperparameter sensitivity / robustness

* Sensitivity to:

  * perturbation size (M)
  * K-NN size (K)
  * temporal abstraction granularity (S) / macro duration
* Whether these settings are robust across scales/domains.

### 4. Efficiency and scalability

* Explain baseline OOM clearly.
* Clarify DyNACO’s:

  * training efficiency
  * parameter count
  * memory usage

### 5. Cross-scale / cross-distribution generalization

* Better explain why training on uniform synthetic 1K instances transfers to:

  * much larger scales
  * real-world distributions

### 6. CVRP-specific behavior

* Explain polarity reversal of mean guidance across scales.
* Clarify whether this is driven by:

  * capacities
  * route count
  * feasibility geometry

### 7. Theory / mechanism depth

* Stronger theoretical explanation for the enhancement → suppression crossover timing.
* More precise causal account of learned anti-stagnation behavior.

### 8. Scope and applicability

* Current scope is mostly Euclidean 2D routing.
* Broader applicability to:

  * non-Euclidean / higher-dimensional problems
  * OP / KP / BPP
    is not yet demonstrated.

If you want, I can next turn this into a **deduplicated rebuttal checklist** ranked by priority: “must answer in experiments,” “answer in text only,” and “move to future work.”

