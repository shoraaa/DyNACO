/**
 * MFACO Training Module - Unified C++ implementation
 *
 * Combines TSP and CVRP solvers into a single extension.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

namespace mfaco {

// ============================================================================
// Constants
// ============================================================================

static constexpr uint32_t MAX_CAND_LIST_SIZE = 64;
static constexpr float EPS = 1e-12f;
static constexpr float LOG_EPS = 1e-12f;
static constexpr int64_t DEMAND_SCALE = 100000;

// Distance types supported by this MFACO training backend.
enum class DistanceType : uint8_t { EXPLICT_EUC_2D };

// ============================================================================
// Random number generator (xoshiro128+)
// ============================================================================

class Xoshiro128Plus {
public:
  void seed(uint64_t s) {
    // SplitMix64 initialization
    auto splitmix = [](uint64_t &x) {
      x += 0x9e3779b97f4a7c15ULL;
      uint64_t z = x;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
      return z ^ (z >> 31);
    };
    state_[0] = splitmix(s);
    state_[1] = splitmix(s);
  }

  uint32_t next_u32() {
    const uint64_t s0 = state_[0];
    uint64_t s1 = state_[1];
    const uint64_t result = s0 + s1;
    s1 ^= s0;
    state_[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
    state_[1] = rotl(s1, 36);
    return static_cast<uint32_t>(result >> 32);
  }

  // Uniform [0, max_exclusive)
  uint32_t next_uint(uint32_t max_exclusive) {
    uint32_t x = next_u32();
    uint64_t m =
        static_cast<uint64_t>(x) * static_cast<uint64_t>(max_exclusive);
    uint32_t l = static_cast<uint32_t>(m);
    if (l < max_exclusive) {
      uint32_t t = -max_exclusive % max_exclusive;
      while (l < t) {
        x = next_u32();
        m = static_cast<uint64_t>(x) * static_cast<uint64_t>(max_exclusive);
        l = static_cast<uint32_t>(m);
      }
    }
    return static_cast<uint32_t>(m >> 32);
  }

  // Uniform [0, 1)
  float next_float() {
    return static_cast<float>(next_u32() >> 8) * (1.0f / (1u << 24));
  }

private:
  static constexpr uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }
  uint64_t state_[2];
};

// ============================================================================
// Trace structure for log-prob replay
// ============================================================================

struct MFACOTrace {
  int32_t start_node;
  std::vector<int32_t> curr_nodes;   // current node at each decision
  std::vector<int32_t> chosen_nodes; // chosen next node at each decision
  std::vector<uint8_t>
      is_stochastic;           // 1 if decision had multiple valid candidates
  std::vector<int16_t> pick_j; // index in nn_list row (or -1)
  std::vector<uint64_t>
      valid_mask; // bitmask over k candidates at decision time
  std::vector<uint8_t> is_new_edge; // 1 if edge is not in source solution

  void clear() {
    start_node = -1;
    curr_nodes.clear();
    chosen_nodes.clear();
    is_stochastic.clear();
    pick_j.clear();
    valid_mask.clear();
    is_new_edge.clear();
  }

  void reserve(size_t n) {
    curr_nodes.reserve(n);
    chosen_nodes.reserve(n);
    is_stochastic.reserve(n);
    pick_j.reserve(n);
    valid_mask.reserve(n);
    is_new_edge.reserve(n);
  }
};

// ============================================================================
// Batch trace structure (compact storage for all ants)
// ============================================================================

struct MFACOTraceBatch {
  // Prefix sums: decision range for ant a is [starts[a], starts[a+1])
  std::vector<int32_t> starts;        // length n_ants + 1
  std::vector<int32_t> curr_nodes;    // length D (total decisions)
  std::vector<int32_t> chosen_nodes;  // length D
  std::vector<uint8_t> is_stochastic; // length D
  std::vector<int16_t> pick_j;        // length D, index in nn_list row (or -1)
  std::vector<uint64_t>
      valid_mask; // length D, bitmask over k candidates at decision time
  std::vector<uint8_t> is_new_edge; // length D
  std::vector<int32_t> start_nodes; // length n_ants (start node per ant)

  void clear() {
    starts.clear();
    curr_nodes.clear();
    chosen_nodes.clear();
    is_stochastic.clear();
    pick_j.clear();
    valid_mask.clear();
    is_new_edge.clear();
    start_nodes.clear();
  }

  void reserve(int32_t n_ants, int32_t max_decisions) {
    starts.reserve(n_ants + 1);
    curr_nodes.reserve(max_decisions);
    chosen_nodes.reserve(max_decisions);
    is_stochastic.reserve(max_decisions);
    pick_j.reserve(max_decisions);
    valid_mask.reserve(max_decisions);
    is_new_edge.reserve(max_decisions);
    start_nodes.reserve(n_ants);
  }
};

// ============================================================================
// Sample result structure
// ============================================================================

struct SampleResult {
  std::vector<float> costs; // (n_ants,)
  std::vector<std::vector<int32_t>>
      routes;                   // (n_ants, n) - each route without repetition
  MFACOTraceBatch traces;       // only populated if require_prob=true
  std::vector<float> logps;     // (n_ants,) log probability of sample
  std::vector<float> costs_raw; // (n_ants,) cost before LS (TSP only)
  std::vector<std::vector<int32_t>>
      routes_raw; // (n_ants, n) - each route before LS (TSP only)
  std::vector<int32_t> new_edges_count; // (n_ants,)
  std::vector<float>
      edge_survival; // (n_ants,) ratio of sampled edges surviving in final tour
  std::vector<std::vector<int32_t>> decoded_routes;

  void clear() {
    costs.clear();
    routes.clear();
    traces.clear();
    logps.clear();
    costs_raw.clear();
    routes_raw.clear();
    new_edges_count.clear();
    edge_survival.clear();
    decoded_routes.clear();
  }
};

// ============================================================================
// MFACO_TSP class - main solver
// ============================================================================

class MFACO_TSP {
public:
  // Configuration
  int32_t n; // number of nodes
  int32_t n_ants;
  int32_t k;  // candidate list size
  int32_t bl; // backup list size
  int32_t min_new_edges;
  int32_t fixed_steps; // if > 0, fixed number of steps
  float rho;           // pheromone decay (1 - evaporation rate)
  float alpha;         // pheromone exponent
  float p_best;        // for tau limits
  bool smooth_mmas; // if true, use Smooth-MMAS style pheromone update (linear
                    // interpolation)
  bool use_local_search;
  bool extend_ls; // if true, extend local search checklist with endpoints of
                  // improving moves
  bool disable_heuristic; // if true, set eta=1 so sampling uses pheromone (+
                          // prior) only
  bool nls;               // if true, use Neural Local Search (guided by prior)
  int32_t T_nls;          // number of NLS iterations (default 10)

  // Distance type (currently fixed to explicit Euclidean 2D without rounding).
  DistanceType distance_type = DistanceType::EXPLICT_EUC_2D;

  // State arrays
  std::vector<float> coords;        // (n, 2) row-major (x,y)
  std::vector<int32_t> nn_list;     // (n, k) row-major: nearest neighbors
  std::vector<int32_t> backup_list; // (n, bl) row-major: backup neighbors
  std::vector<float>
      pheromone_sparse; // (n, k) row-major: pheromone on candidate edges
  std::vector<float>
      heuristic_sparse;        // (n, k) row-major: 1/dist for candidate edges
  std::vector<int32_t> nn_pos; // REMOVED to save memory

  // Solution state
  std::vector<int32_t> source_route; // (n,) current source solution
  float source_cost;
  std::vector<int32_t> best_route; // (n,) best solution found
  float best_cost;
  float tau_min;
  float tau_max;

  // Source positions cache
  std::vector<int32_t> source_positions; // (n,) source_positions[v] = position
                                         // of v in source_route

  // RNG
  Xoshiro128Plus rng_;

  // ========================================================================
  // Constructor
  // ========================================================================

  MFACO_TSP(const float *coords_ptr, // (n, 2) row-major (x,y)
            int32_t n, int32_t n_ants, int32_t cand_list_size = 32,
            int32_t backup_list_size = 32, int32_t min_new_edges = 8,
            float decay = 0.9f, float alpha = 1.0f, float p_best = 0.05f,
            bool use_local_search = true, bool disable_heuristic = false,
            bool extend_ls = false, bool smooth_mmas = false,
            int32_t fixed_steps = 0, bool nls = false, int32_t T_nls = 10);

  // ========================================================================
  // API Methods
  // ========================================================================

  /**
   * Seed the RNG for reproducibility.
   */
  void seed_rng(uint64_t seed);

  /**
   * Sample solutions from all ants.
   *
   * @param require_prob If true, record traces for log-prob replay.
   * @param parallel_traced If true and require_prob=true, parallelize traced
   * sampling. Traces are generated per-ant and merged after; RNG is per-ant.
   * @param prior Optional (n, k) neural prior. Pass nullptr if not used.
   * @param result Output structure filled with costs, routes, and traces.
   */
  void sample(bool require_prob, const float *prior, SampleResult &result,
              bool parallel_traced = false);

  /**
   * Update pheromone: evaporate + deposit on best route.
   * Updates source solution and tau limits.
   *
   * @param best_flat Route as (n+1,) array where last == first.
   * @param best_cost Cost of the route.
   */
  void update_pheromone(const int32_t *best_flat, float best_cost);

  /**
   * Load state from snapshot arrays. Matches Python load_snapshot().
   */
  void load_snapshot(const float *pheromone_ptr,      // (n, k)
                     const int32_t *source_route_ptr, // (n,)
                     float source_cost,
                     const int32_t *best_route_ptr, // (n,)
                     float best_cost, float tau_min, float tau_max,
                     const int32_t *nn_list_ptr,    // (n, k)
                     const int32_t *backup_list_ptr // (n, bl)
  );

  /**
   * Sync pheromone from external numpy array (for compatibility with Python).
   */
  void set_pheromone(const float *pheromone_ptr);

  /**
   * Get raw pointers to internal arrays for numpy views.
   */
  float *pheromone_data() { return pheromone_sparse.data(); }
  int32_t *nn_list_data() { return nn_list.data(); }
  int32_t *backup_list_data() { return backup_list.data(); }
  float *heuristic_data() { return heuristic_sparse.data(); }
  int32_t *source_route_data() { return source_route.data(); }
  int32_t *best_route_data() { return best_route.data(); }
  // int32_t *nn_pos_data() { return nn_pos.data(); }

  inline int32_t find_neighbor_index(int32_t u, int32_t v) const {
    for (int32_t j = 0; j < k; ++j) {
      if (nn_list[u * k + j] == v)
        return j;
    }
    return -1;
  }

private:
  // ========================================================================
  // Internal methods
  // ========================================================================

  void build_nn_lists();
  // void build_nn_pos();
  void build_heuristic();
  void build_initial_tour();

  inline float dist(int32_t a, int32_t b) const {
    const size_t ia = static_cast<size_t>(a) * 2;
    const size_t ib = static_cast<size_t>(b) * 2;
    const float dx = coords[ib + 0] - coords[ia + 0];
    const float dy = coords[ib + 1] - coords[ia + 1];
    return std::sqrt(dx * dx + dy * dy);
  }

  std::pair<float, float> calc_trail_limits_cl(float solution_cost) const;
  std::pair<float, float> calc_trail_limits_smooth(float solution_cost) const;

  float sample_ant_fast(const float *probmat, // (n, k) precomputed weights
                        int32_t start_node, std::vector<int32_t> &route_out,
                        int32_t &new_edges_out, std::vector<int32_t> &checklist,
                        Xoshiro128Plus &rng, const float *prior);

  float sample_ant_traced(const float *probmat, // (n, k) precomputed weights
                          int32_t start_node, std::vector<int32_t> &route_out,
                          std::vector<int32_t> &route_raw_out,
                          float &cost_raw_out, int32_t &new_edges_out,
                          std::vector<int32_t> &checklist, MFACOTrace &trace,
                          Xoshiro128Plus &rng, float &logp_sum,
                          float &survival_out, const float *prior);

  std::tuple<int32_t, bool, float>
  select_next_node(int32_t curr,
                   const float *probmat_row, // probmat[curr*k : curr*k + k]
                   const uint8_t *visited, Xoshiro128Plus &rng,
                   int16_t &out_pick_j, uint64_t &out_valid_mask);

  float relocate_node(int32_t target, int32_t node, std::vector<int32_t> &route,
                      std::vector<int32_t> &positions);

  float two_opt_nn(std::vector<int32_t> &route, std::vector<int32_t> &positions,
                   std::vector<int32_t> &checklist);

  float two_opt_nn_prior(std::vector<int32_t> &route,
                         std::vector<int32_t> &positions,
                         std::vector<int32_t> &checklist,
                         const float *prior_ptr);

  float get_route_cost(const std::vector<int32_t> &route) const;

  bool contains_edge(int32_t a, int32_t b,
                     const std::vector<int32_t> &positions) const;

  int32_t get_succ(int32_t node, const std::vector<int32_t> &route,
                   const std::vector<int32_t> &positions) const;

  int32_t get_pred(int32_t node, const std::vector<int32_t> &route,
                   const std::vector<int32_t> &positions) const;

  void flip_route_section(int32_t start_node, int32_t end_node,
                          std::vector<int32_t> &route,
                          std::vector<int32_t> &positions);

  void compute_probmat(const float *prior, std::vector<float> &probmat);
};

// ============================================================================
// ACO_TSP class - Standard Constructive MMAS
// ============================================================================

class ACO_TSP {
public:
  // Configuration
  int32_t n;
  int32_t n_ants;
  int32_t k; // candidate list size
  float rho; // pheromone decay
  float alpha;
  float beta;   // heuristic exponent (new for ACO)
  float p_best; // for MMAS limits
  bool min_max; // Use MMAS min-max limits
  bool elitist; // Use elitist update (only best ant deposits) - implied by MMAS
                // usually

  // Distance type
  DistanceType distance_type = DistanceType::EXPLICT_EUC_2D;

  // State arrays
  std::vector<float> coords;    // (n, 2)
  std::vector<int32_t> nn_list; // (n, k)
  std::vector<float> pheromone; // (n, k)
  std::vector<float> heuristic; // (n, k)

  // Solution state
  std::vector<int32_t> best_route; // (n,)
  float best_cost;
  float tau_min;
  float tau_max;

  // RNG
  Xoshiro128Plus rng_;

  // Constructor
  ACO_TSP(const float *coords_ptr, int32_t n, int32_t n_ants,
          int32_t cand_list_size = 32, float decay = 0.9f, float alpha = 1.0f,
          float beta = 1.0f, float p_best = 0.05f, bool min_max = true);

  // API
  void seed_rng(uint64_t seed);

  void sample(bool require_prob, const float *prior, SampleResult &result,
              bool parallel_traced = false);

  void update_pheromone(const int32_t *solution_flat, float cost);

  // Data access
  float *pheromone_data() { return pheromone.data(); }
  int32_t *nn_list_data() { return nn_list.data(); }
  float *heuristic_data() { return heuristic.data(); }
  int32_t *best_route_data() { return best_route.data(); }

private:
  void build_nn_lists();
  void build_heuristic();

  inline float dist(int32_t a, int32_t b) const {
    const size_t ia = static_cast<size_t>(a) * 2;
    const size_t ib = static_cast<size_t>(b) * 2;
    const float dx = coords[ib + 0] - coords[ia + 0];
    const float dy = coords[ib + 1] - coords[ia + 1];
    return std::sqrt(dx * dx + dy * dy);
  }

  float get_route_cost(const std::vector<int32_t> &route) const;

  void compute_probmat(const float *prior, std::vector<float> &probmat);

  // Sample one ant
  float sample_ant_constructive(const float *probmat, int32_t start_node,
                                std::vector<int32_t> &route_out,
                                Xoshiro128Plus &rng, bool require_prob,
                                float &logp_out, MFACOTrace *trace = nullptr);

  std::pair<float, float> calc_trail_limits(float solution_cost) const;
};

class MFACO_CVRP {
public:
  // -------------------------
  // Public hyperparameters
  // -------------------------
  int32_t n; // includes depot at 0
  int32_t m; // customers count = n-1
  int32_t n_ants;
  int32_t k;
  int32_t bl;
  int32_t min_new_edges;
  int32_t fixed_steps; // if > 0, fixed number of steps

  float rho;
  float alpha;
  float p_best;

  bool use_local_search;
  bool extend_ls;
  bool smooth_mmas;
  bool disable_heuristic;
  bool nls;
  int32_t T_nls; // number of NLS iterations (default 10)

  // Inter-route LS move selection flags (for ablation study)
  bool use_relocate;
  bool use_swap;
  bool use_2opt_star;

  float capacity;
  int64_t capacity_int;

  // -------------------------
  // State
  // -------------------------
  std::vector<float> coords;       // (n,2)
  std::vector<float> demand;       // (n,) demand[0]=0
  std::vector<int64_t> demand_int; // (n,) scaled/rounded
  std::vector<float> d0;           // (n,) dist to depot (precomputed)

  std::vector<int32_t> nn_list;     // (n,k)
  std::vector<int32_t> backup_list; // (n,bl)
  // std::vector<int32_t> nn_pos;         // REMOVED
  std::vector<float> heuristic_sparse; // (n,k)

  std::vector<float> pheromone_sparse; // (n,k)
  float tau_min = 0.0f;
  float tau_max = 0.0f;

  // -- LS Optimization Structures (Moved to Local) --
  // -------------------------------
  // Routes
  std::vector<int32_t> source_route;
  std::vector<int32_t> best_route;
  std::vector<int32_t> source_positions; // (n,) position of customer

  float source_cost = 0.0f;
  float best_cost = 0.0f;

  // Timings (seconds)
  double time_ant = 0.0;
  double time_ls = 0.0;
  double time_split = 0.0;

  // RNG
  Xoshiro128Plus rng_;

public:
  MFACO_CVRP(const float *coords_ptr, const float *demand_ptr, int32_t n_,
             float capacity_, int32_t n_ants_, int32_t cand_list_size,
             int32_t backup_list_size, int32_t min_new_edges_, float decay,
             float alpha_, float p_best_, bool use_local_search_,
             bool disable_heuristic_, bool extend_ls_ = false,
             bool smooth_mmas_ = false, int32_t fixed_steps_ = 0,
             bool nls_ = false, int32_t T_nls_ = 10);

  void seed_rng(uint64_t seed);

  void sample(bool require_prob, const float *prior_ptr, SampleResult &result,
              bool parallel_traced);

  void reset_timings();

  // Update pheromone with best permutation (length m) and its CVRP cost
  void update_pheromone(const int32_t *best_perm_ptr, float new_best_cost);
  void update_pheromone(const std::vector<int32_t> &best_route_in,
                        float new_best_cost);

  // Decode permutation into route-with-zeros (for convenience / debug)
  // perm_ptr length m
  void decode_perm_to_route0(const int32_t *perm_ptr,
                             std::vector<int32_t> &out_route0) const;

  // Data access for pybind views
  float *pheromone_data() { return pheromone_sparse.data(); }
  int32_t *nn_list_data() { return nn_list.data(); }
  int32_t *backup_list_data() { return backup_list.data(); }
  float *heuristic_data() { return heuristic_sparse.data(); }
  // int32_t *nn_pos_data() { return nn_pos.data(); }

  inline int32_t find_neighbor_index(int32_t u, int32_t v) const {
    for (int32_t j = 0; j < k; ++j) {
      if (nn_list[u * k + j] == v)
        return j;
    }
    return -1;
  }
  int32_t *source_perm_data() { return source_route.data(); }
  int32_t *best_perm_data() { return best_route.data(); }

private:
  // ---- build helpers ----
  void build_nn_lists();
  // void build_nn_pos();
  void build_heuristic();
  void build_d0();
  void build_initial_perm();
  void build_initial_solution();

  // ---- probability mat on sparse graph (n,k) ----
  void compute_probmat(const float *prior_ptr, std::vector<float> &probmat);

  std::tuple<int32_t, bool, float>
  select_next_node(int32_t curr, int32_t curr_route, const float *probmat_row,
                   const std::vector<uint8_t> &visited,
                   const std::vector<int32_t> &node_route,
                   const std::vector<int64_t> &route_loads, Xoshiro128Plus &rng,
                   int16_t &out_pick_j, uint64_t &out_valid_mask);

  // ---- MFACO sampling primitives on giant tour ----
  float sample_ant_fast(const float *probmat, int32_t start_node,
                        std::vector<int32_t> &perm_out, int32_t &new_edges_out,
                        std::vector<int32_t> &checklist, Xoshiro128Plus &rng,
                        const float *prior);

  float sample_ant_direct(const float *probmat, int32_t start_node,
                          std::vector<int32_t> &route_out,
                          int32_t &new_edges_out,
                          std::vector<int32_t> &checklist, Xoshiro128Plus &rng,
                          const float *prior);

  float sample_ant_direct_traced(const float *probmat, int32_t start_node,
                                 std::vector<int32_t> &route_out,
                                 std::vector<int32_t> &route_raw_out,
                                 float &cost_raw_out, int32_t &new_edges_out,
                                 std::vector<int32_t> &checklist,
                                 MFACOTrace &trace, Xoshiro128Plus &rng,
                                 float &logp_sum, float &survival_out,
                                 const float *prior);

  float intra_route_ls(std::vector<int32_t> &route,
                       std::vector<int32_t> &checklist);

  float inter_route_ls_optimized(std::vector<int32_t> &perm,
                                 std::vector<int32_t> &positions,
                                 std::vector<int32_t> &checklist,
                                 std::vector<uint8_t> &in_checklist);

  float sample_ant_traced(const float *probmat, int32_t start_node,
                          std::vector<int32_t> &perm_out,
                          std::vector<int32_t> &perm_raw_out,
                          float &cost_raw_out, int32_t &new_edges_out,
                          std::vector<int32_t> &checklist,
                          class MFACOTrace &trace, Xoshiro128Plus &rng,
                          float &logp_sum, float &survival_out,
                          const float *prior);

  std::tuple<int32_t, bool, float>
  select_next_node(int32_t curr, const float *probmat_row,
                   const uint8_t *visited, Xoshiro128Plus &rng,
                   int16_t &out_pick_j, uint64_t &out_valid_mask);

  // permutation moves (cycle semantics on length m)
  float relocate_node(int32_t target, int32_t node, std::vector<int32_t> &perm,
                      std::vector<int32_t> &positions);
  float two_opt_nn(std::vector<int32_t> &perm, std::vector<int32_t> &positions,
                   std::vector<int32_t> &checklist,
                   std::vector<uint8_t> &in_checklist);
  float two_opt_nn_prior(std::vector<int32_t> &perm,
                         std::vector<int32_t> &positions,
                         std::vector<int32_t> &checklist,
                         std::vector<uint8_t> &in_checklist,
                         const float *prior_ptr);
  void flip_route_section(int32_t start_node, int32_t end_node,
                          std::vector<int32_t> &perm,
                          std::vector<int32_t> &positions);

  // ---- Inter-route LS ----
  void inter_route_ls(std::vector<int32_t> &perm,
                      std::vector<int32_t> &positions,
                      std::vector<int32_t> &checklist,
                      std::vector<uint8_t> &in_checklist);

  // Reconstruct routes from permutation using split_dp, but returning vector
  // of routes
  std::vector<std::vector<int32_t>>
  initial_routes_from_perm(const std::vector<int32_t> &perm) const;

  // Flatten routes back to permutation (simple concatenation)
  void routes_to_perm(const std::vector<std::vector<int32_t>> &routes,
                      std::vector<int32_t> &perm,
                      std::vector<int32_t> &positions);

  bool ls_relocate(std::vector<std::vector<int32_t>> &routes,
                   std::vector<int64_t> &loads, int32_t u_node, int32_t r_u,
                   int32_t idx_u, int32_t r_v, int32_t idx_v,
                   std::vector<int32_t> &node_pos);
  bool ls_swap(std::vector<std::vector<int32_t>> &routes,
               std::vector<int64_t> &loads, int32_t u_node, int32_t r_u,
               int32_t idx_u, int32_t r_v, int32_t idx_v,
               std::vector<int32_t> &node_pos);
  bool ls_2opt_star(std::vector<std::vector<int32_t>> &routes,
                    std::vector<int64_t> &loads, int32_t u_node, int32_t r_u,
                    int32_t idx_u, int32_t r_v, int32_t idx_v,
                    std::vector<int32_t> &node_pos,
                    std::vector<int32_t> &node_to_route);

  bool contains_edge(int32_t a, int32_t b) const;
  int32_t get_succ(int32_t node, const std::vector<int32_t> &perm,
                   const std::vector<int32_t> &positions) const;
  int32_t get_pred(int32_t node, const std::vector<int32_t> &perm,
                   const std::vector<int32_t> &positions) const;

  // ---- CVRP split DP cost + segments ----
  struct SplitResult {
    float cost;
    std::vector<std::pair<int32_t, int32_t>>
        segs; // inclusive [i,j] on perm indices
  };
  bool elitist;
  // bool use_local_search; // already redundant
  SplitResult split_dp(const std::vector<int32_t> &perm) const;
  float split_cost_fast(const std::vector<int32_t> &perm) const;

  // bool use_local_search; // already redundant
  std::pair<float, float> calc_trail_limits_cl(float solution_cost) const;
  std::pair<float, float> calc_trail_limits_smooth(float solution_cost) const;

  // distance
  float dist(int32_t u, int32_t v) const;
};

// ============================================================================
// ACO_CVRP class - Standard Constructive MMAS for CVRP
// ============================================================================

class ACO_CVRP {
public:
  // Configuration
  int32_t n; // includes depot at 0
  int32_t n_ants;
  int32_t k; // candidate list size (usually n-1 for dense)
  float rho; // pheromone decay
  float alpha;
  float beta;   // heuristic exponent
  float p_best; // for MMAS limits
  bool min_max; // Use MMAS min-max limits

  float capacity;

  // State
  std::vector<float> coords; // (n, 2)
  std::vector<float> demand; // (n,)
  bool elitist;
  bool use_local_search;

  std::vector<int32_t> nn_list; // (n, k)
  std::vector<float> pheromone; // (n, k)
  std::vector<float> heuristic; // (n, k)

  std::vector<int32_t> best_route; // (max_len,) - potentially long
  float best_cost;
  float tau_min;
  float tau_max;

  // RNG
  Xoshiro128Plus rng_;

  // Constructor
  ACO_CVRP(const float *coords_ptr, const float *demand_ptr, int32_t n,
           float capacity, int32_t n_ants,
           int32_t cand_list_size = 0, // 0 means dense (n-1)
           float decay = 0.9f, float alpha = 1.0f, float beta = 1.0f,
           float p_best = 0.05f, bool min_max = true, bool elitist = false,
           bool use_local_search = false);

  // API
  void seed_rng(uint64_t seed);

  void sample(bool require_prob, const float *prior, SampleResult &result,
              bool parallel_traced = false);

  float run(int32_t n_iterations);

  void update_pheromone(const int32_t *solution_flat, int32_t solution_len,
                        float cost);

  void update_pheromone_batch(const std::vector<std::vector<int32_t>> &routes,
                              const std::vector<float> &costs);

  void local_search(std::vector<int32_t> &route);
  void two_opt_sequence(std::vector<int32_t> &seq);

  // Data access
  float *pheromone_data() { return pheromone.data(); }
  int32_t *nn_list_data() { return nn_list.data(); }
  float *heuristic_data() { return heuristic.data(); }
  std::vector<int32_t> get_best_route() const { return best_route; }
  void build_dense_nn_lists();
  void build_heuristic();

  inline float dist(int32_t a, int32_t b) const {
    const size_t ia = static_cast<size_t>(a) * 2;
    const size_t ib = static_cast<size_t>(b) * 2;
    const float dx = coords[ib + 0] - coords[ia + 0];
    const float dy = coords[ib + 1] - coords[ia + 1];
    return std::sqrt(dx * dx + dy * dy);
  }

  void compute_probmat(const float *prior, std::vector<float> &probmat);

  // Sample one ant
  // Returns cost. Fills route_out with full path (nodes + depots).
  float sample_ant_constructive(const float *probmat,
                                std::vector<int32_t> &route_out,
                                Xoshiro128Plus &rng, bool require_prob,
                                float &logp_out, MFACOTrace *trace = nullptr);

  std::pair<float, float> calc_trail_limits(float solution_cost) const;
};

} // namespace mfaco
