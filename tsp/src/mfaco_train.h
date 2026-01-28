/**
 * MFACO Training Module - C++ implementation for fast neural-guided ACO
 * training.
 *
 * Designed to match the Python MFACO_TSP API in faco_tsp.py for seamless
 * integration with the train.py REINFORCE/A2C training pipeline.
 *
 * Key features:
 * - Sparse candidate-list pheromone storage (n x k)
 * - Relocate-based focused editing from source solution
 * - Optional neural prior for learned biases
 * - Trace recording for log-probability replay
 * - Checklist-guided 2-opt local search
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace mfaco {

// ============================================================================
// Constants
// ============================================================================

static constexpr uint32_t MAX_CAND_LIST_SIZE = 64;
static constexpr float EPS = 1e-12f;
static constexpr float LOG_EPS = 1e-12f;

// Distance types supported by this MFACO training backend.
// NOTE: spelling kept as requested.
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
  std::vector<float> costs_raw; // (n_ants,) cost before LS
  std::vector<std::vector<int32_t>>
      routes_raw; // (n_ants, n) - each route before LS
  std::vector<int32_t> new_edges_count; // (n_ants,)

  void clear() {
    costs.clear();
    routes.clear();
    traces.clear();
    logps.clear();
    costs_raw.clear();
    routes_raw.clear();
    new_edges_count.clear();
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
  float rho;        // pheromone decay (1 - evaporation rate)
  float alpha;      // pheromone exponent
  float p_best;     // for tau limits
  bool smooth_mmas; // if true, use Smooth-MMAS style pheromone update (linear
                    // interpolation)
  bool use_local_search;
  bool extend_ls; // if true, extend local search checklist with endpoints of
                  // improving moves
  bool disable_heuristic; // if true, set eta=1 so sampling uses pheromone (+
                          // prior) only

  // Distance type (currently fixed to explicit Euclidean 2D without rounding).
  DistanceType distance_type = DistanceType::EXPLICT_EUC_2D;

  // State arrays
  std::vector<float> coords;        // (n, 2) row-major (x,y)
  std::vector<int32_t> nn_list;     // (n, k) row-major: nearest neighbors
  std::vector<int32_t> backup_list; // (n, bl) row-major: backup neighbors
  std::vector<float>
      pheromone_sparse; // (n, k) row-major: pheromone on candidate edges
  std::vector<float>
      heuristic_sparse; // (n, k) row-major: 1/dist for candidate edges
  std::vector<int32_t>
      nn_pos; // (n, n) row-major: nn_pos[u,v] = j if nn_list[u,j]=v, else -1

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
            int32_t fixed_steps = 0);

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
  int32_t *nn_pos_data() { return nn_pos.data(); }

private:
  // ========================================================================
  // Internal methods
  // ========================================================================

  /**
   * Build nearest neighbor and backup lists from distances.
   */
  void build_nn_lists();

  /**
   * Build nn_pos mapping from nn_list.
   */
  void build_nn_pos();

  /**
   * Build heuristic_sparse from distances and nn_list.
   */
  void build_heuristic();

  /**
   * Build initial greedy NN tour.
   */
  void build_initial_tour();

  // Explicit Euclidean 2D distance without rounding.
  inline float dist(int32_t a, int32_t b) const {
    const size_t ia = static_cast<size_t>(a) * 2;
    const size_t ib = static_cast<size_t>(b) * 2;
    const float dx = coords[ib + 0] - coords[ia + 0];
    const float dy = coords[ib + 1] - coords[ia + 1];
    return std::sqrt(dx * dx + dy * dy);
  }

  /**
   * Calculate tau limits using MMAS candidate-list formula.
   */
  std::pair<float, float> calc_trail_limits_cl(float solution_cost) const;

  /**
   * Calculate tau limits for Smooth-MMAS. Used together with the linear
   * interpolation update.
   */
  std::pair<float, float> calc_trail_limits_smooth(float solution_cost) const;

  /**
   * Sample a single ant's solution (fast mode, no tracing).
   */
  float sample_ant_fast(const float *probmat, // (n, k) precomputed weights
                        int32_t start_node, std::vector<int32_t> &route_out,
                        int32_t &new_edges_out,
                        std::vector<int32_t> &checklist, Xoshiro128Plus &rng);

  /**
   * Sample a single ant's solution with trace recording.
   */
  float sample_ant_traced(const float *probmat, // (n, k) precomputed weights
                          int32_t start_node, std::vector<int32_t> &route_out,
                          std::vector<int32_t> &route_raw_out,
                          float &cost_raw_out, int32_t &new_edges_out, std::vector<int32_t> &checklist,
                          MFACOTrace &trace, Xoshiro128Plus &rng,
                          float &logp_sum);

  /**
   * Select next node using roulette wheel on candidate list.
   * Returns (chosen_node, is_stochastic, log_prob_of_choice).
   */
  std::tuple<int32_t, bool, float>
  select_next_node(int32_t curr,
                   const float *probmat_row, // probmat[curr*k : curr*k + k]
                   const uint8_t *visited, Xoshiro128Plus &rng,
                   int16_t &out_pick_j, uint64_t &out_valid_mask);

  /**
   * Relocate node to be successor of target in route.
   * Updates route, positions, and returns cost delta.
   */
  float relocate_node(int32_t target, int32_t node, std::vector<int32_t> &route,
                      std::vector<int32_t> &positions);

  /**
   * 2-opt local search using checklist and nearest neighbor optimization.
   */
  float two_opt_nn(std::vector<int32_t> &route, std::vector<int32_t> &positions,
                   std::vector<int32_t> &checklist);

  /**
   * Get route cost.
   */
  float get_route_cost(const std::vector<int32_t> &route) const;

  /**
   * Check if edge (a, b) exists in route (undirected).
   */
  bool contains_edge(int32_t a, int32_t b,
                     const std::vector<int32_t> &positions) const;

  /**
   * Get successor of node in route.
   */
  int32_t get_succ(int32_t node, const std::vector<int32_t> &route,
                   const std::vector<int32_t> &positions) const;

  /**
   * Get predecessor of node in route.
   */
  int32_t get_pred(int32_t node, const std::vector<int32_t> &route,
                   const std::vector<int32_t> &positions) const;

  /**
   * Flip route section for 2-opt.
   */
  void flip_route_section(int32_t start_node, int32_t end_node,
                          std::vector<int32_t> &route,
                          std::vector<int32_t> &positions);

  /**
   * Compute probability matrix: tau^alpha * eta * prior.
   */
  void compute_probmat(const float *prior, std::vector<float> &probmat);
};

} // namespace mfaco
