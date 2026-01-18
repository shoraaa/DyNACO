// mfaco_train.h
#pragma once
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace mfaco {

// ============================================================================
// Constants
// ============================================================================

static constexpr uint32_t MAX_CAND_LIST_SIZE = 64;
static constexpr float EPS = 1e-8f;
static constexpr float LOG_EPS = 1e-8f;
static constexpr int64_t DEMAND_SCALE = 100000;

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

  void clear() {
    start_node = -1;
    curr_nodes.clear();
    chosen_nodes.clear();
    is_stochastic.clear();
    pick_j.clear();
    valid_mask.clear();
  }

  void reserve(size_t n) {
    curr_nodes.reserve(n);
    chosen_nodes.reserve(n);
    is_stochastic.reserve(n);
    pick_j.reserve(n);
    valid_mask.reserve(n);
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
  std::vector<int32_t> start_nodes; // length n_ants (start node per ant)

  void clear() {
    starts.clear();
    curr_nodes.clear();
    chosen_nodes.clear();
    is_stochastic.clear();
    pick_j.clear();
    valid_mask.clear();
    start_nodes.clear();
  }

  void reserve(int32_t n_ants, int32_t max_decisions) {
    starts.reserve(n_ants + 1);
    curr_nodes.reserve(max_decisions);
    chosen_nodes.reserve(max_decisions);
    is_stochastic.reserve(max_decisions);
    pick_j.reserve(max_decisions);
    valid_mask.reserve(max_decisions);
    start_nodes.reserve(n_ants);
  }
};

// ============================================================================
// Sample result structure
// ============================================================================

struct SampleResult {
  std::vector<float> costs; // (n_ants,)
  std::vector<std::vector<int32_t>>
      routes;               // (n_ants, n) - each route without repetition
  MFACOTraceBatch traces;   // only populated if require_prob=true
  std::vector<float> logps; // (n_ants,) log probability of sample

  void clear() {
    costs.clear();
    routes.clear();
    traces.clear();
    logps.clear();
  }
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

  float rho;
  float alpha;
  float p_best;

  bool use_local_search;
  bool extend_ls;
  bool smooth_mmas;
  bool disable_heuristic;

  float capacity;
  int64_t capacity_int;

  // -------------------------
  // State
  // -------------------------
  std::vector<float> coords;       // (n,2)
  std::vector<float> demand;       // (n,) demand[0]=0
  std::vector<int64_t> demand_int; // (n,) scaled/rounded
  std::vector<float> d0;           // (n,) dist to depot (precomputed)

  std::vector<int32_t> nn_list;        // (n,k)
  std::vector<int32_t> backup_list;    // (n,bl)
  std::vector<int32_t> nn_pos;         // (n,n) dense map (same as TSP)
  std::vector<float> heuristic_sparse; // (n,k)

  std::vector<float> pheromone_sparse; // (n,k)
  float tau_min = 0.0f;
  float tau_max = 0.0f;

  // Giant tour permutation of customers only (length m)
  std::vector<int32_t> source_perm;
  std::vector<int32_t> best_perm;
  std::vector<int32_t>
      source_positions; // (n,) position of customer in source_perm; depot=-1

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
             bool smooth_mmas_ = false);

  void seed_rng(uint64_t seed);

  void sample(bool require_prob, const float *prior_ptr, SampleResult &result,
              bool parallel_traced);

  void reset_timings();

  // Update pheromone with best permutation (length m) and its CVRP cost
  void update_pheromone(const int32_t *best_perm_ptr, float new_best_cost);

  // Decode permutation into route-with-zeros (for convenience / debug)
  // perm_ptr length m
  void decode_perm_to_route0(const int32_t *perm_ptr,
                             std::vector<int32_t> &out_route0) const;

  // Data access for pybind views
  float *pheromone_data() { return pheromone_sparse.data(); }
  int32_t *nn_list_data() { return nn_list.data(); }
  int32_t *backup_list_data() { return backup_list.data(); }
  float *heuristic_data() { return heuristic_sparse.data(); }
  int32_t *nn_pos_data() { return nn_pos.data(); }
  int32_t *source_perm_data() { return source_perm.data(); }
  int32_t *best_perm_data() { return best_perm.data(); }

private:
  // ---- build helpers (reuse your TSP-style approach) ----
  void build_nn_lists();
  void build_nn_pos();
  void build_heuristic();
  void build_d0();
  void build_initial_perm();

  // ---- probability mat on sparse graph (n,k) ----
  void compute_probmat(const float *prior_ptr, std::vector<float> &probmat);

  // ---- MFACO sampling primitives on giant tour ----
  float sample_ant_fast(const float *probmat, int32_t start_node,
                        std::vector<int32_t> &perm_out,
                        std::vector<int32_t> &checklist, Xoshiro128Plus &rng);

  float sample_ant_traced(const float *probmat, int32_t start_node,
                          std::vector<int32_t> &perm_out,
                          std::vector<int32_t> &checklist,
                          class MFACOTrace &trace, Xoshiro128Plus &rng,
                          float &logp_sum);

  std::tuple<int32_t, bool, float>
  select_next_node(int32_t curr, const float *probmat_row,
                   const uint8_t *visited, Xoshiro128Plus &rng,
                   int16_t &out_pick_j, uint64_t &out_valid_mask);

  // permutation moves (cycle semantics on length m)
  float relocate_node(int32_t target, int32_t node, std::vector<int32_t> &perm,
                      std::vector<int32_t> &positions);
  float two_opt_nn(std::vector<int32_t> &perm, std::vector<int32_t> &positions,
                   std::vector<int32_t> &checklist);
  void flip_route_section(int32_t start_node, int32_t end_node,
                          std::vector<int32_t> &perm,
                          std::vector<int32_t> &positions);

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
  SplitResult split_dp(const std::vector<int32_t> &perm) const;
  float split_cost_fast(const std::vector<int32_t> &perm) const;

  // pheromone bounds
  std::pair<float, float> calc_trail_limits_cl(float solution_cost) const;
  std::pair<float, float> calc_trail_limits_smooth(float solution_cost) const;

  // distance (same as your TSP)
  float dist(int32_t u, int32_t v) const;
};

} // namespace mfaco
