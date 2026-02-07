/**
 * MFACO Training Module - Unified C++ implementation
 */

#include "mfaco_train.h"
#include "kd_tree.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include <omp.h>
#include <stdexcept>
#include <tuple>
#include <vector>
namespace mfaco {

// ============================================================================
// Constructor
// ============================================================================

MFACO_TSP::MFACO_TSP(const float *coords_ptr, int32_t n_, int32_t n_ants_,
                     int32_t cand_list_size, int32_t backup_list_size,
                     int32_t min_new_edges_, float decay, float alpha_,
                     float p_best_, bool use_local_search_,
                     bool disable_heuristic_, bool extend_ls_,
                     bool smooth_mmas_, int32_t fixed_steps_, bool nls_,
                     int32_t T_nls_)
    : n(n_), n_ants(n_ants_), k(std::min(cand_list_size, n_ - 1)),
      bl(std::min(backup_list_size, std::max(0, n_ - 1 - k))),
      min_new_edges(min_new_edges_), rho(decay), alpha(alpha_), p_best(p_best_),
      smooth_mmas(smooth_mmas_), use_local_search(use_local_search_),
      extend_ls(extend_ls_), disable_heuristic(disable_heuristic_),
      fixed_steps(fixed_steps_), nls(nls_), T_nls(T_nls_) {
  if (coords_ptr == nullptr) {
    throw std::runtime_error("coords_ptr must not be null");
  }

  // Store coordinates and compute distances on-demand (EXPLICT_EUC_2D).
  coords.resize(static_cast<size_t>(n) * 2);
  std::memcpy(coords.data(), coords_ptr,
              sizeof(float) * static_cast<size_t>(n) * 2);

  // Build nearest neighbor lists
  build_nn_lists();

  build_heuristic();

  // Initialize source/best solutions
  source_route.resize(n);
  best_route.resize(n);
  source_positions.resize(n);
  build_initial_tour();

  // Initialize pheromone
  auto [tmin, tmax] = smooth_mmas ? calc_trail_limits_smooth(source_cost)
                                  : calc_trail_limits_cl(source_cost);
  tau_min = tmin;
  tau_max = tmax;
  pheromone_sparse.assign(n * k, tau_max);

  // Initialize RNG
  rng_.seed(42);
}

// ============================================================================
// Public API
// ============================================================================

void MFACO_TSP::seed_rng(uint64_t seed) { rng_.seed(seed); }

void MFACO_TSP::sample(bool require_prob, const float *prior,
                       SampleResult &result, bool parallel_traced) {
  result.clear();
  result.costs.resize(n_ants);
  result.routes.resize(n_ants);
  if (require_prob) {
    result.costs_raw.resize(n_ants);
    result.routes_raw.resize(n_ants);
  }
  result.new_edges_count.resize(n_ants);
  result.edge_survival.resize(n_ants);

  // Compute probability matrix: tau^alpha * eta * prior
  std::vector<float> probmat(n * k);
  compute_probmat(prior, probmat);

  // Generate random start nodes
  std::vector<int32_t> start_nodes(n_ants);
  for (int32_t a = 0; a < n_ants; ++a) {
    start_nodes[a] = rng_.next_uint(n);
  }

  // Per-ant RNG seeds are only needed for OpenMP paths.
  // IMPORTANT: avoid consuming extra RNG in the default traced single-thread
  // path.
  std::vector<uint64_t> ant_seeds;
  auto ensure_ant_seeds = [&]() {
    if (!ant_seeds.empty())
      return;
    ant_seeds.resize(static_cast<size_t>(n_ants));
    for (int32_t a = 0; a < n_ants; ++a) {
      uint64_t hi = static_cast<uint64_t>(rng_.next_u32());
      uint64_t lo = static_cast<uint64_t>(rng_.next_u32());
      ant_seeds[static_cast<size_t>(a)] =
          (hi << 32) ^ lo ^ (0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(a));
    }
  };

  if (require_prob) {
    result.logps.resize(n_ants);
    if (!parallel_traced) {
      // Traced mode: single-threaded (original behavior)
      result.traces.reserve(n_ants, n * n_ants);
      result.traces.starts.push_back(0);

      std::vector<int32_t> checklist;
      checklist.reserve(n);

      for (int32_t a = 0; a < n_ants; ++a) {
        result.routes[a].resize(n);
        result.routes_raw[a].resize(n);
        MFACOTrace trace;
        trace.reserve(min_new_edges * 2);

        float logp_sum = 0.0f;
        int32_t mne_out = 0;
        float surv_out = 0.0f;
        float cost = sample_ant_traced(probmat.data(), start_nodes[a],
                                       result.routes[a], result.routes_raw[a],
                                       result.costs_raw[a], mne_out, checklist,
                                       trace, rng_, logp_sum, surv_out, prior);
        result.new_edges_count[a] = mne_out;
        result.edge_survival[a] = surv_out;

        result.costs[a] = cost;
        result.logps[a] = logp_sum;

        // Append trace to batch
        result.traces.start_nodes.push_back(trace.start_node);
        for (size_t i = 0; i < trace.curr_nodes.size(); ++i) {
          result.traces.curr_nodes.push_back(trace.curr_nodes[i]);
          result.traces.chosen_nodes.push_back(trace.chosen_nodes[i]);
          result.traces.is_stochastic.push_back(trace.is_stochastic[i]);
          result.traces.pick_j.push_back(trace.pick_j[i]);
          result.traces.valid_mask.push_back(trace.valid_mask[i]);
          result.traces.is_new_edge.push_back(trace.is_new_edge[i]);
        }
        result.traces.starts.push_back(
            static_cast<int32_t>(result.traces.curr_nodes.size()));
      }
    } else {
      // Traced mode: parallelized with per-ant RNG and per-ant traces
      ensure_ant_seeds();
      std::vector<MFACOTrace> traces_per_ant(static_cast<size_t>(n_ants));

#pragma omp parallel
      {
        std::vector<int32_t> checklist;
        checklist.reserve(n);

#pragma omp for schedule(static, 1)
        for (int32_t a = 0; a < n_ants; ++a) {
          result.routes[a].resize(n);
          result.routes_raw[a].resize(n);
          MFACOTrace &trace = traces_per_ant[static_cast<size_t>(a)];
          trace.reserve(min_new_edges * 2);
          Xoshiro128Plus rng_local;
          rng_local.seed(ant_seeds[static_cast<size_t>(a)]);

          float logp_sum = 0.0f;
          int32_t mne_out = 0;
          float surv_out = 0.0f;
          result.costs[a] = sample_ant_traced(
              probmat.data(), start_nodes[a], result.routes[a],
              result.routes_raw[a], result.costs_raw[a], mne_out, checklist,
              trace, rng_local, logp_sum, surv_out, prior);

          result.new_edges_count[a] = mne_out;
          result.edge_survival[a] = surv_out;
          result.logps[a] = logp_sum;
        }
      }

      // Merge traces in ant index order (deterministic)
      result.traces.clear();
      result.traces.starts.resize(static_cast<size_t>(n_ants) + 1);
      result.traces.start_nodes.resize(static_cast<size_t>(n_ants));
      result.traces.starts[0] = 0;
      for (int32_t a = 0; a < n_ants; ++a) {
        const MFACOTrace &t = traces_per_ant[static_cast<size_t>(a)];
        result.traces.start_nodes[static_cast<size_t>(a)] = t.start_node;
        result.traces.starts[static_cast<size_t>(a) + 1] =
            result.traces.starts[static_cast<size_t>(a)] +
            static_cast<int32_t>(t.curr_nodes.size());
      }
      int32_t total = result.traces.starts[static_cast<size_t>(n_ants)];
      result.traces.curr_nodes.resize(static_cast<size_t>(total));
      result.traces.chosen_nodes.resize(static_cast<size_t>(total));
      result.traces.is_stochastic.resize(static_cast<size_t>(total));
      result.traces.pick_j.resize(static_cast<size_t>(total));
      result.traces.valid_mask.resize(static_cast<size_t>(total));
      result.traces.is_new_edge.resize(static_cast<size_t>(total));

      for (int32_t a = 0; a < n_ants; ++a) {
        const MFACOTrace &t = traces_per_ant[static_cast<size_t>(a)];
        int32_t off = result.traces.starts[static_cast<size_t>(a)];
        for (size_t i = 0; i < t.curr_nodes.size(); ++i) {
          result.traces.curr_nodes[static_cast<size_t>(off) + i] =
              t.curr_nodes[i];
          result.traces.chosen_nodes[static_cast<size_t>(off) + i] =
              t.chosen_nodes[i];
          result.traces.is_stochastic[static_cast<size_t>(off) + i] =
              t.is_stochastic[i];
          result.traces.pick_j[static_cast<size_t>(off) + i] = t.pick_j[i];
          result.traces.valid_mask[static_cast<size_t>(off) + i] =
              t.valid_mask[i];
          result.traces.is_new_edge[static_cast<size_t>(off) + i] =
              t.is_new_edge[i];
        }
      }
    }
  } else {
    // Fast mode: parallel
    ensure_ant_seeds();
#pragma omp parallel
    {
      std::vector<int32_t> checklist;
      checklist.reserve(n);

#pragma omp for schedule(static, 1)
      for (int32_t a = 0; a < n_ants; ++a) {
        result.routes[a].resize(n);
        Xoshiro128Plus rng_local;
        rng_local.seed(ant_seeds[static_cast<size_t>(a)]);
        result.costs[a] = sample_ant_fast(
            probmat.data(), start_nodes[a], result.routes[a],
            result.new_edges_count[a], checklist, rng_local, prior);
      }
    }
  }
}

void MFACO_TSP::update_pheromone(const int32_t *best_flat,
                                 float new_best_cost) {
  // Update global best
  if (new_best_cost < best_cost) {
    best_cost = new_best_cost;
    std::copy(best_flat, best_flat + n, best_route.begin());
  }

  // Update trail limits based on global best
  auto [tmin, tmax] = smooth_mmas ? calc_trail_limits_smooth(best_cost)
                                  : calc_trail_limits_cl(best_cost);
  tau_min = tmin;
  tau_max = tmax;

  // Precompute positions for O(1) in-route check (used by both methods now)
  std::vector<int32_t> pos(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    int32_t v = best_flat[i];
    pos[static_cast<size_t>(v)] = i;
  }

  auto in_route_edge = [&](int32_t u, int32_t v) -> bool {
    int32_t pu = pos[static_cast<size_t>(u)];
    int32_t pv = pos[static_cast<size_t>(v)];
    int32_t diff = std::abs(pu - pv);
    return diff == 1 || diff == n - 1;
  };

  const float decay_factor = 1.0f - rho;
  const float deposit = (!smooth_mmas) ? (1.0f / (new_best_cost + EPS)) : 0.0f;

#pragma omp parallel for schedule(static)
  for (int32_t u = 0; u < n; ++u) {
    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_list[u * k + j];
      float &tau = pheromone_sparse[u * k + j];
      bool is_in = in_route_edge(u, v);

      if (smooth_mmas) {
        float target = is_in ? tau_max : tau_min;
        tau = decay_factor * tau + rho * target;
      } else {
        tau *= decay_factor;
        if (is_in) {
          tau += deposit;
        }
        tau = std::max(tau_min, std::min(tau_max, tau));
      }
    }
  }

  // Update source solution
  std::copy(best_flat, best_flat + n, source_route.begin());
  source_cost = new_best_cost;
  for (int32_t i = 0; i < n; ++i) {
    source_positions[source_route[i]] = i;
  }
}

void MFACO_TSP::load_snapshot(const float *pheromone_ptr,
                              const int32_t *source_route_ptr,
                              float source_cost_, const int32_t *best_route_ptr,
                              float best_cost_, float tau_min_, float tau_max_,
                              const int32_t *nn_list_ptr,
                              const int32_t *backup_list_ptr) {
  // Copy pheromone
  std::copy(pheromone_ptr, pheromone_ptr + n * k, pheromone_sparse.begin());

  // Copy source route and cost
  std::copy(source_route_ptr, source_route_ptr + n, source_route.begin());
  source_cost = source_cost_;

  // Copy best route and cost
  std::copy(best_route_ptr, best_route_ptr + n, best_route.begin());
  best_cost = best_cost_;

  // Copy tau limits
  tau_min = tau_min_;
  tau_max = tau_max_;

  // Copy nn_list and backup_list
  std::copy(nn_list_ptr, nn_list_ptr + n * k, nn_list.begin());
  if (bl > 0) {
    std::copy(backup_list_ptr, backup_list_ptr + n * bl, backup_list.begin());
  }

  // Rebuild nn_pos from nn_list if needed
  // if (!smooth_mmas)
  //   build_nn_pos();

  // Rebuild heuristic (in case nn_list changed)
  build_heuristic();

  // Rebuild source_positions
  for (int32_t i = 0; i < n; ++i) {
    source_positions[source_route[i]] = i;
  }
}

void MFACO_TSP::set_pheromone(const float *pheromone_ptr) {
  std::copy(pheromone_ptr, pheromone_ptr + n * k, pheromone_sparse.begin());
}

// ============================================================================
// Private methods
// ============================================================================

void MFACO_TSP::build_nn_lists() {
  nn_list.resize(n * k);
  backup_list.resize(n * bl);

  const int32_t total = k + bl;
  if (total <= 0) {
    return;
  }

  // Build kd-tree points (double precision for kd-tree internal ops).
  std::vector<Vec2d> pts(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const size_t off = static_cast<size_t>(i) * 2;
    pts[static_cast<size_t>(i)] = Vec2d{static_cast<double>(coords[off + 0]),
                                        static_cast<double>(coords[off + 1])};
  }

  // NOTE: use exact distances (no rounding) for neighbor selection.
  KDTree shared_kdtree(pts, /*round_distances=*/false);

#pragma omp parallel default(none) shared(shared_kdtree, nn_list, backup_list) \
    firstprivate(n, k, bl, total)
  {
    KDTree kdtree = shared_kdtree; // private copy (supports delete/undelete)

#pragma omp for schedule(static)
    for (int32_t u = 0; u < n; ++u) {
      // Collect total nearest (k + bl) using repeated NN queries with
      // deletions.
      for (int32_t j = 0; j < total; ++j) {
        uint32_t pt_idx = kdtree.nn_bottom_up(static_cast<uint32_t>(u));
        if (j < k) {
          nn_list[u * k + j] = static_cast<int32_t>(pt_idx);
        } else {
          backup_list[u * bl + (j - k)] = static_cast<int32_t>(pt_idx);
        }
        kdtree.delete_point(pt_idx);
      }

      // Revert changes so that kdtree can be reused for other rows.
      for (int32_t j = 0; j < k; ++j) {
        kdtree.undelete_point(static_cast<uint32_t>(nn_list[u * k + j]));
      }
      for (int32_t j = 0; j < bl; ++j) {
        kdtree.undelete_point(static_cast<uint32_t>(backup_list[u * bl + j]));
      }
    }
  }
}

void MFACO_TSP::build_heuristic() {
  heuristic_sparse.resize(n * k);
  if (disable_heuristic) {
    std::fill(heuristic_sparse.begin(), heuristic_sparse.end(), 1.0f);
    return;
  }
  for (int32_t u = 0; u < n; ++u) {
    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_list[u * k + j];
      float d = dist(u, v);
      heuristic_sparse[u * k + j] = (d > 0) ? (1.0f / d) : 1.0f;
    }
  }
}

void MFACO_TSP::build_initial_tour() {
  // Build greedy NN tours from multiple starts, keep best
  float best = std::numeric_limits<float>::max();
  std::vector<int32_t> best_tour(n);

  int32_t num_starts = std::min(8, n);
  std::vector<int32_t> tour(n);
  std::vector<uint8_t> visited(n);

  for (int32_t start = 0; start < num_starts; ++start) {
    std::fill(visited.begin(), visited.end(), 0);
    tour[0] = start;
    visited[start] = 1;

    for (int32_t i = 1; i < n; ++i) {
      int32_t curr = tour[i - 1];
      int32_t next = -1;

      // Try nn_list first
      for (int32_t j = 0; j < k; ++j) {
        int32_t v = nn_list[curr * k + j];
        if (!visited[v]) {
          next = v;
          break;
        }
      }

      // Fallback: find closest unvisited
      if (next < 0) {
        float min_dist = std::numeric_limits<float>::max();
        for (int32_t v = 0; v < n; ++v) {
          float d = dist(curr, v);
          if (!visited[v] && d < min_dist) {
            min_dist = d;
            next = v;
          }
        }
      }

      tour[i] = next;
      visited[next] = 1;
    }

    float cost = get_route_cost(tour);
    if (cost < best) {
      best = cost;
      best_tour = tour;
    }
  }

  source_route = best_tour;
  source_cost = best;
  best_route = best_tour;
  best_cost = best;

  for (int32_t i = 0; i < n; ++i) {
    source_positions[source_route[i]] = i;
  }
}

std::pair<float, float>
MFACO_TSP::calc_trail_limits_cl(float solution_cost) const {
  float tau_max_ = 1.0f / (solution_cost * (1.0f - rho) + EPS);
  float avg = static_cast<float>(std::max(2, k));
  float p = std::pow(p_best, 1.0f / avg);
  float tau_min_ =
      std::min(tau_max_, tau_max_ * (1.0f - p) / ((avg - 1.0f) * p + EPS));
  return {tau_min_, tau_max_};
}

std::pair<float, float>
MFACO_TSP::calc_trail_limits_smooth(float solution_cost) const {
  (void)solution_cost;
  float tau_max_ = 1.0f;
  float denom = static_cast<float>(std::max<int32_t>(1, k));
  float tau_min_ = 1.0f / denom;

  // With Smooth MMAS, we linearly deposit pheromone

  // p = evaporated + deposit
  // With edges not in source solution, deposit = rho * tau_min
  // with edges in source solution, deposit = rho * tau_max
  return {tau_min_, tau_max_};
}

void MFACO_TSP::compute_probmat(const float *prior_ptr,
                                std::vector<float> &probmat) {
  probmat.resize((size_t)n * (size_t)k);

  const float beta = 1.0f;  // if you want classic eta^beta
  const float gamma = 1.0f; // strength of learned prior
  const float eps = EPS;

#pragma omp parallel for schedule(static)
  for (int32_t u = 0; u < n; ++u) {
    // ---- compute prior normalization stats for this row (u) ----
    float mean_z = 0.0f;
    float var_z = 0.0f;

    // ---- first pass: compute logits and max for stable exp ----
    float max_logit = -std::numeric_limits<float>::infinity();

    // store logits temporarily (stack or reuse probmat as scratch)
    // since k is small (32), a small local array is fine:
    float logits[MAX_CAND_LIST_SIZE];

    for (int32_t j = 0; j < k; ++j) {
      int32_t idx = u * k + j;

      float tau = pheromone_sparse[idx];
      float eta = heuristic_sparse[idx]; // currently 1/d or 1 if disabled

      // Base: alpha*log(tau)
      float logit = alpha * std::log(tau + eps);

      // Heuristic: beta*log(eta)  (if disabled, eta==1 => log=0)
      if (!disable_heuristic) {
        logit += beta * std::log(eta + eps);
      }

      // Prior logits: gamma * normalized_z
      if (prior_ptr) {
        float z = prior_ptr[idx];

        logit += gamma * z;
      }

      logits[j] = logit;
      if (logit > max_logit)
        max_logit = logit;
    }

    // ---- second pass: exp(logit - max) -> positive weights ----
    for (int32_t j = 0; j < k; ++j) {
      int32_t idx = u * k + j;
      float w = std::exp(logits[j] - max_logit);
      probmat[idx] = std::max(w, eps);
    }
  }
}

float MFACO_TSP::sample_ant_fast(const float *probmat, int32_t start_node,
                                 std::vector<int32_t> &route_out,
                                 int32_t &new_edges_out,
                                 std::vector<int32_t> &checklist,
                                 Xoshiro128Plus &rng, const float *prior) {
  // Initialize route as copy of source
  std::vector<int32_t> route = source_route;
  std::vector<int32_t> positions(n);
  for (int32_t i = 0; i < n; ++i) {
    positions[route[i]] = i;
  }

  std::vector<uint8_t> visited(n, 0);
  visited[start_node] = 1;
  int32_t visited_count = 1;

  checklist.clear();
  checklist.push_back(start_node);

  int32_t new_edges = 0;
  int32_t steps = 0;
  int32_t curr = start_node;

  while (true) {
    if (fixed_steps > 0) {
      if (steps >= fixed_steps)
        break;
    } else {
      if (new_edges >= min_new_edges || visited_count >= n)
        break;
    }
    if (visited_count >= n)
      break;
    int16_t pick_j = -1;
    uint64_t valid_mask = 0;
    auto [chosen, is_stoch, used_unif] = select_next_node(
        curr, &probmat[curr * k], visited.data(), rng, pick_j, valid_mask);

    // Check if this creates a new edge
    if (!contains_edge(curr, chosen, positions)) {
      ++new_edges;
      // Add endpoints to checklist
      if (std::find(checklist.begin(), checklist.end(), curr) ==
          checklist.end()) {
        checklist.push_back(curr);
      }
      if (std::find(checklist.begin(), checklist.end(), chosen) ==
          checklist.end()) {
        checklist.push_back(chosen);
      }
      int32_t chosen_pred = get_pred(chosen, route, positions);
      if (std::find(checklist.begin(), checklist.end(), chosen_pred) ==
          checklist.end()) {
        checklist.push_back(chosen_pred);
      }
    }

    // Relocate chosen to be successor of curr
    relocate_node(curr, chosen, route, positions);

    visited[chosen] = 1;
    ++visited_count;
    ++steps;
    curr = chosen;
  }

  new_edges_out = new_edges;

  // Apply local search if enabled
  if (use_local_search && !checklist.empty()) {
    if (nls && prior) {
      two_opt_nn(route, positions, checklist);

      float best_cost = get_route_cost(route);
      std::vector<int32_t> best_route = route;

      for (int t = 0; t < T_nls; ++t) {
        two_opt_nn_prior(route, positions, checklist, prior);
        two_opt_nn(route, positions, checklist);

        float current_cost = get_route_cost(route);
        if (current_cost < best_cost) {
          best_cost = current_cost;
          best_route = route;
        }
      }
      route = best_route;
    } else {
      two_opt_nn(route, positions, checklist);
    }
  }

  // Copy result
  route_out = route;
  return get_route_cost(route);
}

float MFACO_TSP::sample_ant_traced(const float *probmat, int32_t start_node,
                                   std::vector<int32_t> &route_out,
                                   std::vector<int32_t> &route_raw_out,
                                   float &cost_raw_out, int32_t &new_edges_out,
                                   std::vector<int32_t> &checklist,
                                   MFACOTrace &trace, Xoshiro128Plus &rng,
                                   float &logp_sum, float &survival_out,
                                   const float *prior) {
  trace.clear();
  trace.start_node = start_node;
  trace.reserve(min_new_edges * 2);

  // Initialize route as copy of source
  std::vector<int32_t> route = source_route;
  std::vector<int32_t> positions(n);
  for (int32_t i = 0; i < n; ++i) {
    positions[route[i]] = i;
  }

  std::vector<uint8_t> visited(n, 0);
  visited[start_node] = 1;
  int32_t visited_count = 1;

  checklist.clear();
  checklist.push_back(start_node);

  int32_t new_edges = 0;
  int32_t steps = 0;
  int32_t curr = start_node;
  logp_sum = 0.0f;

  while (true) {
    if (fixed_steps > 0) {
      if (steps >= fixed_steps)
        break;
    } else {
      if (new_edges >= min_new_edges || visited_count >= n)
        break;
    }
    if (visited_count >= n)
      break;
    int16_t pick_j = -1;
    uint64_t valid_mask = 0;
    auto [chosen, is_stoch, log_prob] = select_next_node(
        curr, &probmat[curr * k], visited.data(), rng, pick_j, valid_mask);

    if (is_stoch) {
      logp_sum += log_prob;
    }

    // Record decision
    trace.curr_nodes.push_back(curr);
    trace.chosen_nodes.push_back(chosen);
    trace.is_stochastic.push_back(is_stoch ? 1 : 0);
    trace.pick_j.push_back(pick_j);
    trace.valid_mask.push_back(valid_mask);

    // Check if this creates a new edge
    bool is_new = !contains_edge(curr, chosen, positions);
    trace.is_new_edge.push_back(is_new ? 1 : 0);

    if (is_new) {
      ++new_edges;
      if (std::find(checklist.begin(), checklist.end(), curr) ==
          checklist.end()) {
        checklist.push_back(curr);
      }
      if (std::find(checklist.begin(), checklist.end(), chosen) ==
          checklist.end()) {
        checklist.push_back(chosen);
      }
      int32_t chosen_pred = get_pred(chosen, route, positions);
      if (std::find(checklist.begin(), checklist.end(), chosen_pred) ==
          checklist.end()) {
        checklist.push_back(chosen_pred);
      }
    }

    // Relocate chosen to be successor of curr
    relocate_node(curr, chosen, route, positions);

    visited[chosen] = 1;
    ++visited_count;
    ++steps;
    curr = chosen;
  }

  new_edges_out = new_edges;

  // Capture raw results
  route_raw_out = route;
  cost_raw_out = get_route_cost(route);

  float best_cost = cost_raw_out;
  std::vector<int32_t> best_route = route;

  // Apply local search if enabled
  if (use_local_search && !checklist.empty()) {
    if (nls && prior) {
      two_opt_nn(route, positions, checklist);
      for (int t = 0; t < T_nls; ++t) {
        two_opt_nn_prior(route, positions, checklist, prior);
        two_opt_nn(route, positions, checklist);

        float current_cost = get_route_cost(route);
        if (current_cost < best_cost) {
          best_cost = current_cost;
          best_route = route;
        }
      }
      route = best_route;
    } else {
      two_opt_nn(route, positions, checklist);
    }
  }

  // Compute survival
  float sv_num = 0.0f;
  float sv_den = 0.0f;
  size_t trace_sz = trace.curr_nodes.size();
  for (size_t i = 0; i < trace_sz; ++i) {
    if (trace.pick_j[i] >= 0) {
      sv_den += 1.0f;
      // Check existence using local positions
      int32_t u = trace.curr_nodes[i];
      int32_t v = trace.chosen_nodes[i];
      // Bounds check to prevent out-of-bounds access
      if (u >= 0 && u < n && v >= 0 && v < n) {
        int32_t pos_u = positions[u];
        int32_t pos_v = positions[v];
        // Additional check: positions should be valid (>= 0 and < n)
        if (pos_u >= 0 && pos_u < n && pos_v >= 0 && pos_v < n) {
          int32_t diff = std::abs(pos_u - pos_v);
          if (diff == 1 || diff == (n - 1)) {
            sv_num += 1.0f;
          }
        }
      }
    }
  }
  survival_out = (sv_den > 0.5f) ? (sv_num / sv_den) : 0.0f;

  // Copy result
  route_out = route;
  return get_route_cost(route);
}

std::tuple<int32_t, bool, float>
MFACO_TSP::select_next_node(int32_t curr, const float *probmat_row,
                            const uint8_t *visited, Xoshiro128Plus &rng,
                            int16_t &out_pick_j, uint64_t &out_valid_mask) {
  // Build candidate list from nn_list
  int32_t cl[MAX_CAND_LIST_SIZE];
  float cl_prods[MAX_CAND_LIST_SIZE];
  int16_t cl_jidx[MAX_CAND_LIST_SIZE];
  int32_t cl_size = 0;
  float sum = 0.0f;
  float max_prod = 0.0f;
  int32_t max_node = curr;
  int16_t max_j = -1;

  out_valid_mask = 0;
  out_pick_j = -1;

  for (int32_t j = 0; j < k; ++j) {
    int32_t v = nn_list[curr * k + j];
    if (v >= 0 && !visited[v]) {
      if (j < 64) {
        out_valid_mask |= (1ULL << static_cast<uint64_t>(j));
      }
      float prod = probmat_row[j];
      cl[cl_size] = v;
      cl_prods[cl_size] = prod;
      cl_jidx[cl_size] = static_cast<int16_t>(j);
      sum += prod;
      if (prod > max_prod) {
        max_prod = prod;
        max_node = v;
        max_j = static_cast<int16_t>(j);
      }
      ++cl_size;
    }
  }

  bool is_stochastic = false;
  float log_prob = 0.0f;
  int32_t chosen = max_node;
  out_pick_j = max_j;
  const float EPS = 1e-9f; // Define EPS for numerical stability

  if (cl_size > 1) {
    is_stochastic = true;

    // Roulette wheel selection
    float r = rng.next_float() * sum;
    float cumsum = 0.0f;
    chosen = cl[cl_size - 1]; // Fallback
    out_pick_j = cl_jidx[cl_size - 1];
    float chosen_prod = cl_prods[cl_size - 1];

    for (int32_t i = 0; i < cl_size; ++i) {
      cumsum += cl_prods[i];
      if (r <= cumsum) {
        chosen = cl[i];
        out_pick_j = cl_jidx[i];
        chosen_prod = cl_prods[i];
        break;
      }
    }
    // Calculate log prob
    if (sum > EPS) {
      log_prob = std::log(chosen_prod / sum);
    }
  } else if (cl_size == 1) {
    // Deterministic choice from CL
    chosen = cl[0];
    out_pick_j = cl_jidx[0];
    log_prob = 0.0f; // prob = 1.0
  } else {
    // No candidate in nn_list, try backup list...
    // Fallback logic usually considered deterministic or outside of learned
    // prob scope in this specialized impl We treat it as prob=1.0 for now
    // (logp=0.0)
    bool found = false;
    for (int32_t j = 0; j < bl; ++j) {
      int32_t v = backup_list[curr * bl + j];
      if (v >= 0 && !visited[v]) {
        chosen = v;
        found = true;
        break;
      }
    }

    if (!found) {
      // Still nothing? Find closest unvisited globally
      float min_dist = 1e9f;
      for (int32_t v = 0; v < n; ++v) {
        if (!visited[v]) {
          float d = dist(curr, v);
          if (d < min_dist) {
            min_dist = d;
            chosen = v;
          }
        }
      }
    }
  }

  return {chosen, is_stochastic, log_prob};
}

float MFACO_TSP::relocate_node(int32_t target, int32_t node,
                               std::vector<int32_t> &route,
                               std::vector<int32_t> &positions) {
  if (node == target)
    return 0.0f;

  int32_t target_succ = get_succ(target, route, positions);
  if (target_succ == node)
    return 0.0f;

  int32_t node_pos = positions[node];
  int32_t target_pos = positions[target];

  int32_t node_pred = get_pred(node, route, positions);
  int32_t node_succ = get_succ(node, route, positions);

  // Calculate cost delta
  float cost_delta = (-dist(node_pred, node) - dist(node, node_succ) -
                      dist(target, target_succ) + dist(node_pred, node_succ) +
                      dist(target, node) + dist(node, target_succ));

  // Perform relocation
  if (target_pos < node_pos) {
    // Case 1: target is before node
    int32_t node_value = route[node_pos];
    for (int32_t i = node_pos; i > target_pos + 1; --i) {
      route[i] = route[i - 1];
    }
    route[target_pos + 1] = node_value;
    for (int32_t i = target_pos + 1; i <= node_pos; ++i) {
      positions[route[i]] = i;
    }
  } else {
    // Case 2: target is after node
    int32_t node_value = route[node_pos];
    for (int32_t i = node_pos; i < target_pos; ++i) {
      route[i] = route[i + 1];
    }
    route[target_pos] = node_value;
    for (int32_t i = node_pos; i <= target_pos; ++i) {
      positions[route[i]] = i;
    }
  }

  return cost_delta;
}

float MFACO_TSP::two_opt_nn(std::vector<int32_t> &route,
                            std::vector<int32_t> &positions,
                            std::vector<int32_t> &checklist) {
  int32_t changes_count = 0;
  float total_change = 0.0f;
  size_t checklist_pos = 0;

  while (checklist_pos < checklist.size()) {
    int32_t a = checklist[checklist_pos++];
    if (a < 0 || a >= n)
      continue;

    int32_t a_next = get_succ(a, route, positions);
    int32_t a_prev = get_pred(a, route, positions);

    float dist_a_to_next = dist(a, a_next);
    float dist_a_to_prev = dist(a_prev, a);

    float max_diff = 0.0f;
    int32_t best_move[4] = {-1, -1, -1, -1};

    // Check moves with a -> a_next edge
    for (int32_t j = 0; j < k; ++j) {
      int32_t b = nn_list[a * k + j];
      if (b < 0 || b >= n)
        break;

      float dist_ab = dist(a, b);
      if (dist_a_to_next > dist_ab) {
        int32_t b_next = get_succ(b, route, positions);
        float diff =
            dist_a_to_next + dist(b, b_next) - dist_ab - dist(a_next, b_next);
        if (diff > max_diff) {
          best_move[0] = a_next;
          best_move[1] = b_next;
          best_move[2] = a;
          best_move[3] = b;
          max_diff = diff;
        }
      } else {
        break;
      }
    }

    // Check moves with a_prev -> a edge
    for (int32_t j = 0; j < k; ++j) {
      int32_t b = nn_list[a * k + j];
      if (b < 0 || b >= n)
        break;

      float dist_ab = dist(a, b);
      if (dist_a_to_prev > dist_ab) {
        int32_t b_prev = get_pred(b, route, positions);
        float diff =
            dist_a_to_prev + dist(b_prev, b) - dist_ab - dist(a_prev, b_prev);
        if (diff > max_diff) {
          best_move[0] = a;
          best_move[1] = b;
          best_move[2] = a_prev;
          best_move[3] = b_prev;
          max_diff = diff;
        }
      } else {
        break;
      }
    }

    if (max_diff > 0) {
      flip_route_section(best_move[0], best_move[1], route, positions);
      ++changes_count;
      total_change -= max_diff;

      // if extend_ls, then add endpoints to checklist
      if (extend_ls) {
        for (int32_t i = 0; i < 4; ++i) {
          int32_t node = best_move[i];
          if (std::find(checklist.begin(), checklist.end(), node) ==
              checklist.end()) {
            checklist.push_back(node);
          }
        }
      }
    }
  }

  return total_change;
}

float MFACO_TSP::two_opt_nn_prior(std::vector<int32_t> &route,
                                  std::vector<int32_t> &positions,
                                  std::vector<int32_t> &checklist,
                                  const float *prior_ptr) {
  int32_t changes_count = 0;
  float total_gain = 0.0f;
  size_t checklist_pos = 0;

  auto get_prior = [&](int32_t u, int32_t v) -> float {
    int32_t idx = find_neighbor_index(u, v);
    if (idx >= 0) {
      return prior_ptr[u * k + idx];
    }
    return -1e9f; // Missing edge -> very low score
  };

  while (checklist_pos < checklist.size()) {
    int32_t a = checklist[checklist_pos++];
    if (a < 0 || a >= n)
      continue;

    int32_t a_next = get_succ(a, route, positions);
    int32_t a_prev = get_pred(a, route, positions);

    float prior_a_next = get_prior(a, a_next);
    float prior_a_prev = get_prior(a_prev, a);

    float max_gain = 0.0f;
    int32_t best_move[4] = {-1, -1, -1, -1};

    // Check moves with a -> a_next edge
    for (int32_t j = 0; j < k; ++j) {
      int32_t b = nn_list[a * k + j];
      if (b < 0 || b >= n)
        break;

      // Swap a->a_next and b->b_next with a->b and a_next->b_next
      // Gain = (new_prior) - (old_prior)
      // New: (a, b) + (a_next, b_next)
      // Old: (a, a_next) + (b, b_next)

      float prior_ab = get_prior(a, b);

      // We are maximizing sum of priors.
      // Current sum (partial): prior(a, a_next)
      // New sum (partial): prior(a, b)
      // Check if candidate edge (a,b) is even worth looking at?
      // Typically we blindly check all neighbors.

      int32_t b_next = get_succ(b, route, positions);
      float prior_b_bnext = get_prior(b, b_next);
      float prior_anext_bnext = get_prior(a_next, b_next);

      float current_score = prior_a_next + prior_b_bnext;
      float new_score = prior_ab + prior_anext_bnext;

      float gain = new_score - current_score;

      if (gain > max_gain) {
        best_move[0] = a_next;
        best_move[1] = b_next;
        best_move[2] = a;
        best_move[3] = b;
        max_gain = gain;
      }
    }

    // Check moves with a_prev -> a edge
    for (int32_t j = 0; j < k; ++j) {
      int32_t b = nn_list[a * k + j];
      if (b < 0 || b >= n)
        break;

      float prior_ab = get_prior(a, b);
      int32_t b_prev = get_pred(b, route, positions);
      float prior_bprev_b = get_prior(b_prev, b);
      float prior_aprev_bprev = get_prior(a_prev, b_prev);

      float current_score = prior_a_prev + prior_bprev_b;
      float new_score = prior_ab + prior_aprev_bprev;

      float gain = new_score - current_score;

      if (gain > max_gain) {
        best_move[0] = a;
        best_move[1] = b;
        best_move[2] = a_prev;
        best_move[3] = b_prev;
        max_gain = gain;
      }
    }

    if (max_gain > 0) {
      flip_route_section(best_move[0], best_move[1], route, positions);
      ++changes_count;
      total_gain += max_gain;

      // if extend_ls, then add endpoints to checklist
      if (extend_ls) {
        for (int32_t i = 0; i < 4; ++i) {
          int32_t node = best_move[i];
          if (std::find(checklist.begin(), checklist.end(), node) ==
              checklist.end()) {
            checklist.push_back(node);
          }
        }
      }
    }
  }

  return total_gain;
}

float MFACO_TSP::get_route_cost(const std::vector<int32_t> &route) const {
  float cost = 0.0f;
  for (int32_t i = 0; i < n - 1; ++i) {
    cost += dist(route[i], route[i + 1]);
  }
  cost += dist(route[n - 1], route[0]);
  return cost;
}

bool MFACO_TSP::contains_edge(int32_t a, int32_t b,
                              const std::vector<int32_t> &positions) const {
  // Need to use source positions for edge checking
  int32_t a_pos = source_positions[a];
  int32_t b_pos = source_positions[b];

  // Check if adjacent in source route
  int32_t diff = std::abs(a_pos - b_pos);
  return diff == 1 || diff == n - 1;
}

int32_t MFACO_TSP::get_succ(int32_t node, const std::vector<int32_t> &route,
                            const std::vector<int32_t> &positions) const {
  int32_t pos = positions[node];
  return route[(pos + 1) % n];
}

int32_t MFACO_TSP::get_pred(int32_t node, const std::vector<int32_t> &route,
                            const std::vector<int32_t> &positions) const {
  int32_t pos = positions[node];
  return route[(pos - 1 + n) % n];
}

void MFACO_TSP::flip_route_section(int32_t start_node, int32_t end_node,
                                   std::vector<int32_t> &route,
                                   std::vector<int32_t> &positions) {
  int32_t first = positions[start_node];
  int32_t last = positions[end_node];

  if (first > last) {
    std::swap(first, last);
  }

  int32_t segment_length = last - first;
  int32_t remaining_length = n - segment_length;

  if (segment_length <= remaining_length) {
    // Flip the segment
    int32_t left = first;
    int32_t right = last - 1;
    while (left < right) {
      std::swap(route[left], route[right]);
      ++left;
      --right;
    }
    for (int32_t i = first; i < last; ++i) {
      positions[route[i]] = i;
    }
  } else {
    // Flip the other segment (wrap around)
    int32_t first_adj = (first > 0) ? first - 1 : n - 1;
    int32_t last_adj = last % n;
    std::swap(first_adj, last_adj);

    int32_t l = first_adj;
    int32_t r = last_adj;
    int32_t i = 0;
    int32_t j = n - first_adj + last_adj + 1;

    while (i < j) {
      std::swap(route[l], route[r]);
      positions[route[l]] = l;
      positions[route[r]] = r;
      l = (l + 1) % n;
      r = (r - 1 + n) % n;
      ++i;
      --j;
    }
  }
}

} // namespace mfaco

namespace mfaco {

MFACO_CVRP::MFACO_CVRP(const float *coords_ptr, const float *demand_ptr,
                       int32_t n_, float capacity_, int32_t n_ants_,
                       int32_t cand_list_size, int32_t backup_list_size,
                       int32_t min_new_edges_, float decay, float alpha_,
                       float p_best_, bool use_local_search_,
                       bool disable_heuristic_, bool extend_ls_,
                       bool smooth_mmas_, int32_t fixed_steps_, bool nls_,
                       int32_t T_nls_)
    : n(n_), m(n_ - 1), n_ants(n_ants_), k(std::min(cand_list_size, n_ - 1)),
      bl(std::min(backup_list_size, std::max(0, n_ - 1 - k))),
      min_new_edges(min_new_edges_), fixed_steps(fixed_steps_), rho(decay),
      alpha(alpha_), p_best(p_best_), use_local_search(use_local_search_),
      disable_heuristic(disable_heuristic_), extend_ls(extend_ls_),
      smooth_mmas(smooth_mmas_), capacity(capacity_),
      capacity_int(static_cast<int64_t>(std::round(capacity_ * DEMAND_SCALE))),
      nls(nls_), T_nls(T_nls_), use_relocate(true), use_swap(true),
      use_2opt_star(true) {
  if (!coords_ptr || !demand_ptr) {
    throw std::runtime_error("coords_ptr and demand_ptr must not be null");
  }
  if (n < 2)
    throw std::runtime_error("n must be >= 2 (depot + at least one customer)");
  if (capacity <= 0)
    throw std::runtime_error("capacity must be > 0");
  capacity_int = (int64_t)std::round(capacity * DEMAND_SCALE);

  coords.resize(static_cast<size_t>(n) * 2);
  std::memcpy(coords.data(), coords_ptr,
              sizeof(float) * static_cast<size_t>(n) * 2);

  demand.resize(static_cast<size_t>(n));
  std::memcpy(demand.data(), demand_ptr,
              sizeof(float) * static_cast<size_t>(n));
  demand[0] = 0.0f; // enforce

  demand_int.resize(n);
  for (int32_t i = 0; i < n; ++i) {
    demand_int[i] = (int64_t)std::round(demand[i] * DEMAND_SCALE);
  }

  build_nn_lists();
  // if (!smooth_mmas || nls)
  //   build_nn_pos();
  build_heuristic();
  build_d0();

  // source_perm.resize(m); // REMOVED
  // best_perm.resize(m);   // REMOVED
  // source_positions.assign(n, -1); // REMOVED

  // Initialize routes
  source_route.reserve(n * 2);
  best_route.reserve(n * 2);

  build_initial_solution();

  auto [tmin, tmax] = smooth_mmas ? calc_trail_limits_smooth(source_cost)
                                  : calc_trail_limits_cl(source_cost);
  tau_min = tmin;
  tau_max = tmax;
  pheromone_sparse.assign(n * k, tau_max);

  rng_.seed(42);
}

void MFACO_CVRP::seed_rng(uint64_t seed) { rng_.seed(seed); }

void MFACO_CVRP::reset_timings() {
  time_ant = 0.0;
  time_ls = 0.0;
  time_split = 0.0;
}

// -------------------- distance --------------------
float MFACO_CVRP::dist(int32_t u, int32_t v) const {
  const size_t ou = static_cast<size_t>(u) * 2;
  const size_t ov = static_cast<size_t>(v) * 2;
  float dx = coords[ou] - coords[ov];
  float dy = coords[ou + 1] - coords[ov + 1];
  return std::sqrt(dx * dx + dy * dy);
}

// -------------------- NN lists (same as TSP) --------------------
void MFACO_CVRP::build_nn_lists() {
  nn_list.resize(n * k);
  backup_list.resize(n * bl);

  const int32_t total = k + bl;
  if (total <= 0)
    return;

  std::vector<Vec2d> pts(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const size_t off = static_cast<size_t>(i) * 2;
    pts[static_cast<size_t>(i)] =
        Vec2d{(double)coords[off], (double)coords[off + 1]};
  }

  KDTree shared_kdtree(pts, /*round_distances=*/false);

#pragma omp parallel default(none) shared(shared_kdtree, nn_list, backup_list) \
    firstprivate(n, k, bl, total)
  {
    KDTree kdtree = shared_kdtree;
    std::vector<int32_t> deleted_nodes;
    deleted_nodes.reserve(total + 8);

#pragma omp for schedule(static)
    for (int32_t u = 0; u < n; ++u) {
      deleted_nodes.clear();
      int32_t current_k = 0;
      int32_t current_bl = 0;

      // Force depot as first neighbor for customers
      if (u > 0 && k > 0) {
        nn_list[u * k + 0] = 0;
        current_k = 1;
      }

      // Search KDTree
      // We search a bit more than total to account for self/depot skips
      int32_t search_limit = total + 5;

      for (int32_t step = 0; step < search_limit; ++step) {
        if (current_k >= k && current_bl >= bl)
          break;

        uint32_t pt_idx = kdtree.nn_bottom_up(static_cast<uint32_t>(u));
        int32_t v = static_cast<int32_t>(pt_idx);

        kdtree.delete_point(pt_idx);
        deleted_nodes.push_back(v);

        if (v == u)
          continue; // Skip self
        if (u > 0 && v == 0)
          continue; // Skip depot for customers (already added)

        if (current_k < k) {
          nn_list[u * k + current_k] = v;
          current_k++;
        } else if (current_bl < bl) {
          backup_list[u * bl + current_bl] = v;
          current_bl++;
        }
      }

      // Undelete all deleted nodes
      for (int32_t v_del : deleted_nodes) {
        kdtree.undelete_point(static_cast<uint32_t>(v_del));
      }
    }
  }
}

// void MFACO_CVRP::build_nn_pos() { ... } REMOVED

void MFACO_CVRP::build_heuristic() {
  heuristic_sparse.resize(n * k);
  if (disable_heuristic) {
    std::fill(heuristic_sparse.begin(), heuristic_sparse.end(), 1.0f);
    return;
  }
  for (int32_t u = 0; u < n; ++u) {
    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_list[u * k + j];
      float d = dist(u, v);
      float d0 = dist(u, 0);
      float d1 = dist(0, v);
      // Savings heuristic
      d = d0 + d1 - d;
      heuristic_sparse[u * k + j] = d;
    }
  }
}

void MFACO_CVRP::build_d0() {
  d0.resize(n);
  for (int32_t v = 0; v < n; ++v) {
    d0[v] = dist(0, v);
  }
}

// -------------------- initial perm (greedy NN on customers, score by split)
// --------------------
void MFACO_CVRP::build_initial_solution() {
  // Greedy construction with capacity constraint
  // Result is a valid Tour with 0s
  float best_c = std::numeric_limits<float>::max();
  best_route.clear();

  int32_t num_starts = std::min(8, n);

  std::vector<int32_t> current_route;
  current_route.reserve(n * 2);
  std::vector<uint8_t> visited(n, 0);

  for (int32_t s = 0; s < num_starts; ++s) {
    current_route.clear();
    std::fill(visited.begin(), visited.end(), 0);
    visited[0] = 1;

    int32_t start_node = 1 + (s % (n - 1));

    // Start with explicit customer
    current_route.push_back(start_node);
    visited[start_node] = 1;

    int32_t curr = start_node;
    int64_t cur_cap = capacity_int - demand_int[start_node];
    float cost = dist(0, start_node);

    int32_t visited_cnt = 1;
    while (visited_cnt < n - 1) {
      int32_t best_nb = -1;
      float min_d = 1e9f;

      // Try NN list
      for (int32_t j = 0; j < k; ++j) {
        int32_t v = nn_list[curr * k + j];
        if (v > 0 && !visited[v]) {
          if (demand_int[v] <= cur_cap) {
            float d = dist(curr, v);
            if (d < min_d) {
              min_d = d;
              best_nb = v;
              break; // NN is sorted
            }
          }
        }
      }

      // If not found in NN or cap violation, try all
      if (best_nb == -1) {
        bool can_fit = false;
        for (int32_t v = 1; v < n; ++v) {
          if (!visited[v] && demand_int[v] <= cur_cap) {
            can_fit = true;
            if (dist(curr, v) < min_d) {
              min_d = dist(curr, v);
              best_nb = v;
            }
          }
        }
        if (!can_fit) {
          // Must return to depot
          cost += dist(curr, 0);
          current_route.push_back(0); // Depot
          curr = 0;
          cur_cap = capacity_int;
          continue;
        }
      }

      if (best_nb != -1) {
        visited[best_nb] = 1;
        visited_cnt++;
        current_route.push_back(best_nb);
        cost += dist(curr, best_nb);
        cur_cap -= demand_int[best_nb];
        curr = best_nb;
      }
    }
    // Return to depot
    cost += dist(curr, 0);
    // current_route implicitly ends at curr. Add final 0?
    // If we conform to [0, c1, ..., 0], let's fix it after.

    if (cost < best_c) {
      best_c = cost;
      best_route = current_route;
    }
  }

  // Canonicalize best_route to [0, ..., 0]
  if (!best_route.empty()) {
    std::vector<int32_t> full;
    full.reserve(best_route.size() + 2);
    full.push_back(0);
    for (int32_t x : best_route)
      full.push_back(x);
    full.push_back(0);
    best_route = full;

    source_cost = best_c;
    best_cost = best_c;
    source_route = best_route;
  }

  // Use simple min-max for initial
  tau_max = 1.0f / (rho * best_cost);
  tau_min = tau_max * 0.001f;
  std::fill(pheromone_sparse.begin(), pheromone_sparse.end(), tau_max);
}

// REMOVED split_dp, split_cost_fast, greedy_cost, decode_perm_to_route0

// -------------------- pheromone bounds (same formula as TSP)
// --------------------
std::pair<float, float>
MFACO_CVRP::calc_trail_limits_cl(float solution_cost) const {
  float tau_max_ = 1.0f / (solution_cost * (1.0f - rho) + EPS);
  float avg = static_cast<float>(std::max(2, k));
  float p = std::pow(p_best, 1.0f / avg);
  float tau_min_ =
      std::min(tau_max_, tau_max_ * (1.0f - p) / ((avg - 1.0f) * p + EPS));
  return {tau_min_, tau_max_};
}

std::pair<float, float>
MFACO_CVRP::calc_trail_limits_smooth(float solution_cost) const {
  (void)solution_cost;
  float tau_max_ = 1.0f;
  float denom = static_cast<float>(std::max<int32_t>(1, k));
  float tau_min_ = 1.0f / denom;
  return {tau_min_, tau_max_};
}

// -------------------- probmat (same as TSP) --------------------
void MFACO_CVRP::compute_probmat(const float *prior_ptr,
                                 std::vector<float> &probmat) {
  probmat.resize((size_t)n * (size_t)k);

  const float beta = 1.0f;  // if you want classic eta^beta
  const float gamma = 1.0f; // strength of learned prior
  const float eps = EPS;

#pragma omp parallel for schedule(static)
  for (int32_t u = 0; u < n; ++u) {
    // ---- compute prior normalization stats for this row (u) ----
    float mean_z = 0.0f;
    float var_z = 0.0f;

    // if (prior_ptr) {
    //   // mean
    //   for (int32_t j = 0; j < k; ++j) {
    //     float z = prior_ptr[u * k + j];
    //     // optional clamp for safety (avoid huge exp)
    //     z = std::max(-10.0f, std::min(10.0f, z));
    //     mean_z += z;
    //   }
    //   mean_z /= (float)k;

    //   // variance
    //   for (int32_t j = 0; j < k; ++j) {
    //     float z = prior_ptr[u * k + j];
    //     z = std::max(-10.0f, std::min(10.0f, z));
    //     float dz = z - mean_z;
    //     var_z += dz * dz;
    //   }
    //   var_z /= (float)k;
    // }
    // float std_z = (prior_ptr ? std::sqrt(var_z + 1e-6f) : 1.0f);

    // ---- first pass: compute logits and max for stable exp ----
    float max_logit = -std::numeric_limits<float>::infinity();

    // store logits temporarily (stack or reuse probmat as scratch)
    // since k is small (32), a small local array is fine:
    float logits[MAX_CAND_LIST_SIZE];

    for (int32_t j = 0; j < k; ++j) {
      int32_t idx = u * k + j;

      float tau = pheromone_sparse[idx];
      float eta = heuristic_sparse[idx];

      // Base: alpha*log(tau)
      float logit = alpha * std::log(tau + eps);

      // Heuristic: beta*log(eta)  (if disabled, eta==1 => log=0)
      if (!disable_heuristic) {
        logit += beta * std::log(eta + eps);
      }

      // Prior logits: gamma * normalized_z
      if (prior_ptr) {
        float z = prior_ptr[idx];
        // z = std::max(-10.0f, std::min(10.0f, z));
        // float z_norm = (z - mean_z) / std_z; // row-center + row-scale
        logit += gamma * z;
      }

      logits[j] = logit;
      if (logit > max_logit)
        max_logit = logit;
    }

    // ---- second pass: exp(logit - max) -> positive weights ----
    for (int32_t j = 0; j < k; ++j) {
      int32_t idx = u * k + j;
      float w = std::exp(logits[j] - max_logit);
      probmat[idx] = std::max(w, eps);
    }
  }
}

void MFACO_CVRP::sample(bool require_prob, const float *prior_ptr,
                        SampleResult &result, bool parallel_traced) {
  auto route_cost_euclid = [&](const std::vector<int32_t> &route) -> float {
    float c = 0.0f;
    if (route.size() < 2)
      return c;
    for (size_t i = 0; i + 1 < route.size(); ++i)
      c += dist(route[i], route[i + 1]);
    return c;
  };

  result.clear();
  result.costs.resize(n_ants);
  // CVRP solutions are represented as full depot-separated routes with
  // multiple 0s (e.g., 0 ... 0 ... 0).
  result.routes.resize(n_ants);
  result.decoded_routes.resize(n_ants);

  if (require_prob) {
    result.costs_raw.resize(n_ants);
    result.routes_raw.resize(n_ants);
    result.logps.resize(n_ants);
  }

  result.new_edges_count.resize(n_ants);
  result.edge_survival.resize(n_ants);

  std::vector<float> probmat;
  compute_probmat(prior_ptr, probmat);

  std::vector<int32_t> start_nodes(n_ants);
  for (int32_t a = 0; a < n_ants; ++a) {
    start_nodes[a] = 1 + (int32_t)rng_.next_uint((uint32_t)m);
  }

  std::vector<uint64_t> ant_seeds;
  auto ensure_ant_seeds = [&]() {
    if (!ant_seeds.empty())
      return;
    ant_seeds.resize((size_t)n_ants);
    for (int32_t a = 0; a < n_ants; ++a) {
      uint64_t hi = (uint64_t)rng_.next_u32();
      uint64_t lo = (uint64_t)rng_.next_u32();
      ant_seeds[(size_t)a] =
          (hi << 32) ^ lo ^ (0x9e3779b97f4a7c15ULL + (uint64_t)a);
    }
  };

  if (require_prob) {
    if (!parallel_traced) {
      result.traces.reserve(n_ants, n_ants * min_new_edges * 2);
      result.traces.starts.push_back(0);

      std::vector<int32_t> checklist;
      checklist.reserve(m);

      for (int32_t a = 0; a < n_ants; ++a) {
        // Trace construction into a decoded CVRP route (with depot zeros).
        result.decoded_routes[a].clear();
        std::vector<int32_t> route_raw_unused;

        MFACOTrace trace;
        trace.reserve(min_new_edges * 2);

        float logp_sum = 0.0f;
        int32_t mne_out = 0;

        float surv_out = 0.0f;
        (void)sample_ant_direct_traced(
            probmat.data(), start_nodes[a], result.decoded_routes[a],
            route_raw_unused, result.costs_raw[a], mne_out, checklist, trace,
            rng_, logp_sum, surv_out, prior_ptr);
        result.new_edges_count[a] = mne_out;
        result.edge_survival[a] = surv_out;
        result.logps[a] = logp_sum;

        // Canonicalize decoded route (some variants output a permutation only).
        if (result.decoded_routes[a].empty() ||
            result.decoded_routes[a].front() != 0)
          result.decoded_routes[a].insert(result.decoded_routes[a].begin(), 0);
        if (result.decoded_routes[a].back() != 0)
          result.decoded_routes[a].push_back(0);

        // Store the full depot-separated route and compute true CVRP cost.
        result.routes[a] = result.decoded_routes[a];
        result.costs[a] = route_cost_euclid(result.routes[a]);

        result.traces.start_nodes.push_back(trace.start_node);
        for (size_t i = 0; i < trace.curr_nodes.size(); ++i) {
          result.traces.curr_nodes.push_back(trace.curr_nodes[i]);
          result.traces.chosen_nodes.push_back(trace.chosen_nodes[i]);
          result.traces.is_stochastic.push_back(trace.is_stochastic[i]);
          result.traces.pick_j.push_back(trace.pick_j[i]);
          result.traces.valid_mask.push_back(trace.valid_mask[i]);
          result.traces.is_new_edge.push_back(trace.is_new_edge[i]);
        }
        result.traces.starts.push_back(
            (int32_t)result.traces.curr_nodes.size());
      }
    } else {
      ensure_ant_seeds();
      std::vector<MFACOTrace> traces_per_ant((size_t)n_ants);

#pragma omp parallel
      {
        std::vector<int32_t> checklist;
        checklist.reserve(m);

#pragma omp for schedule(static, 1)
        for (int32_t a = 0; a < n_ants; ++a) {
          MFACOTrace &trace = traces_per_ant[(size_t)a];
          trace.reserve(min_new_edges * 2);
          Xoshiro128Plus rng_local;
          rng_local.seed(ant_seeds[(size_t)a]);

          float logp_sum = 0.0f;
          int32_t mne_out = 0;

          float surv_out = 0.0f;
          (void)sample_ant_direct_traced(
              probmat.data(), start_nodes[a], result.routes[a],
              result.routes_raw[a], result.costs_raw[a], mne_out, checklist,
              trace, rng_local, logp_sum, surv_out, prior_ptr);
          result.new_edges_count[a] = mne_out;
          result.new_edges_count[a] = mne_out;
          result.edge_survival[a] = surv_out;
          result.logps[a] = logp_sum;

          // Canonicalize route format to depot-separated representation.
          if (result.routes[a].empty() || result.routes[a].front() != 0)
            result.routes[a].insert(result.routes[a].begin(), 0);
          if (result.routes[a].back() != 0)
            result.routes[a].push_back(0);

          if (!result.routes_raw[a].empty()) {
            if (result.routes_raw[a].front() != 0)
              result.routes_raw[a].insert(result.routes_raw[a].begin(), 0);
            if (result.routes_raw[a].back() != 0)
              result.routes_raw[a].push_back(0);
          }

          // Keep decoded_routes in sync for legacy return_decoded.
          result.decoded_routes[a] = result.routes[a];
          result.costs[a] = route_cost_euclid(result.routes[a]);
        }
      }

      // Merge traces in ant index order
      result.traces.clear();
      result.traces.starts.resize((size_t)n_ants + 1);
      result.traces.start_nodes.resize((size_t)n_ants);
      result.traces.starts[0] = 0;
      for (int32_t a = 0; a < n_ants; ++a) {
        const MFACOTrace &t = traces_per_ant[(size_t)a];
        result.traces.start_nodes[(size_t)a] = t.start_node;
        result.traces.starts[(size_t)a + 1] =
            result.traces.starts[(size_t)a] + (int32_t)t.curr_nodes.size();
      }
      int32_t total = result.traces.starts[(size_t)n_ants];
      result.traces.curr_nodes.resize((size_t)total);
      result.traces.chosen_nodes.resize((size_t)total);
      result.traces.is_stochastic.resize((size_t)total);
      result.traces.pick_j.resize((size_t)total);
      result.traces.valid_mask.resize((size_t)total);
      result.traces.is_new_edge.resize((size_t)total);

      for (int32_t a = 0; a < n_ants; ++a) {
        const MFACOTrace &t = traces_per_ant[(size_t)a];
        int32_t off = result.traces.starts[(size_t)a];
        for (size_t i = 0; i < t.curr_nodes.size(); ++i) {
          result.traces.curr_nodes[(size_t)off + i] = t.curr_nodes[i];
          result.traces.chosen_nodes[(size_t)off + i] = t.chosen_nodes[i];
          result.traces.is_stochastic[(size_t)off + i] = t.is_stochastic[i];
          result.traces.pick_j[(size_t)off + i] = t.pick_j[i];
          result.traces.valid_mask[(size_t)off + i] = t.valid_mask[i];
          result.traces.is_new_edge[(size_t)off + i] = t.is_new_edge[i];
        }
      }
    }
  } else {
    // Fast mode: parallel
    ensure_ant_seeds();
#pragma omp parallel
    {
      std::vector<int32_t> checklist;
      checklist.reserve(m);

#pragma omp for schedule(static, 1)
      for (int32_t a = 0; a < n_ants; ++a) {
        // Build a candidate solution; we will canonicalize to a full CVRP route
        // with depot (0) separators and compute the true CVRP cost.
        Xoshiro128Plus rng_local;
        rng_local.seed(ant_seeds[(size_t)a]);

        // Pass dummy perm_out if removed from signature? I changed it in
        // header. The implementation of sample_ant_direct must match header.
        // Let's assume implementation does NOT take perm_out anymore.

        // Write whatever the sampler returns into routes[a] first. Some
        // sampler variants return only a customer permutation (no depot zeros).
        // We handle both and ensure we end with a depot-separated route.
        (void)sample_ant_direct(probmat.data(), start_nodes[a],
                                result.routes[a], result.new_edges_count[a],
                                checklist, rng_local, prior_ptr);

        // Ensure canonical start/end depot.
        if (result.routes[a].empty() || result.routes[a].front() != 0)
          result.routes[a].insert(result.routes[a].begin(), 0);
        if (result.routes[a].back() != 0)
          result.routes[a].push_back(0);

        // Keep decoded_routes in sync for legacy return_decoded.
        result.decoded_routes[a] = result.routes[a];

        // Recompute true CVRP cost over the depot-separated route.
        result.costs[a] = route_cost_euclid(result.routes[a]);
      }
    }
  }
}

// -------------------- update pheromone: deposit on decoded VRP edges
// --------------------
void MFACO_CVRP::update_pheromone(const std::vector<int32_t> &best_route_in,
                                  float new_best_cost) {
  // Update global best
  if (new_best_cost < best_cost) {
    best_cost = new_best_cost;
    best_route = best_route_in;
  }

  // Update trail limits based on global best
  auto [tmin, tmax] = smooth_mmas ? calc_trail_limits_smooth(best_cost)
                                  : calc_trail_limits_cl(best_cost);
  tau_min = tmin;
  tau_max = tmax;

  // Precompute positions for O(1) in-route check
  // For CVRP, best_route_in has depot 0 multiple times.
  // We map each customer node (1..n-1) to its index in best_route_in.
  int32_t R = (int32_t)best_route_in.size();
  std::vector<int32_t> pos(n, -1);
  for (int32_t i = 0; i < R; ++i) {
    int32_t v = best_route_in[i];
    if (v != 0) {
      pos[v] = i;
    }
  }

  auto in_route_edge = [&](int32_t u, int32_t v) -> bool {
    // Edge (u, v). If both are customers, they apply adjacency in best_route_in
    if (u != 0 && v != 0) {
      int32_t pu = pos[u];
      int32_t pv = pos[v];
      return std::abs(pu - pv) == 1;
    }

    // One is depot 0.
    if (u == 0 && v == 0)
      return false; // Loop (0,0) not relevant

    int32_t cust = (u != 0) ? u : v;
    int32_t p = pos[cust];
    // Check neighbors of cust in best_route_in
    if (p > 0 && best_route_in[p - 1] == 0)
      return true;
    if (p < R - 1 && best_route_in[p + 1] == 0)
      return true;

    return false;
  };

  const float decay_factor = 1.0f - rho;
  const float deposit = (!smooth_mmas) ? (1.0f / (new_best_cost + EPS)) : 0.0f;

#pragma omp parallel for schedule(static)
  for (int32_t u = 0; u < n; ++u) {
    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_list[u * k + j];
      float &tau = pheromone_sparse[u * k + j];
      bool is_in = in_route_edge(u, v);

      if (smooth_mmas) {
        float target = is_in ? tau_max : tau_min;
        tau = decay_factor * tau + rho * target;
      } else {
        tau *= decay_factor;
        if (is_in) {
          tau += deposit;
        }
        tau = std::max(tau_min, std::min(tau_max, tau));
      }
    }
  }

  // Update source solution (unconditionally, to track iteration best)
  source_route = best_route_in;
  source_cost = new_best_cost;
}

// REMOVED two_opt_nn_prior

// -------------------- Inter-route LS --------------------

std::vector<std::vector<int32_t>> MFACO_CVRP::initial_routes_from_perm(
    const std::vector<int32_t> &solution) const {
  // Parse full route [0, c1..c2, 0, c3..c4, 0] into vector of routes
  // Each route should be [0, c1..c2, 0] (as expected by LS helpers)
  std::vector<std::vector<int32_t>> routes;

  // Solution should start with 0? sample_ant_direct output starts with customer
  // usually if called with start_node, but build_initial_solution adds 0s.
  // Let's robustly parse.

  std::vector<int32_t> current;
  bool in_route = false;

  // Handling standard [0, ..., 0] form
  for (int32_t node : solution) {
    if (node == 0) {
      if (in_route) {
        current.push_back(0); // Close route
        routes.push_back(current);
        current.clear();
        in_route = false;
      }
      // Start new route potentially?
      // if next is customer, yes.
    } else {
      if (!in_route) {
        current.push_back(0); // Start new route
        in_route = true;
      }
      current.push_back(node);
    }
  }
  if (!current
           .empty()) { // Should imply last node wasn't 0 or route wasn't closed
    current.push_back(0);
    routes.push_back(current);
  }
  return routes;
}

void MFACO_CVRP::routes_to_perm(const std::vector<std::vector<int32_t>> &routes,
                                std::vector<int32_t> &solution,
                                std::vector<int32_t> &positions) {
  solution.clear();
  solution.reserve(n * 2); // Roughly
  positions.assign(
      n, -1); // Not really useful for full route? but might be used by LS?
  // Actually node_pos in ls_* is passed in directly. `positions` argument here
  // might be unused or legacy.

  // Flatten routes: [0, r1, 0, r2, 0 ...]
  // Routes are [0, c.., 0]
  // We want to merge them. R1=[0, A, 0], R2=[0, B, 0] -> [0, A, 0, B, 0]

  bool first = true;
  int32_t idx = 0;

  for (const auto &r : routes) {
    if (r.size() < 2)
      continue; // Invalid

    // If not first, skip the leading 0 (it overlaps with previous trailing 0)?
    // OR strictly concat: [0, A, 0, 0, B, 0].
    // ACO_CVRP usually merges: [0, A, 0, B, 0].

    size_t start = 0;
    if (!first) {
      if (r[0] == 0)
        start = 1; // Skip leading 0
    }

    for (size_t i = start; i < r.size(); ++i) {
      int32_t node = r[i];
      solution.push_back(node);
      // Position map for customers
      if (node > 0) {
        // positions[node] = ...? In flattened array?
        // positions vector mainly used for Permutation logic.
        // Maybe safe to ignore or set.
      }
    }
    first = false;
  }
  // Ensure starts with 0? Yes if first route started with 0.
}

// ============================================================================
// Intra-Route Local Search (2-opt)
// ============================================================================

float MFACO_CVRP::intra_route_ls(std::vector<int32_t> &route,
                                 std::vector<int32_t> &checklist) {
  float total_improvement = 0.0f;
  // Build route structure: identify which route each node belongs to
  std::vector<int32_t> node_route(n, -1);
  std::vector<std::vector<int32_t>> routes;
  std::vector<int32_t> current;

  for (size_t i = 0; i < route.size(); ++i) {
    int32_t node = route[i];
    if (node == 0) {
      if (!current.empty()) {
        int32_t route_idx = (int32_t)routes.size();
        for (int32_t c : current) {
          node_route[c] = route_idx;
        }
        routes.push_back(current);
        current.clear();
      }
    } else {
      current.push_back(node);
    }
  }

  if (routes.empty())
    return 0.0f;

  // Build positions within each route
  std::vector<int32_t> pos_in_route(n, -1);
  for (size_t r = 0; r < routes.size(); ++r) {
    for (size_t i = 0; i < routes[r].size(); ++i) {
      pos_in_route[routes[r][i]] = (int32_t)i;
    }
  }

  // Track nodes already in checklist to avoid duplicates
  std::vector<uint8_t> in_checklist_local(n, 0);
  for (int32_t node : checklist) {
    if (node > 0 && node < n) {
      in_checklist_local[node] = 1;
    }
  }

  // Process checklist - focused 2-opt
  size_t checklist_pos = 0;
  while (checklist_pos < checklist.size()) {
    int32_t a = checklist[checklist_pos++];
    if (a <= 0 || a >= n)
      continue;

    int32_t r = node_route[a];
    if (r < 0 || r >= (int32_t)routes.size())
      continue;

    auto &seq = routes[r];
    int32_t size = (int32_t)seq.size();
    if (size < 2)
      continue;

    int32_t a_pos = pos_in_route[a];
    if (a_pos < 0)
      continue;

    // Get neighbors of a in the route (including depot endpoints)
    int32_t a_prev = (a_pos > 0) ? seq[a_pos - 1] : 0;
    int32_t a_next = (a_pos < size - 1) ? seq[a_pos + 1] : 0;

    float dist_a_prev = dist(a_prev, a);
    float dist_a_next = dist(a, a_next);

    float max_diff = 0.0f;
    int32_t best_i = -1, best_j = -1;

    // Check 2-opt moves involving edge (a_prev, a)
    for (int32_t jj = 0; jj < k; ++jj) {
      int32_t b = nn_list[a * k + jj];
      if (b <= 0 || b >= n)
        continue;
      if (node_route[b] != r)
        continue; // Must be same route

      float dist_ab = dist(a, b);
      if (dist_a_prev <= dist_ab)
        break; // NN list is sorted

      int32_t b_pos = pos_in_route[b];
      if (b_pos < 0 || b_pos == a_pos)
        continue;

      // Get b's predecessor
      int32_t b_prev = (b_pos > 0) ? seq[b_pos - 1] : 0;

      // 2-opt: reverse segment between a and b
      // Current: ...-a_prev-a-...-b_prev-b-...
      // New:     ...-a_prev-b_prev-...-a-b-...
      if (b_pos > a_pos) {
        // Reverse [a, b_prev]
        float d_old = dist_a_prev + dist(b_prev, b);
        float d_new = dist(a_prev, b_prev) + dist_ab;
        float diff = d_old - d_new;
        if (diff > max_diff) {
          max_diff = diff;
          best_i = a_pos;
          best_j = b_pos;
        }
      } else {
        // Reverse [b, a_prev]
        float d_old = dist(b_prev, b) + dist_a_prev;
        float d_new = dist(b_prev, a) + dist(b, a_prev);
        float diff = d_old - d_new;
        if (diff > max_diff) {
          max_diff = diff;
          best_i = b_pos;
          best_j = a_pos;
        }
      }
    }

    // Check 2-opt moves involving edge (a, a_next)
    for (int32_t jj = 0; jj < k; ++jj) {
      int32_t b = nn_list[a * k + jj];
      if (b <= 0 || b >= n)
        continue;
      if (node_route[b] != r)
        continue;

      float dist_ab = dist(a, b);
      if (dist_a_next <= dist_ab)
        break;

      int32_t b_pos = pos_in_route[b];
      if (b_pos < 0 || b_pos == a_pos)
        continue;

      int32_t b_next = (b_pos < size - 1) ? seq[b_pos + 1] : 0;

      if (b_pos > a_pos) {
        // Reverse [a_next, b]
        float d_old = dist_a_next + dist(b, b_next);
        float d_new = dist_ab + dist(a_next, b_next);
        float diff = d_old - d_new;
        if (diff > max_diff) {
          max_diff = diff;
          best_i = a_pos + 1;
          best_j = b_pos + 1;
        }
      }
    }

    // Apply best move if found
    if (max_diff > 1e-6f && best_i >= 0 && best_j > best_i) {
      std::reverse(seq.begin() + best_i, seq.begin() + best_j);
      total_improvement += max_diff;

      // Update positions
      for (int32_t i = best_i; i < best_j; ++i) {
        pos_in_route[seq[i]] = i;
      }

      // Add affected nodes to checklist if extend_ls
      if (extend_ls) {
        for (int32_t i = std::max(0, best_i - 1);
             i < std::min(size, best_j + 1); ++i) {
          int32_t node = seq[i];
          if (node > 0 && node < n && !in_checklist_local[node]) {
            checklist.push_back(node);
            in_checklist_local[node] = 1;
          }
        }
      }
    }
  }

  // Reconstruct route from modified segments
  std::vector<int32_t> new_route;
  new_route.reserve(route.size());
  for (const auto &seg : routes) {
    new_route.push_back(0);
    for (int32_t c : seg) {
      new_route.push_back(c);
    }
  }
  new_route.push_back(0);
  route = new_route;
  return total_improvement;
}

// ============================================================================
// Optimized Inter-Route Local Search (Linked List + DLB + O(1) Delta)
// ============================================================================

float MFACO_CVRP::inter_route_ls_optimized(std::vector<int32_t> &perm,
                                           std::vector<int32_t> &positions,
                                           std::vector<int32_t> &checklist,
                                           std::vector<uint8_t> &in_checklist) {
  float total_improvement = 0.0f;
  // 1. Initialization (Thread-Local Vectors)
  std::vector<int32_t> next_node(2 * n);
  std::vector<int32_t> prev_node(2 * n);
  std::vector<int32_t> node_route(2 * n);
  std::vector<int64_t> cum_demand(2 * n);
  // We'll resize route_loads after finding num_routes
  std::vector<int64_t> route_loads;
  std::vector<bool> dlb(n, false);

  // Helpers
  auto dist = [&](int32_t u, int32_t v) {
    int32_t ru = (u >= n) ? 0 : u;
    int32_t rv = (v >= n) ? 0 : v;
    float dx = coords[ru * 2] - coords[rv * 2];
    float dy = coords[ru * 2 + 1] - coords[rv * 2 + 1];
    return std::sqrt(dx * dx + dy * dy);
  };

  auto touch = [&](int32_t u) {
    if (u < n) {
      dlb[u] = false;
      if (!in_checklist[u]) {
        checklist.push_back(u);
        in_checklist[u] = 1;
      }
    }
  };

  auto update_route_state = [&](int32_t start_node, int32_t route_id) {
    int32_t curr = start_node;
    if (curr < n)
      return;

    cum_demand[curr] = 0;
    node_route[curr] = route_id;
    int64_t load = 0;

    curr = next_node[curr];
    while (curr < n) {
      load += demand_int[curr];
      cum_demand[curr] = load;
      node_route[curr] = route_id;
      curr = next_node[curr];
    }
    route_loads[route_id] = load;
  };

  // Focused LS: only process nodes in checklist (no fallback to all nodes)
  if (checklist.empty()) {
    return 0.0f; // Nothing to do
  }

  // Get current routes
  auto routes =
      initial_routes_from_perm(perm); // Uses "perm" which is now full route
  int32_t num_routes = (int32_t)routes.size();

  if (num_routes > n) {
    // Just in case, though guaranteed by split logic usually
    next_node.resize(n + num_routes);
    prev_node.resize(n + num_routes);
    node_route.resize(n + num_routes);
    cum_demand.resize(n + num_routes);
  }
  route_loads.resize(num_routes);

  // Build Linked List from Routes
  for (int r = 0; r < num_routes; ++r) {
    const auto &current_route = routes[r];
    // current_route is [0, c1, ..., ck, 0]

    int32_t depot = n + r;
    int32_t prev = depot;
    int64_t load = 0;

    node_route[depot] = r;
    cum_demand[depot] = 0;

    // Iterate customers (skip first 0, last 0)
    for (size_t i = 1; i < current_route.size() - 1; ++i) {
      int32_t u = current_route[i];
      next_node[prev] = u;
      prev_node[u] = prev;
      node_route[u] = r;
      load += demand_int[u];
      cum_demand[u] = load;
      prev = u;
    }
    // Close loop to depot
    next_node[prev] = depot;
    prev_node[depot] = prev;
    route_loads[r] = load;
  }

  const float EPS = 1e-5f; // Tighten EPS

  // 2. Main Loop
  int32_t head = 0;
  while (head < (int32_t)checklist.size()) {
    int32_t u = checklist[head++];
    in_checklist[u] = 0;

    if (dlb[u])
      continue;

    bool improved = false;
    int32_t r_u = node_route[u];

    // Check neighbors
    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_list[u * k + j];
      if (v == 0)
        continue;

      int32_t r_v = node_route[v];

      // Pruning removed (was unsafe for Relocate/Swap involving prev edges)
      int32_t next_u = next_node[u];
      int32_t next_v = next_node[v];
      int32_t prev_v = prev_node[v];
      /*
      if (dist(u, v) > dist(u, next_u) + dist(v, next_v) + EPS) {
         continue;
      }
      */

      // A. Relocate u after v
      if (use_relocate && r_u != r_v) {
        if (route_loads[r_v] + demand_int[u] <= capacity_int) {
          // Case 1: Insert After v
          int32_t prev_u = prev_node[u];
          float delta = dist(prev_u, next_u) + dist(v, u) + dist(u, next_v) -
                        dist(prev_u, u) - dist(u, next_u) - dist(v, next_v);

          if (delta < -EPS) {
            // Unlink u
            next_node[prev_u] = next_u;
            prev_node[next_u] = prev_u;
            // Link u after v
            int32_t old_next_v = next_node[v];
            next_node[v] = u;
            prev_node[u] = v;
            next_node[u] = old_next_v;
            prev_node[old_next_v] = u;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            touch(u);
            touch(v);
            touch(prev_u);
            touch(next_u);
            touch(old_next_v);
            improved = true;
            total_improvement -= delta; // delta is negative
            break;
          }

          // Case 2: Insert Before v (After prev_v)
          // Effectively: Relocate u after prev_v
          // Only possible if we didn't do Case 1 (improved=false)
          // But r_v is same. prev_v might be depot.
          // Check if prev_v is actually valid target (it is, since r_prev_v ==
          // r_v != r_u)

          float delta2 = dist(prev_u, next_u) + dist(prev_v, u) + dist(u, v) -
                         dist(prev_u, u) - dist(u, next_u) - dist(prev_v, v);

          if (delta2 < -EPS) {
            // Unlink u
            next_node[prev_u] = next_u;
            prev_node[next_u] = prev_u;
            // Link u after prev_v
            // current next of prev_v is v.
            next_node[prev_v] = u;
            prev_node[u] = prev_v;
            next_node[u] = v;
            prev_node[v] = u;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            // Touches
            touch(u);
            touch(prev_v);
            touch(v);
            touch(prev_u);
            touch(next_u);
            improved = true;
            total_improvement -= delta2;
            break;
          }
        }
      }

      // B. Swap u, v
      if (use_swap && r_u != r_v) {
        int64_t load_u_new = route_loads[r_u] - demand_int[u] + demand_int[v];
        int64_t load_v_new = route_loads[r_v] - demand_int[v] + demand_int[u];

        if (load_u_new <= capacity_int && load_v_new <= capacity_int) {
          int32_t prev_u = prev_node[u];
          int32_t prev_v = prev_node[v];
          float delta = dist(prev_u, v) + dist(v, next_u) + dist(prev_v, u) +
                        dist(u, next_v) - dist(prev_u, u) - dist(u, next_u) -
                        dist(prev_v, v) - dist(v, next_v);
          if (delta < -EPS) {
            int32_t nu = next_node[u], pu = prev_node[u];
            int32_t nv = next_node[v], pv = prev_node[v];
            next_node[pu] = v;
            prev_node[nu] = v;
            next_node[pv] = u;
            prev_node[nv] = u;
            next_node[u] = nv;
            prev_node[u] = pv;
            next_node[v] = nu;
            prev_node[v] = pu;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            touch(u);
            touch(v);
            touch(pu);
            touch(nu);
            touch(pv);
            touch(nv);
            improved = true;
            total_improvement -= delta;
            break;
          }
        }
      }

      // C. 2-Opt*
      if (use_2opt_star && r_u != r_v) {
        int64_t head_u = cum_demand[u];
        int64_t tail_u = route_loads[r_u] - head_u;
        int64_t head_v = cum_demand[v];
        int64_t tail_v = route_loads[r_v] - head_v;

        if (head_u + tail_v <= capacity_int &&
            head_v + tail_u <= capacity_int) {
          float delta = dist(u, next_v) + dist(v, next_u) - dist(u, next_u) -
                        dist(v, next_v);
          if (delta < -EPS) {
            int32_t nu = next_node[u];
            int32_t nv = next_node[v];
            next_node[u] = nv;
            prev_node[nv] = u;
            next_node[v] = nu;
            prev_node[nu] = v;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            touch(u);
            touch(v);
            touch(nu);
            touch(nv);
            improved = true;
            total_improvement -= delta;
            break;
          }
        }
      }
    }
    if (!improved)
      dlb[u] = true;
  }

  // 3. Reconstruct depot-separated route (not permutation)
  perm.clear();
  perm.reserve(m + num_routes + 1);
  for (int r = 0; r < num_routes; ++r) {
    int32_t depot_node = n + r;
    int32_t curr = next_node[depot_node];
    // Skip empty routes
    if (curr >= n)
      continue;
    perm.push_back(0); // Start depot
    while (curr < n) {
      perm.push_back(curr);
      curr = next_node[curr];
    }
  }
  if (!perm.empty() && perm.back() != 0)
    perm.push_back(0); // End depot

  return total_improvement;
}

// ============================================================================
// ACO_TSP Implementation
// ============================================================================

ACO_TSP::ACO_TSP(const float *coords_ptr, int32_t n_, int32_t n_ants_,
                 int32_t cand_list_size, float decay, float alpha_, float beta_,
                 float p_best_, bool min_max_)
    : n(n_), n_ants(n_ants_), k(std::min(cand_list_size, n_ - 1)),
      rho(1.0f - decay), alpha(alpha_), beta(beta_), p_best(p_best_),
      min_max(min_max_), elitist(false) {
  if (coords_ptr == nullptr) {
    throw std::runtime_error("coords_ptr must not be null");
  }

  coords.resize(static_cast<size_t>(n) * 2);
  std::memcpy(coords.data(), coords_ptr,
              sizeof(float) * static_cast<size_t>(n) * 2);

  build_nn_lists();
  build_heuristic();

  best_route.resize(n);
  // Initial random best
  for (int i = 0; i < n; ++i)
    best_route[i] = i;
  best_cost = std::numeric_limits<float>::max();

  // Initialize pheromone
  float initial_tau = 1.0f;
  if (min_max) {
    tau_max = 1.0f;
    tau_min = 0.0001f;
    initial_tau = tau_max;
  } else {
    initial_tau = 1.0f;
    tau_min = 0.0f;
    tau_max = std::numeric_limits<float>::max();
  }

  pheromone.assign(n * k, initial_tau);
  rng_.seed(1234);
}

void ACO_TSP::seed_rng(uint64_t seed) { rng_.seed(seed); }

void ACO_TSP::build_nn_lists() {
  nn_list.resize(n * k);

  std::vector<Vec2d> pts(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const size_t off = static_cast<size_t>(i) * 2;
    pts[i] = Vec2d{static_cast<double>(coords[off + 0]),
                   static_cast<double>(coords[off + 1])};
  }

  KDTree kdtree(pts, false); // no rounding

#pragma omp parallel for schedule(static)
  for (int32_t u = 0; u < n; ++u) {
    KDTree local_tree = kdtree;

    for (int32_t j = 0; j < k; ++j) {
      uint32_t pt_idx = local_tree.nn_bottom_up(static_cast<uint32_t>(u));
      nn_list[u * k + j] = static_cast<int32_t>(pt_idx);
      local_tree.delete_point(pt_idx);
    }
  }
}

void ACO_TSP::build_heuristic() {
  heuristic.resize(n * k);
  for (int32_t u = 0; u < n; ++u) {
    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_list[u * k + j];
      float d = dist(u, v);
      heuristic[u * k + j] = (d > EPS) ? (1.0f / d) : 1.0f;
    }
  }
}

float ACO_TSP::get_route_cost(const std::vector<int32_t> &route) const {
  float cost = 0.0f;
  for (int32_t i = 0; i < n; ++i) {
    cost += dist(route[i], route[(i + 1) % n]);
  }
  return cost;
}

std::pair<float, float> ACO_TSP::calc_trail_limits(float solution_cost) const {
  if (!min_max)
    return {0.0f, std::numeric_limits<float>::max()};

  float max_t = 1.0f / (rho * solution_cost);
  float avg = static_cast<float>(k);
  float p = std::pow(p_best, 1.0f / n);

  float min_t = max_t * (1.0f - p) / ((avg - 1.0f) * p + EPS);
  if (min_t > max_t)
    min_t = max_t;

  return {min_t, max_t};
}

void ACO_TSP::compute_probmat(const float *prior, std::vector<float> &probmat) {
  probmat.resize(n * k);

#pragma omp parallel for schedule(static)
  for (int32_t u = 0; u < n; ++u) {
    float max_logit = -std::numeric_limits<float>::infinity();
    float logits[MAX_CAND_LIST_SIZE];

    for (int32_t j = 0; j < k; ++j) {
      int32_t idx = u * k + j;
      float tau = pheromone[idx];
      float eta = heuristic[idx];

      float logit = alpha * std::log(tau + EPS) + beta * std::log(eta + EPS);

      if (prior) {
        logit += prior[idx];
      }

      logits[j] = logit;
      if (logit > max_logit)
        max_logit = logit;
    }

    for (int32_t j = 0; j < k; ++j) {
      probmat[u * k + j] = std::exp(logits[j] - max_logit);
    }
  }
}

float ACO_TSP::sample_ant_constructive(const float *probmat, int32_t start_node,
                                       std::vector<int32_t> &route_out,
                                       Xoshiro128Plus &rng, bool require_prob,
                                       float &logp_out, MFACOTrace *trace) {
  route_out.resize(n);
  std::vector<uint8_t> visited(n, 0);

  int32_t curr = start_node;
  route_out[0] = curr;
  visited[curr] = 1;

  float log_p_total = 0.0f;

  // Initialize trace if provided
  if (trace) {
    trace->clear();
    trace->start_node = start_node;
    trace->reserve(n);
  }

  for (int32_t step = 1; step < n; ++step) {
    float weights[MAX_CAND_LIST_SIZE];
    int32_t candidates[MAX_CAND_LIST_SIZE];
    int32_t cand_j[MAX_CAND_LIST_SIZE]; // j index in nn_list
    int32_t count = 0;
    float sum_w = 0.0f;
    uint64_t valid_mask = 0;

    const float *w_row = &probmat[curr * k];
    const int32_t *nn_row = &nn_list[curr * k];

    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_row[j];
      if (!visited[v]) {
        candidates[count] = v;
        cand_j[count] = j;
        weights[count] = w_row[j];
        sum_w += weights[count];
        if (j < 64)
          valid_mask |= (1ULL << j);
        count++;
      }
    }

    int32_t next_node = -1;
    int16_t pick_j = -1;

    if (count > 0) {
      float r = rng.next_float() * sum_w;
      float current_sum = 0.0f;
      int32_t selected_idx = -1;

      for (int32_t i = 0; i < count; ++i) {
        current_sum += weights[i];
        if (current_sum >= r) {
          selected_idx = i;
          break;
        }
      }
      if (selected_idx == -1)
        selected_idx = count - 1;

      next_node = candidates[selected_idx];
      pick_j = static_cast<int16_t>(cand_j[selected_idx]);

      if (require_prob) {
        log_p_total += std::log(weights[selected_idx] / sum_w);
      }
    } else {
      // Fallback: nearest unvisited
      float min_d = std::numeric_limits<float>::max();
      int32_t best_fallback = -1;

      for (int32_t v = 0; v < n; ++v) {
        if (!visited[v]) {
          float d = dist(curr, v);
          if (d < min_d) {
            min_d = d;
            best_fallback = v;
          }
        }
      }

      if (best_fallback != -1) {
        next_node = best_fallback;
        pick_j = -1; // Not in nn_list
      } else {
        break;
      }
    }

    // Record trace
    if (trace) {
      trace->curr_nodes.push_back(curr);
      trace->chosen_nodes.push_back(next_node);
      trace->is_stochastic.push_back(count > 1 ? 1 : 0);
      trace->pick_j.push_back(pick_j);
      trace->valid_mask.push_back(valid_mask);
      trace->is_new_edge.push_back(0); // ACO_TSP doesn't track this
    }

    route_out[step] = next_node;
    visited[next_node] = 1;
    curr = next_node;
  }

  logp_out = log_p_total;
  return get_route_cost(route_out);
}

void ACO_TSP::sample(bool require_prob, const float *prior,
                     SampleResult &result, bool parallel_traced) {
  result.clear();
  result.costs.resize(n_ants);
  result.routes.resize(n_ants);
  if (require_prob)
    result.logps.resize(n_ants);

  std::vector<float> probmat;
  compute_probmat(prior, probmat);

  std::vector<int32_t> starts(n_ants);
  for (int i = 0; i < n_ants; ++i)
    starts[i] = rng_.next_uint(n);

  // Collect per-ant traces
  std::vector<MFACOTrace> ant_traces(n_ants);

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    Xoshiro128Plus local_rng = rng_;
    for (int k = 0; k < thread_id * 100; ++k)
      local_rng.next_u32();

#pragma omp for schedule(dynamic)
    for (int i = 0; i < n_ants; ++i) {
      float logp = 0.0f;
      MFACOTrace *trace_ptr = require_prob ? &ant_traces[i] : nullptr;
      result.costs[i] =
          sample_ant_constructive(probmat.data(), starts[i], result.routes[i],
                                  local_rng, require_prob, logp, trace_ptr);
      if (require_prob)
        result.logps[i] = logp;
    }
  }

  // Batch traces if require_prob
  if (require_prob) {
    result.traces.clear();
    result.traces.starts.push_back(0);
    result.traces.start_nodes.reserve(n_ants);

    for (int i = 0; i < n_ants; ++i) {
      const auto &tr = ant_traces[i];
      result.traces.start_nodes.push_back(tr.start_node);

      for (size_t d = 0; d < tr.curr_nodes.size(); ++d) {
        result.traces.curr_nodes.push_back(tr.curr_nodes[d]);
        result.traces.chosen_nodes.push_back(tr.chosen_nodes[d]);
        result.traces.is_stochastic.push_back(tr.is_stochastic[d]);
        result.traces.pick_j.push_back(tr.pick_j[d]);
        result.traces.valid_mask.push_back(tr.valid_mask[d]);
        result.traces.is_new_edge.push_back(tr.is_new_edge[d]);
      }
      result.traces.starts.push_back(
          static_cast<int32_t>(result.traces.curr_nodes.size()));
    }
  }
}

void ACO_TSP::update_pheromone(const int32_t *solution_flat, float cost) {
  if (cost < best_cost) {
    best_cost = cost;
    best_route.assign(solution_flat, solution_flat + n);

    if (min_max) {
      auto limits = calc_trail_limits(best_cost);
      tau_min = limits.first;
      tau_max = limits.second;
    }
  }

  std::vector<int32_t> pos(n);
  for (int i = 0; i < n; ++i)
    pos[static_cast<size_t>(solution_flat[i])] = i;

  auto in_route = [&](int32_t u, int32_t v) {
    int32_t pu = pos[u];
    int32_t pv = pos[v];
    int32_t diff = std::abs(pu - pv);
    return diff == 1 || diff == n - 1;
  };

  float deposit = 1.0f / best_cost;

#pragma omp parallel for schedule(static)
  for (int32_t u = 0; u < n; ++u) {
    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_list[u * k + j];
      float &val = pheromone[u * k + j];

      val *= (1.0f - rho);

      if (in_route(u, v)) {
        val += deposit;
      }

      if (min_max) {
        if (val > tau_max)
          val = tau_max;
        if (val < tau_min)
          val = tau_min;
      }
    }
  }
}

// ============================================================================
// ACO_CVRP Implementation
// ============================================================================

ACO_CVRP::ACO_CVRP(const float *coords_ptr, const float *demand_ptr, int32_t n_,
                   float capacity_, int32_t n_ants_, int32_t cand_list_size,
                   float decay, float alpha_, float beta_, float p_best_,
                   bool min_max_, bool elitist_, bool use_local_search_)
    : n(n_), n_ants(n_ants_), rho(1.0f - decay), alpha(alpha_), beta(beta_),
      p_best(p_best_), min_max(min_max_), elitist(elitist_),
      use_local_search(use_local_search_), capacity(capacity_) {

  // Dense mode by default if k=0 passed
  // Force k=n for MMAS/ACO_CVRP as it relies on dense indexing
  // We disregard cand_list_size if it results in sparse behavior for this class
  k = n;

  // Copy data
  coords.resize(static_cast<size_t>(n) * 2);
  std::memcpy(coords.data(), coords_ptr,
              sizeof(float) * static_cast<size_t>(n) * 2);

  demand.resize(n);
  std::memcpy(demand.data(), demand_ptr,
              sizeof(float) * static_cast<size_t>(n));

  // Build lists
  build_dense_nn_lists();
  build_heuristic();

  // Initialize pheromone to 1.0 (or small value? MMAS usually uses 1/something)
  tau_min = min_max_ ? 0.1f : 0.0001f;
  tau_max = 1000.0f; // Unbounded initially

  pheromone.assign(n * k, min_max_ ? tau_min : 1.0f);
  best_cost = 0.0f;

  // RNG
  rng_.seed(42);
}

void ACO_CVRP::seed_rng(uint64_t seed) { rng_.seed(seed); }

void ACO_CVRP::build_dense_nn_lists() {
  nn_list.resize(n * k);
  for (int32_t i = 0; i < n; ++i) {
    for (int32_t j = 0; j < k; ++j) {
      nn_list[i * k + j] = j;
    }
  }
}

void ACO_CVRP::build_heuristic() {
  heuristic.resize(n * k);
  for (int32_t i = 0; i < n; ++i) {
    for (int32_t j = 0; j < k; ++j) {
      int32_t neighbor = nn_list[i * k + j];
      if (i == neighbor) {
        heuristic[i * k + j] = 0.0f; // Self
      } else {
        float d = dist(i, neighbor);
        heuristic[i * k + j] = (d < EPS) ? 1e9f : (1.0f / d);
      }
    }
  }
}

void ACO_CVRP::compute_probmat(const float *prior,
                               std::vector<float> &probmat) {
  size_t sz = static_cast<size_t>(n) * k;
  probmat.resize(sz);

  for (size_t i = 0; i < sz; ++i) {
    float tau = pheromone[i];
    float eta = heuristic[i];

    // Python: dist = pheromone ** alpha * heuristic ** beta
    float prob = std::pow(tau, alpha) * std::pow(eta, beta);

    // Prior from neural network (if provided) - treated as multiplicative
    if (prior) {
      // prior is log-additive from neural net, convert to multiplicative
      prob *= std::exp(prior[i]);
    }

    probmat[i] = prob;
  }
}

float ACO_CVRP::sample_ant_constructive(const float *probmat,
                                        std::vector<int32_t> &route_out,
                                        Xoshiro128Plus &rng, bool require_prob,
                                        float &logp_out, MFACOTrace *trace) {
  route_out.clear();
  route_out.reserve(n * 2);

  int32_t curr = 0; // Depot
  route_out.push_back(curr);

  float cur_capacity = capacity;
  float total_cost = 0.0f;
  logp_out = 0.0f;

  // Initialize trace if provided
  if (trace) {
    trace->clear();
    trace->start_node = 0; // CVRP always starts from depot
    trace->reserve(n * 2);
  }

  std::vector<uint8_t> visited(n, 0);
  visited[0] = 1;
  int32_t visited_count = 1; // Depot visited

  while (visited_count < n) {
    int32_t next_node = -1;
    int16_t pick_j = -1;
    uint64_t valid_mask = 0;
    bool is_stochastic = false;

    // Optimized scan for dense
    std::vector<int32_t> candidates;
    std::vector<int32_t> cand_j; // j index in nn_list
    std::vector<float> probs;
    candidates.reserve(n);
    cand_j.reserve(n);
    probs.reserve(n);
    double sum_prob = 0.0;

    for (int32_t j = 1; j < n; ++j) {
      if (!visited[j]) {
        if (demand[j] <= cur_capacity) {
          candidates.push_back(j);
          cand_j.push_back(j); // For CVRP dense, j == node index
          float p = probmat[curr * k + j];
          probs.push_back(p);
          sum_prob += p;
          if (j < 64)
            valid_mask |= (1ULL << j);
        }
      }
    }

    if (candidates.empty()) {
      if (curr == 0) {
        // Impossible to satisfy next customer? Or done?
        // If visited_count < n, and no customer fits capacity (even after
        // refill if curr==0), implies infeasible for this capacity. For
        // robustness, we abort or loop? We'll break loop.
        break;
      } else {
        // Split Route
        // Check bounds

        // Insert new depot after curr
        next_node = 0;           // Return to depot
        candidates.push_back(0); // Necessary for selection logic below if we
                                 // wanted to unify, but here we just pick 0
        // Actuallly, simply setting next_node=0 and handling it is cleaner,
        // but the code below expects candidates selection.
        // Let's just set selection directly.

        pick_j = 0;
        is_stochastic = false;
        // Proceed to use next_node directly, skip selection loop?
        // The code below line 3937 uses candidates.back() if logic flow
        // continues. But we are in the 'candidates.empty()' block. We set
        // next_node = 0. We need to skip the selection loop. The original code
        // had `else { next_node = 0; ... }` and fell through? No, looking at
        // lines 3910+: if (candidates.empty()) { ... } else { ... selection ...
        // } So we are fine.
      }
    } else {
      is_stochastic = true;

      // Enforce greedy capacity usage matching Python reference:
      // If we can visit a customer, we MUST visit a customer.
      // Do NOT add depot (0) to candidates if we have valid customer
      // candidates.

      // if (curr != 0) { ... } REMOVED

      // Align with Python: If not at depot, we can technically return to depot
      // if we choose to (probabilistic).
      // Python: visit_mask[0] = 1 always, except if we are at depot and
      // customers exist. So if curr != 0, depot 0 is a valid candidate.
      if (curr != 0) {
        candidates.push_back(0);
        // j for depot 0? In dense mode, j corresponds to node index 0.
        cand_j.push_back(0);
        float p = probmat[curr * k + 0];
        probs.push_back(p);
        sum_prob += p;
        if (0 < 64)
          valid_mask |= (1ULL << 0);
      }

      double r = rng.next_float() * sum_prob;
      double running = 0.0;
      next_node = candidates.back();
      pick_j = static_cast<int16_t>(cand_j.back());
      int32_t selected_idx = static_cast<int32_t>(candidates.size()) - 1;

      for (size_t i = 0; i < candidates.size(); ++i) {
        running += probs[i];
        if (running >= r) {
          next_node = candidates[i];
          pick_j = static_cast<int16_t>(cand_j[i]);
          selected_idx = static_cast<int32_t>(i);
          if (require_prob) {
            // Avoid log(0)
            if (probs[i] > 0 && sum_prob > 0)
              logp_out += std::log(probs[i] / sum_prob);
          }
          break;
        }
      }
    }

    // Record trace for stochastic decisions only
    if (trace && is_stochastic) {
      trace->curr_nodes.push_back(curr);
      trace->chosen_nodes.push_back(next_node);
      trace->is_stochastic.push_back(1);
      trace->pick_j.push_back(pick_j);
      trace->valid_mask.push_back(valid_mask);
      trace->is_new_edge.push_back(0); // ACO_CVRP doesn't track this
    }

    total_cost += dist(curr, next_node);
    route_out.push_back(next_node);

    if (next_node == 0) {
      cur_capacity = capacity;
      curr = 0;
    } else {
      cur_capacity -= demand[next_node];
      if (!visited[next_node]) {
        visited[next_node] = 1;
        visited_count++;
      }
      curr = next_node;
    }
  }

  // Return to depot
  if (curr != 0) {
    total_cost += dist(curr, 0);
    route_out.push_back(0);
  }

  return total_cost;
}

void ACO_CVRP::sample(bool require_prob, const float *prior,
                      SampleResult &result, bool parallel_traced) {
  result.clear();
  result.costs.resize(n_ants);
  result.routes.resize(n_ants);

  if (require_prob) {
    result.logps.resize(n_ants);
  }

  std::vector<float> probmat;
  compute_probmat(prior, probmat);

  // Collect per-ant traces
  std::vector<MFACOTrace> ant_traces(n_ants);

#pragma omp parallel if (parallel_traced || !require_prob)
  {
    int tid = omp_get_thread_num();
    Xoshiro128Plus local_rng = rng_;
    // Mix seed
    local_rng.seed(rng_.next_u32() + tid * 123456789ULL);

    std::vector<int32_t> route;
#pragma omp for
    for (int32_t i = 0; i < n_ants; ++i) {
      float logp = 0.0f;
      MFACOTrace *trace_ptr = require_prob ? &ant_traces[i] : nullptr;
      float c = sample_ant_constructive(probmat.data(), route, local_rng,
                                        require_prob, logp, trace_ptr);

      result.costs[i] = c;
      result.routes[i] = route;
      if (require_prob)
        result.logps[i] = logp;
    }
  }

  // Batch traces if require_prob
  if (require_prob) {
    result.traces.clear();
    result.traces.starts.push_back(0);
    result.traces.start_nodes.reserve(n_ants);

    for (int i = 0; i < n_ants; ++i) {
      const auto &tr = ant_traces[i];
      result.traces.start_nodes.push_back(tr.start_node);

      for (size_t d = 0; d < tr.curr_nodes.size(); ++d) {
        result.traces.curr_nodes.push_back(tr.curr_nodes[d]);
        result.traces.chosen_nodes.push_back(tr.chosen_nodes[d]);
        result.traces.is_stochastic.push_back(tr.is_stochastic[d]);
        result.traces.pick_j.push_back(tr.pick_j[d]);
        result.traces.valid_mask.push_back(tr.valid_mask[d]);
        result.traces.is_new_edge.push_back(tr.is_new_edge[d]);
      }
      result.traces.starts.push_back(
          static_cast<int32_t>(result.traces.curr_nodes.size()));
    }
  }
}

void ACO_CVRP::update_pheromone(const int32_t *solution_flat,
                                int32_t solution_len, float cost) {
  for (size_t i = 0; i < pheromone.size(); ++i) {
    pheromone[i] *= (1.0 - rho);
    if (min_max) {
      if (pheromone[i] < tau_min)
        pheromone[i] = tau_min;
    }
  }

  if (solution_len < 2)
    return;

  float deposit = 1.0f / cost;

  for (int32_t i = 0; i < solution_len - 1; ++i) {
    int32_t u = solution_flat[i];
    int32_t v = solution_flat[i + 1];
    if (u < n && v < n) {
      pheromone[u * k + v] += deposit;
    }
  }

  if (min_max) {
    if (cost < best_cost || best_cost == 0.0f) {
      best_cost = cost;
      best_route.assign(solution_flat, solution_flat + solution_len);

      auto limits = calc_trail_limits(best_cost);
      tau_min = limits.first;
      tau_max = limits.second;

      // Apply explicit limits immediately?
      // User code: "self.max = max", "pheromone[... > max] = max", etc.
      // We do it below.
    }
    // Clamp
    for (auto &val : pheromone) {
      if (val > tau_max)
        val = tau_max;
      if (val < tau_min)
        val = tau_min;
    }
  }
}

std::pair<float, float> ACO_CVRP::calc_trail_limits(float solution_cost) const {
  // Match Python: max = problem_size / lowest_cost
  float max_t = static_cast<float>(n) / solution_cost;
  // min = 0.1 is fixed in Python (self.min = 0.1 if min is None)
  float min_t = 0.1f;
  return {min_t, max_t};
}

float ACO_CVRP::run(int32_t n_iterations) {
  SampleResult result;
  // Pre-allocate result buffers
  result.costs.resize(n_ants);
  result.routes.resize(n_ants);

  // We reuse a single result object to avoid reallocations
  // but sample() calls result.clear()...
  // Actually sample() does: result.clear(); result.costs.resize(...);
  // So it effectively reallocates or at least clears.
  // Standard vector clear/resize keeps capacity usually.

  for (int32_t iter = 0; iter < n_iterations; ++iter) {
    // 1. Generate paths
    // We don't use prior in standard run (unless passed? API doesn't differ)
    // We assume prior=nullptr for basic run.
    sample(false, nullptr, result, false);

    // 2. Local Search (if enabled)
    if (use_local_search) {
#pragma omp parallel for schedule(static)
      for (int32_t i = 0; i < n_ants; ++i) {
        local_search(result.routes[i]);
        // Recompute cost
        float new_cost = 0.0f;
        const auto &r = result.routes[i];
        for (size_t k = 0; k < r.size() - 1; ++k) {
          new_cost += dist(r[k], r[k + 1]);
        }
        result.costs[i] = new_cost;
      }
    }

    // 3. Find iteration best
    int32_t iter_best_idx = -1;
    float iter_best_cost = std::numeric_limits<float>::max();

    for (int32_t i = 0; i < n_ants; ++i) {
      if (result.costs[i] < iter_best_cost) {
        iter_best_cost = result.costs[i];
        iter_best_idx = i;
      }
    }

    bool improved = false;
    if (iter_best_cost < best_cost || best_cost == 0.0f) {
      // update_pheromone will handle best_cost/best_route update internally
      // if we pass this solution to it?
      // Actually update_pheromone logic: "if (cost < best_cost) { update ... }"
      // So we just need to ensure we call it.
      improved = true;
    }

    // 3. Update Pheromone
    // Python logic:
    // if not adaptive or improved: update_pheromone
    // else: diversification (if adaptive)
    // We assume adaptive=False for now essentially, or simplify.
    // ALWAYS update for standard ACO.

    // Elitist vs All?
    // User Python: if elitist (adaptive default=False): best only.
    //              else: all.
    // The "standard" ACO_CVRP usually uses elitist in MMAS.
    // But user passed "elitist=False" in one example (pure AS?).
    // However, the class name implies MMAS features (min_max).
    // Let's implement specific logic matching Python variable `elitist`?
    // Python code *inferred*: `self.elitist = elitist or adaptive`.
    // My C++ `ACO_CVRP` does NOT have an `elitist` member variable exposed in
    // header yet? Checking header... ACO_TSP has `elitist`. ACO_CVRP in
    // header... `class ACO_CVRP { ... public: ... float p_best; bool min_max;
    // ... }` It seems I missed `elitist` in ACO_CVRP definition in header when
    // reviewing. Let's check header content again for ACO_CVRP. Line 693: class
    // ACO_CVRP It has `bool min_max`. No `elitist`. I should infer elitist
    // behavior or defaults. Given the user wants to "NOT use split", and use
    // "ACO for CVRP". I made `update_pheromone` take single solution (Elitist).
    // I added `update_pheromone_batch` (All).
    // I will use `batch` update if not min_max? Or should I add `elitist` flag?
    // Using `min_max` corresponds strongly to MMAS which is Elitist.
    // If `min_max` is true -> Elitist update (best iter or best global).
    // If `min_max` is false -> "AS" style?
    // In Python code: `min_max` and `elitist` are separate.
    // `aco = ACO(..., elitist=False, adaptive=False)`.
    // I will add a `bool elitist = true;` to ACO_CVRP? Or just hardcode
    // behavior based on `min_max`? Safest is to check `min_max`. If min_max,
    // likely MMAS -> Elitist. But wait, Python code allows `elitist=False,
    // min_max=False`. I will assume `min_max` implies elitist behavior in C++
    // for simplicity unless I add a field. But wait, `update_pheromone`
    // (scalar) updates ONE path. `update_pheromone_batch` updates ALL. In
    // `run()`: If `min_max` is true: default to Elitist (update iteration
    // best). Else: update all.

    // If elitist is true -> Elitist update (best iter)
    // Else -> AS update (all ants)
    // Note: min_max affects clamping, elitist affects who deposits.

    if (elitist) {
      // Elitist update using iteration best
      update_pheromone(result.routes[iter_best_idx].data(),
                       (int32_t)result.routes[iter_best_idx].size(),
                       iter_best_cost);
    } else {
      // AS update (all ants)
      update_pheromone_batch(result.routes, result.costs);
    }

    // Note: Python `update_pheromone` also does decay *inside*.
    // And clamping.
    // If I call `update_pheromone_batch`, it should do decay ONCE, then deposit
    // all.
  }
  return best_cost;
}

void ACO_CVRP::update_pheromone_batch(
    const std::vector<std::vector<int32_t>> &routes,
    const std::vector<float> &costs) {

  // 1. Decay
  for (size_t i = 0; i < pheromone.size(); ++i) {
    pheromone[i] *= (1.0f - rho);
    if (min_max) { // Should not happen if we use this for non-min_max, but safe
                   // to keep
      if (pheromone[i] < tau_min)
        pheromone[i] = tau_min;
    }
  }

  // 2. Deposit all
  for (size_t a = 0; a < routes.size(); ++a) {
    const auto &route = routes[a];
    float c = costs[a];
    if (c < 1e-9f)
      continue;

    float deposit = 1.0f / c;
    for (size_t i = 0; i < route.size() - 1; ++i) {
      int32_t u = route[i];
      int32_t v = route[i + 1];
      if (u < n && v < n) {
        pheromone[u * k + v] += deposit;
      }
    }
  }

  // 3. Min-Max Clamping (if enabled)
  // And update best if needed (scan batch)
  // Actually usually caller handles best update.
  // But `update_pheromone` (scalar) handles it.
  // Here we should also check if any route is new best.

  if (min_max) {
    float batch_best = std::numeric_limits<float>::max();
    const std::vector<int32_t> *batch_best_route = nullptr;

    for (size_t a = 0; a < costs.size(); ++a) {
      if (costs[a] < batch_best) {
        batch_best = costs[a];
        batch_best_route = &routes[a];
      }
    }

    if (batch_best < best_cost || best_cost == 0.0f) {
      best_cost = batch_best;
      if (batch_best_route)
        best_route = *batch_best_route;

      auto limits = calc_trail_limits(best_cost);
      tau_min = limits.first;
      tau_max = limits.second;
    }

    // Clamp
    for (auto &val : pheromone) {
      if (val > tau_max)
        val = tau_max;
      if (val < tau_min)
        val = tau_min;
    }
  } else {
    // Even if not min_max, we might want to track best_cost?
    // Yes, for reporting.
    for (size_t a = 0; a < costs.size(); ++a) {
      if (costs[a] < best_cost || best_cost == 0.0f) {
        best_cost = costs[a];
        best_route = routes[a];
      }
    }
  }
}

void ACO_CVRP::local_search(std::vector<int32_t> &route) {
  // Parsing route: 0 -> c... -> 0 -> c... -> 0
  // Identify segments between 0s
  // Optimize each segment using two_opt_sequence
  // Note: route might change size? No, 2-opt on sequence preserves size.

  if (route.empty())
    return;

  // Extract segments
  std::vector<int32_t> new_route;
  new_route.reserve(route.size());
  new_route.push_back(0); // Start with depot

  std::vector<int32_t> segment;
  segment.reserve(n);

  for (size_t i = 1; i < route.size(); ++i) {
    int32_t node = route[i];
    if (node == 0) {
      // End of segment
      if (!segment.empty()) {
        two_opt_sequence(segment);
        // Append optimized segment
        for (int32_t c : segment)
          new_route.push_back(c);
      }
      new_route.push_back(0); // Append delimiter
      segment.clear();
    } else {
      segment.push_back(node);
    }
  }

  // Replace
  route = new_route;
}

void ACO_CVRP::two_opt_sequence(std::vector<int32_t> &seq) {
  // Construct path including endpoints (0)
  std::vector<int32_t> path;
  path.reserve(seq.size() + 2);
  path.push_back(0);
  for (int32_t c : seq)
    path.push_back(c);
  path.push_back(0);

  int32_t size = static_cast<int32_t>(path.size());
  if (size < 4)
    return; // Need at least 0-A-B-0 (4 nodes) for 2-opt?
  // 0-A-0: size 3. No swap possible.
  // 0-A-B-0: size 4. Edges (0,A), (A,B), (B,0).
  // Swap? i=0 (0,A), j=2 (B,0). Swap implies new edges (0,B) (A,0). Reverses
  // A-B to B-A. Yes, possible.

  bool improved = true;
  int iter = 0;
  // Limit iterations for speed
  while (improved && iter < 50) {
    improved = false;
    iter++;

    for (int i = 0; i < size - 2; ++i) {
      for (int j = i + 2; j < size - 1; ++j) {
        // Edge 1: (path[i], path[i+1])
        // Edge 2: (path[j], path[j+1])
        // Candidate: (path[i], path[j]) and (path[i+1], path[j+1])
        // Reverse path[i+1...j]

        int32_t u1 = path[i];
        int32_t v1 = path[i + 1];
        int32_t u2 = path[j];
        int32_t v2 = path[j + 1];

        float d_current = dist(u1, v1) + dist(u2, v2);
        float d_new = dist(u1, u2) + dist(v1, v2);

        if (d_new < d_current - 1e-6f) { // Epsilon improvement
          // Apply swap: reverse [i+1, j]
          std::reverse(path.begin() + i + 1, path.begin() + j + 1);
          improved = true;
        }
      }
    }
  }

  // Copy back
  for (size_t k = 0; k < seq.size(); ++k) {
    seq[k] = path[k + 1];
  }
}

} // namespace mfaco

// ============================================================================
// MFACO_CVRP::sample_ant_direct / traced (Relocation-Based Sampling)
// ============================================================================

namespace mfaco {

float MFACO_CVRP::sample_ant_direct(const float *probmat, int32_t start_node,
                                    std::vector<int32_t> &route_out,
                                    int32_t &new_edges_out,
                                    std::vector<int32_t> &checklist,
                                    Xoshiro128Plus &rng, const float *prior) {
  // Relocation-Based Sampling (Phase 1) with Linked List & Split Logic

  // 1. Build Adjacency of Source (for new edge detection)
  std::vector<int32_t> src_prev(n, -1);
  std::vector<int32_t> src_next(n, -1);
  std::vector<uint8_t> src_adj_to_depot(n, 0);

  // Original route IDs for cross-route checks
  std::vector<int32_t> source_node_route_id(n, -1);
  int32_t current_route_id = 0;
  for (size_t i = 0; i < source_route.size(); ++i) {
    int32_t u = source_route[i];
    if (u == 0) {
      if (i > 0 && source_route[i - 1] != 0) {
        current_route_id++;
      }
    } else {
      source_node_route_id[u] = current_route_id;
    }
  }

  for (size_t i = 0; i + 1 < source_route.size(); ++i) {
    int32_t u = source_route[i];
    int32_t v = source_route[i + 1];
    if (u > 0 && u < n && v > 0 && v < n) {
      src_next[u] = v;
      src_prev[v] = u;
    } else {
      if (u == 0 && v > 0 && v < n)
        src_adj_to_depot[v] = 1;
      if (v == 0 && u > 0 && u < n)
        src_adj_to_depot[u] = 1;
    }
  }

  auto is_source_edge = [&](int32_t u, int32_t v) -> bool {
    if (u >= n || v >= n)
      return false;
    if (u == 0)
      return src_adj_to_depot[v];
    if (v == 0)
      return src_adj_to_depot[u];
    return (src_next[u] == v || src_prev[u] == v);
  };

  // 2. Initialize Linked List from Source Solution
  // Routes: [0, c1..ck, 0]. In LL: depot_node -> c1 -> ... -> ck -> depot_node
  std::vector<std::vector<int32_t>> init_routes =
      initial_routes_from_perm(source_route);
  int32_t num_routes = (int32_t)init_routes.size();
  int32_t max_routes = n; // Allow growth

  // Structures
  std::vector<int32_t> next_node(n + max_routes);
  std::vector<int32_t> prev_node(n + max_routes);
  std::vector<int32_t> node_route(n, -1); // Only for customers 1..n-1
  std::vector<int64_t> route_loads(max_routes, 0);

  for (int r = 0; r < num_routes; ++r) {
    int32_t depot = n + r;
    int32_t prev = depot;
    int64_t load = 0;

    const auto &rt = init_routes[r];
    // rt is [0, c1...ck, 0]
    for (size_t i = 1; i < rt.size() - 1; ++i) {
      int32_t u = rt[i];
      next_node[prev] = u;
      prev_node[u] = prev;
      node_route[u] = r;
      load += demand_int[u];
      prev = u;
    }
    next_node[prev] = depot;
    prev_node[depot] = prev;
    route_loads[r] = load;
  }

  // 3. Setup Sampling
  std::vector<uint8_t> visited(n, 0);
  int32_t visited_count = 0;

  int32_t curr = start_node;
  if (curr <= 0 || curr >= n) {
    // Robust start node selection
    int attempts = 0;
    while (attempts < 10) {
      curr = 1 + (int32_t)rng.next_uint((uint32_t)m);
      if (curr > 0 && curr < n)
        break;
      attempts++;
    }
    if (curr <= 0 || curr >= n)
      curr = 1;
  }

  visited[curr] = 1;
  visited_count++;

  checklist.clear();
  checklist.push_back(curr);
  std::vector<uint8_t> in_checklist(n, 0);
  in_checklist[curr] = 1;

  int32_t new_edges_all = 0;
  int32_t new_edges_cross = 0;
  int32_t steps = 0;
  int32_t max_steps = m * 4;
  if (fixed_steps > 0)
    max_steps = fixed_steps;

  // 4. Main Relocation Loop
  while (true) {
    // Termination Check
    if (fixed_steps > 0) {
      if (steps >= fixed_steps)
        break;
    } else {
      if (new_edges_cross >= min_new_edges || visited_count >= m)
        break;
    }
    if (steps > max_steps)
      break;

    // Determine Current Route info
    int32_t r_curr;
    if (curr >= n) {
      r_curr = curr - n;
    } else {
      r_curr = node_route[curr];
    }

    // Select Next Node 'v'
    int32_t chosen = -1;

    // A. Candidate List
    {
      int32_t candidates[MAX_CAND_LIST_SIZE + 1];
      float probs[MAX_CAND_LIST_SIZE + 1];
      int32_t c_size = 0;
      float sum_prob = 0.0f;

      // Use `curr` for NN lookup. If curr is depot (>=n), use 0.
      int32_t lookup_node = (curr >= n) ? 0 : curr;
      const float *row_prob = probmat + (size_t)lookup_node * k;

      for (int32_t j = 0; j < k; ++j) {
        int32_t v = nn_list[lookup_node * k + j];
        // Filter
        if (v == 0) {
          // Depot is valid target? Only if curr is NOT depot
          if (curr >= n)
            continue;
        } else {
          if (v < 0 || v >= n)
            continue;
          if (visited[v])
            continue;

          // Capacity Check: can we put v after curr?
          // If v is in same route already: Moving it doesn't increase total
          // load unless we duplicate (we relocate). So only check if differernt
          // route.
          int32_t r_v = node_route[v];
          if (r_v != r_curr) {
            if (route_loads[r_curr] + demand_int[v] > capacity_int)
              continue;
          }
        }

        candidates[c_size] = v;
        probs[c_size] = row_prob[j];
        sum_prob += row_prob[j];
        c_size++;
      }

      if (c_size > 0) {
        float r = rng.next_float() * sum_prob;
        float run = 0.0f;
        chosen = candidates[c_size - 1];
        for (int32_t i = 0; i < c_size; ++i) {
          run += probs[i];
          if (r <= run) {
            chosen = candidates[i];
            break;
          }
        }
      }
    }

    // B. Backup List
    if (chosen == -1) {
      int32_t lookup_node = (curr >= n) ? 0 : curr;
      for (int32_t j = 0; j < bl; ++j) {
        int32_t v = backup_list[lookup_node * bl + j];
        if (v == 0) {
          if (curr >= n)
            continue;
          chosen = 0;
          break;
        }
        if (v > 0 && v < n && !visited[v]) {
          int32_t r_v = node_route[v];
          if (r_curr == r_v ||
              route_loads[r_curr] + demand_int[v] <= capacity_int) {
            chosen = v;
            break;
          }
        }
      }
    }

    // C. Global Fallback (Closest Valid Unvisited)
    if (chosen == -1) {
      float min_d = std::numeric_limits<float>::max();
      int32_t best_global = -1;
      int32_t lookup_node = (curr >= n) ? 0 : curr;

      // Optimization: Skip if we simply can't find anything?
      // We must scan.
      for (int32_t v = 1; v < n; ++v) {
        if (!visited[v]) {
          int32_t r_v = node_route[v];
          if (r_curr == r_v ||
              route_loads[r_curr] + demand_int[v] <= capacity_int) {
            float d = dist(lookup_node, v);
            if (d < min_d) {
              min_d = d;
              best_global = v;
            }
          }
        }
      }
      // Also consider Depot (0) as fallback to close route?
      // If we can't find any customer, we MUST close route.
      if (best_global == -1) {
        if (curr < n)
          chosen = 0; // Force close/split if customer
      } else {
        // Check if depot is closer than best_global?
        float d_depot = dist(lookup_node, 0);
        if (curr < n && d_depot < min_d) {
          chosen = 0;
        } else {
          chosen = best_global;
        }
      }
    }

    if (chosen == -1)
      break;

    // Execute Transition
    // 1. If v == 0: Split / End Route
    if (chosen == 0) {
      int32_t next_c = next_node[curr];

      // If curr is already at end of route (next is depot), just step.
      if (next_c >= n) {
        // Simply traverse to depot
        // Stats
        bool is_cross = true; // depot edge is cross/boundary
        if (is_cross)
          new_edges_cross++; // Count moving to depot as perturbation?
        if (!is_source_edge(curr >= n ? 0 : curr, 0)) {
          new_edges_all++;
        }

        curr = next_c;
        steps++;
        continue;
      }

      // Else: Split.
      if (num_routes >= max_routes) {
        break;
      }

      // Insert new depot after curr
      int32_t r_new = num_routes++;
      int32_t new_depot = n + r_new;
      int32_t old_depot = n + r_curr;

      // Find end of old route (it connects to old_depot)
      int32_t route_end = prev_node[old_depot];

      // Relinking
      next_node[curr] = old_depot;
      prev_node[old_depot] = curr;

      next_node[new_depot] = next_c;
      prev_node[next_c] = new_depot;

      next_node[route_end] = new_depot;
      prev_node[new_depot] = route_end;

      // Update Loads & Route IDs for the new segment
      int64_t load_shift = 0;
      int32_t w = next_c;
      while (w != new_depot && w < n) {
        node_route[w] = r_new;
        load_shift += demand_int[w];
        w = next_node[w];
      }
      route_loads[r_curr] -= load_shift;
      route_loads[r_new] = load_shift;

      // Stats for split edge (curr, 0)
      if (!is_source_edge(curr >= n ? 0 : curr, 0)) {
        new_edges_all++;
        if (!in_checklist[curr >= n ? 0 : curr]) {
          checklist.push_back(curr >= n ? 0 : curr);
          in_checklist[curr >= n ? 0 : curr] = 1;
        }
        new_edges_cross++;
      }

      curr = new_depot; // We are now at the start of new route
      steps++;
      continue;
    }

    // 2. If v is Customer: Relocate v after curr
    int32_t v = chosen;
    int32_t r_v = node_route[v];

    // Unlink v
    int32_t prev_v = prev_node[v];
    int32_t next_v = next_node[v];

    if (prev_v == curr) {
      // Already after curr? Just Traverse
      visited[v] = 1;
      visited_count++;
      curr = v;
      steps++;
      continue;
    }

    next_node[prev_v] = next_v;
    prev_node[next_v] = prev_v;

    // Insert v after curr
    int32_t next_c = next_node[curr];
    next_node[curr] = v;
    prev_node[v] = curr;
    next_node[v] = next_c;
    prev_node[next_c] = v;

    // Updates
    if (r_curr != r_v) {
      if (r_v != -1) {
        route_loads[r_v] -= demand_int[v];
      }
      route_loads[r_curr] += demand_int[v];
      node_route[v] = r_curr;
    }

    // Stats
    visited[v] = 1;
    visited_count++;

    int32_t u_idx = (curr >= n) ? 0 : curr;
    if (!is_source_edge(u_idx, v)) {
      new_edges_all++;
      if (!in_checklist[u_idx]) {
        checklist.push_back(u_idx);
        in_checklist[u_idx] = 1;
      }
      if (!in_checklist[v]) {
        checklist.push_back(v);
        in_checklist[v] = 1;
      }

      // Cross check
      bool is_cross = false;
      if (u_idx == 0)
        is_cross = true; // Depot->Node is boundary edge
      else if (source_node_route_id[u_idx] != source_node_route_id[v])
        is_cross = true;

      if (is_cross)
        new_edges_cross++;
    }

    curr = v;
    steps++;
  }

  new_edges_out = new_edges_cross;

  // 5. Flatten Routes
  route_out.clear();
  route_out.reserve(m + num_routes + 2);

  for (int r = 0; r < num_routes; ++r) {
    int32_t d = n + r;
    int32_t w = next_node[d];
    if (w >= n)
      continue; // Empty

    route_out.push_back(0);
    while (w < n) {
      route_out.push_back(w);
      w = next_node[w];
    }
  }
  if (!route_out.empty())
    route_out.push_back(0);

  // 6. Apply Local Search
  if (use_local_search && !checklist.empty()) {
    // Intra-Route LS (2-opt)
    intra_route_ls(route_out, checklist);
    // Inter-Route LS
    std::vector<int32_t> pos_ls(n, -1);
    inter_route_ls_optimized(route_out, pos_ls, checklist, in_checklist);
    // Intra-Route LS (2-opt)
    intra_route_ls(route_out, checklist);
  }

  // Calculate final cost
  float final_cost = 0.0f;
  for (size_t i = 0; i + 1 < route_out.size(); ++i) {
    final_cost += dist(route_out[i], route_out[i + 1]);
  }

  return final_cost;
}

std::tuple<int32_t, bool, float> MFACO_CVRP::select_next_node(
    int32_t curr, int32_t curr_route, const float *probmat_row,
    const std::vector<uint8_t> &visited, const std::vector<int32_t> &node_route,
    const std::vector<int64_t> &route_loads, Xoshiro128Plus &rng,
    int16_t &out_pick_j, uint64_t &out_valid_mask) {
  int32_t c_size = 0;
  float sum_prob = 0.0f;
  out_valid_mask = 0;
  out_pick_j = -1;

  // Arrays on stack
  int32_t candidates[MAX_CAND_LIST_SIZE + 1];
  float probs[MAX_CAND_LIST_SIZE + 1];
  int16_t j_indices[MAX_CAND_LIST_SIZE + 1];

  int32_t lookup_node = (curr >= n) ? 0 : curr;

  // A. NN List
  for (int32_t j = 0; j < k; ++j) {
    int32_t v = nn_list[lookup_node * k + j];
    if (v < 0 || v >= n)
      continue;
    if (v == curr)
      continue;
    if (visited[v])
      continue;

    int32_t v_route = node_route[v];
    bool can_relocate = true;
    if (curr_route >= 0 && v_route >= 0 && curr_route != v_route) {
      int64_t new_load = route_loads[curr_route] + demand_int[v];
      if (new_load > capacity_int)
        can_relocate = false;
    }

    if (can_relocate) {
      float p = probmat_row[j];
      candidates[c_size] = v;
      probs[c_size] = p;
      j_indices[c_size] = (int16_t)j;
      if (j < 64)
        out_valid_mask |= (1ULL << j);
      sum_prob += p;
      c_size++;
    }
  }

  // B. Backup List
  if (c_size == 0) {
    for (int32_t j = 0; j < bl; ++j) {
      int32_t v = backup_list[lookup_node * bl + j];
      if (v <= 0 || v >= n)
        continue;
      if (visited[v])
        continue;

      int32_t v_route = node_route[v];
      bool can_relocate = true;
      if (curr_route >= 0 && v_route >= 0 && curr_route != v_route) {
        int64_t new_load = route_loads[curr_route] + demand_int[v];
        if (new_load > capacity_int)
          can_relocate = false;
      }

      if (can_relocate) {
        candidates[c_size] = v;
        probs[c_size] = 1.0f;
        j_indices[c_size] = -1;
        sum_prob += 1.0f;
        c_size++;
        break;
      }
    }
  }

  // C. Global Fallback
  if (c_size == 0) {
    float min_d = std::numeric_limits<float>::max();
    int32_t best_global = -1;

    for (int32_t v = 1; v < n; ++v) {
      if (!visited[v]) {
        int32_t r_v = node_route[v];
        if (curr_route == r_v ||
            (curr_route >= 0 &&
             route_loads[curr_route] + demand_int[v] <= capacity_int)) {
          float d = dist(lookup_node, v);
          if (d < min_d) {
            min_d = d;
            best_global = v;
          }
        }
      }
    }

    if (best_global == -1) {
      if (curr < n)
        best_global = 0;
    } else {
      float d_depot = dist(lookup_node, 0);
      if (curr < n && d_depot < min_d) {
        best_global = 0;
      }
    }

    if (best_global != -1) {
      candidates[c_size] = best_global;
      probs[c_size] = 1.0f;
      j_indices[c_size] = -1;
      sum_prob += 1.0f;
      c_size++;
    }
  }

  if (c_size == 0) {
    return {-1, false, 0.0f};
  }

  // Selection
  bool is_stoch = (c_size > 1);
  int32_t chosen = candidates[c_size - 1];
  out_pick_j = j_indices[c_size - 1];
  float picked_prob = probs[c_size - 1];

  if (is_stoch) {
    float r = rng.next_float() * sum_prob;
    float running = 0.0f;
    for (int32_t i = 0; i < c_size; ++i) {
      running += probs[i];
      if (r <= running) {
        chosen = candidates[i];
        out_pick_j = j_indices[i];
        picked_prob = probs[i];
        break;
      }
    }
  }

  float log_prob = 0.0f;
  if (is_stoch && sum_prob > EPS) {
    log_prob = std::log(picked_prob / sum_prob);
  }

  return {chosen, is_stoch, log_prob};
}

float MFACO_CVRP::sample_ant_direct_traced(
    const float *probmat, int32_t start_node, std::vector<int32_t> &route_out,
    std::vector<int32_t> &route_raw_out, float &cost_raw_out,
    int32_t &new_edges_out, std::vector<int32_t> &checklist, MFACOTrace &trace,
    Xoshiro128Plus &rng, float &logp_sum, float &survival_out,
    const float *prior) {
  // Copy-paste from sample_ant_direct, with tracing added
  trace.clear();
  trace.start_node = start_node;
  trace.reserve(min_new_edges * 2);

  // 1. Build Adjacency of Source (for new edge detection)
  std::vector<int32_t> src_prev(n, -1);
  std::vector<int32_t> src_next(n, -1);
  std::vector<uint8_t> src_adj_to_depot(n, 0);

  // Original route IDs for cross-route checks
  std::vector<int32_t> source_node_route_id(n, -1);
  int32_t current_route_id = 0;
  for (size_t i = 0; i < source_route.size(); ++i) {
    int32_t u = source_route[i];
    if (u == 0) {
      if (i > 0 && source_route[i - 1] != 0) {
        current_route_id++;
      }
    } else {
      source_node_route_id[u] = current_route_id;
    }
  }

  for (size_t i = 0; i + 1 < source_route.size(); ++i) {
    int32_t u = source_route[i];
    int32_t v = source_route[i + 1];
    if (u > 0 && u < n && v > 0 && v < n) {
      src_next[u] = v;
      src_prev[v] = u;
    } else {
      if (u == 0 && v > 0 && v < n)
        src_adj_to_depot[v] = 1;
      if (v == 0 && u > 0 && u < n)
        src_adj_to_depot[u] = 1;
    }
  }

  auto is_source_edge = [&](int32_t u, int32_t v) -> bool {
    if (u >= n || v >= n)
      return false;
    if (u == 0)
      return src_adj_to_depot[v];
    if (v == 0)
      return src_adj_to_depot[u];
    return (src_next[u] == v || src_prev[u] == v);
  };

  // 2. Initialize Linked List from Source Solution
  // Routes: [0, c1..ck, 0]. In LL: depot_node -> c1 -> ... -> ck -> depot_node
  std::vector<std::vector<int32_t>> init_routes =
      initial_routes_from_perm(source_route);
  int32_t num_routes = (int32_t)init_routes.size();
  int32_t max_routes = n; // Allow growth

  // Structures
  std::vector<int32_t> next_node(n + max_routes);
  std::vector<int32_t> prev_node(n + max_routes);
  std::vector<int32_t> node_route(n, -1); // Only for customers 1..n-1
  std::vector<int64_t> route_loads(max_routes, 0);

  for (int r = 0; r < num_routes; ++r) {
    int32_t depot = n + r;
    int32_t prev = depot;
    int64_t load = 0;

    const auto &rt = init_routes[r];
    // rt is [0, c1...ck, 0]
    for (size_t i = 1; i < rt.size() - 1; ++i) {
      int32_t u = rt[i];
      next_node[prev] = u;
      prev_node[u] = prev;
      node_route[u] = r;
      load += demand_int[u];
      prev = u;
    }
    next_node[prev] = depot;
    prev_node[depot] = prev;
    route_loads[r] = load;
  }

  // 3. Setup Sampling
  std::vector<uint8_t> visited(n, 0);
  int32_t visited_count = 0;

  int32_t curr = start_node;
  if (curr <= 0 || curr >= n) {
    // Robust start node selection
    int attempts = 0;
    while (attempts < 10) {
      curr = 1 + (int32_t)rng.next_uint((uint32_t)m);
      if (curr > 0 && curr < n)
        break;
      attempts++;
    }
    if (curr <= 0 || curr >= n)
      curr = 1;
  }

  visited[curr] = 1;
  visited_count++;

  checklist.clear();
  checklist.push_back(curr);
  std::vector<uint8_t> in_checklist(n, 0);
  in_checklist[curr] = 1;

  int32_t new_edges_all = 0;
  int32_t new_edges_cross = 0;
  int32_t steps = 0;
  int32_t max_steps = m * 4;
  if (fixed_steps > 0)
    max_steps = fixed_steps;

  logp_sum = 0.0f;

  // 4. Main Relocation Loop
  while (true) {
    // Termination Check
    if (fixed_steps > 0) {
      if (steps >= fixed_steps)
        break;
    } else {
      if (new_edges_cross >= min_new_edges || visited_count >= m)
        break;
    }
    if (steps > max_steps)
      break;

    // Determine Current Route info
    int32_t r_curr;
    if (curr >= n) {
      r_curr = curr - n;
    } else {
      r_curr = node_route[curr];
    }

    // Select Next Node 'v'
    int16_t pick_j = -1;
    uint64_t valid_mask = 0;

    // Map depot nodes to 0 for NN/probmat lookups
    int32_t lookup_node = (curr >= n) ? 0 : curr;
    const float *row_prob = probmat + (size_t)lookup_node * k;

    auto [chosen, is_stoch, log_prob] =
        select_next_node(curr, r_curr, row_prob, visited, node_route,
                         route_loads, rng, pick_j, valid_mask);

    if (chosen == -1)
      break;

    if (is_stoch)
      logp_sum += log_prob;

    // --- Trace Logic ---
    int32_t trace_curr = (curr >= n) ? 0 : curr;
    int32_t trace_chosen = (chosen >= n) ? 0 : chosen;
    bool is_new = !is_source_edge(trace_curr, trace_chosen);

    trace.curr_nodes.push_back(trace_curr);
    trace.chosen_nodes.push_back(trace_chosen);
    trace.is_stochastic.push_back(is_stoch ? 1 : 0);
    trace.pick_j.push_back(pick_j);
    trace.valid_mask.push_back(valid_mask);
    trace.is_new_edge.push_back(is_new ? 1 : 0);
    // -------------------

    // Execute Transition
    // 1. If v == 0: Split / End Route
    if (chosen == 0) {
      int32_t next_c = next_node[curr];

      // If curr is already at end of route (next is depot), just step.
      if (next_c >= n) {
        // Simply traverse to depot
        // Stats
        bool is_cross = true; // depot edge is cross/boundary
        if (is_cross)
          new_edges_cross++; // Count moving to depot as perturbation?
        if (!is_source_edge(curr >= n ? 0 : curr, 0)) {
          new_edges_all++;
        }

        curr = next_c;
        steps++;
        continue;
      }

      // Else: Split.
      if (num_routes >= max_routes) {
        break;
      }

      // Insert new depot after curr
      int32_t r_new = num_routes++;
      int32_t new_depot = n + r_new;
      int32_t old_depot = n + r_curr;

      // Find end of old route (it connects to old_depot)
      int32_t route_end = prev_node[old_depot];

      // Relinking
      next_node[curr] = old_depot;
      prev_node[old_depot] = curr;

      next_node[new_depot] = next_c;
      prev_node[next_c] = new_depot;

      next_node[route_end] = new_depot;
      prev_node[new_depot] = route_end;

      // Update Loads & Route IDs for the new segment
      int64_t load_shift = 0;
      int32_t w = next_c;
      int32_t walk_steps = 0;
      while (w != new_depot && w < n) {
        walk_steps++;
        if (walk_steps > 2 * n)
          break; // safety
        node_route[w] = r_new;
        load_shift += demand_int[w];
        w = next_node[w];
      }
      route_loads[r_curr] -= load_shift;
      route_loads[r_new] = load_shift;

      // Stats for split edge (curr, 0)
      if (!is_source_edge(curr >= n ? 0 : curr, 0)) {
        new_edges_all++;
        if (!in_checklist[curr >= n ? 0 : curr]) {
          checklist.push_back(curr >= n ? 0 : curr);
          in_checklist[curr >= n ? 0 : curr] = 1;
        }
        new_edges_cross++;
      }

      curr = new_depot; // We are now at the start of new route
      steps++;
      continue;
    }

    // 2. If v is Customer: Relocate v after curr
    int32_t v = chosen;
    int32_t r_v = node_route[v];

    // Unlink v
    int32_t prev_v = prev_node[v];
    int32_t next_v = next_node[v];

    if (prev_v == curr) {
      // Already after curr? Just Traverse
      visited[v] = 1;
      visited_count++;
      curr = v;
      steps++;
      continue;
    }

    next_node[prev_v] = next_v;
    prev_node[next_v] = prev_v;

    // Insert v after curr
    int32_t next_c = next_node[curr];
    next_node[curr] = v;
    prev_node[v] = curr;
    next_node[v] = next_c;
    prev_node[next_c] = v;

    // Updates
    if (r_curr != r_v) {
      if (r_v != -1) {
        route_loads[r_v] -= demand_int[v];
      }
      route_loads[r_curr] += demand_int[v];
      node_route[v] = r_curr;
    }

    // Stats
    visited[v] = 1;
    visited_count++;

    int32_t u_idx = (curr >= n) ? 0 : curr;
    if (!is_source_edge(u_idx, v)) {
      new_edges_all++;
      if (!in_checklist[u_idx]) {
        checklist.push_back(u_idx);
        in_checklist[u_idx] = 1;
      }
      if (!in_checklist[v]) {
        checklist.push_back(v);
        in_checklist[v] = 1;
      }

      // Cross check
      bool is_cross = false;
      if (u_idx == 0)
        is_cross = true; // Depot->Node is boundary edge
      else if (source_node_route_id[u_idx] != source_node_route_id[v])
        is_cross = true;

      if (is_cross)
        new_edges_cross++;
    }

    curr = v;
    steps++;
  }

  new_edges_out = new_edges_cross;

  // 5. Flatten Routes
  route_out.clear();
  route_out.reserve(m + num_routes + 1);

  for (int r = 0; r < num_routes; ++r) {
    int32_t d = n + r;
    int32_t w = next_node[d];
    if (w >= n)
      continue; // Empty

    route_out.push_back(0);
    while (w < n) {
      route_out.push_back(w);
      w = next_node[w];
    }
  }
  if (!route_out.empty() && route_out.back() != 0) {
    route_out.push_back(0);
  }

  // Capture raw route before LS
  route_raw_out = route_out;
  cost_raw_out = 0.0f;
  for (size_t i = 0; i + 1 < route_out.size(); ++i) {
    cost_raw_out += dist(route_out[i], route_out[i + 1]);
  }

  // 6. Apply Local Search
  if (use_local_search && !checklist.empty()) {
    // Intra-Route LS (2-opt)
    intra_route_ls(route_out, checklist);
    // Inter-Route LS
    std::vector<int32_t> pos_ls(n, -1);
    inter_route_ls_optimized(route_out, pos_ls, checklist, in_checklist);
    // Intra-Route LS (2-opt)
    intra_route_ls(route_out, checklist);
  }

  // Calculate final cost
  float final_cost = 0.0f;
  for (size_t i = 0; i + 1 < route_out.size(); ++i) {
    final_cost += dist(route_out[i], route_out[i + 1]);
  }

  return final_cost;
}

} // namespace mfaco
