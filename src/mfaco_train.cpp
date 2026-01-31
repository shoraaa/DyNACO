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
  // if (!smooth_mmas || nls)
  //   build_nn_pos();
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

  if (!smooth_mmas) {
    // Classic MMAS-style: evaporate and additively deposit on best route
    float decay_factor = 1.0f - rho;
    for (int32_t i = 0; i < n * k; ++i) {
      pheromone_sparse[i] *= decay_factor;
      pheromone_sparse[i] =
          std::max(tau_min, std::min(tau_max, pheromone_sparse[i]));
    }

    // Deposit on best route
    float deposit = 1.0f / (new_best_cost + EPS);
    int32_t prev = best_flat[n - 1]; // last node before wrap
    for (int32_t i = 0; i < n; ++i) {
      int32_t cur = best_flat[i];

      // Forward edge (prev -> cur)
      int32_t j = find_neighbor_index(prev, cur);
      if (j >= 0) {
        pheromone_sparse[prev * k + j] =
            std::min(pheromone_sparse[prev * k + j] + deposit, tau_max);
      }

      // Symmetric: reverse edge (cur -> prev)
      int32_t jr = find_neighbor_index(cur, prev);
      if (jr >= 0) {
        pheromone_sparse[cur * k + jr] =
            std::min(pheromone_sparse[cur * k + jr] + deposit, tau_max);
      }

      prev = cur;
    }
  } else {
    // Smooth-MMAS: linear interpolation toward tau_min/tau_max targets
    // tau = (1-rho)*tau + rho*tau_target
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

    float keep = 1.0f - rho;
    for (int32_t u = 0; u < n; ++u) {
      for (int32_t j = 0; j < k; ++j) {
        int32_t v = nn_list[u * k + j];
        float target = in_route_edge(u, v) ? tau_max : tau_min;
        float &tau = pheromone_sparse[u * k + j];
        tau = keep * tau + rho * target;
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

// void MFACO_TSP::build_nn_pos() { ... } REMOVED

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

  source_perm.resize(m);
  best_perm.resize(m);
  source_positions.assign(n, -1);

  build_initial_perm();

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

#pragma omp for schedule(static)
    for (int32_t u = 0; u < n; ++u) {
      for (int32_t j = 0; j < total; ++j) {
        uint32_t pt_idx = kdtree.nn_bottom_up(static_cast<uint32_t>(u));
        if (j < k)
          nn_list[u * k + j] = (int32_t)pt_idx;
        else
          backup_list[u * bl + (j - k)] = (int32_t)pt_idx;
        kdtree.delete_point(pt_idx);
      }
      for (int32_t j = 0; j < k; ++j)
        kdtree.undelete_point((uint32_t)nn_list[u * k + j]);
      for (int32_t j = 0; j < bl; ++j)
        kdtree.undelete_point((uint32_t)backup_list[u * bl + j]);
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
      heuristic_sparse[u * k + j] = (d > 0) ? (1.0f / d) : 1.0f;
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
void MFACO_CVRP::build_initial_perm() {
  float best = std::numeric_limits<float>::max();
  std::vector<int32_t> best_p(m);

  int32_t num_starts = std::min(8, m);
  std::vector<int32_t> perm(m);
  std::vector<uint8_t> visited(n, 0);

  for (int32_t s = 0; s < num_starts; ++s) {
    std::fill(visited.begin(), visited.end(), 0);
    visited[0] = 1;

    int32_t start_customer = 1 + (s % m);
    perm[0] = start_customer;
    visited[start_customer] = 1;

    for (int32_t i = 1; i < m; ++i) {
      int32_t curr = perm[i - 1];
      int32_t nxt = -1;

      for (int32_t j = 0; j < k; ++j) {
        int32_t v = nn_list[curr * k + j];
        if (v > 0 && !visited[v]) {
          nxt = v;
          break;
        }
      }
      if (nxt < 0) {
        float md = std::numeric_limits<float>::max();
        for (int32_t v = 1; v < n; ++v) {
          if (visited[v])
            continue;
          float d = dist(curr, v);
          if (d < md) {
            md = d;
            nxt = v;
          }
        }
      }
      perm[i] = nxt;
      visited[nxt] = 1;
    }

    float cost = split_cost_fast(perm);
    if (cost < best) {
      best = cost;
      best_p = perm;
    }
  }

  source_perm = best_p;
  best_perm = best_p;
  source_cost = best;
  best_cost = best;

  std::fill(source_positions.begin(), source_positions.end(), -1);
  for (int32_t i = 0; i < m; ++i)
    source_positions[source_perm[i]] = i;
}

MFACO_CVRP::SplitResult
MFACO_CVRP::split_dp(const std::vector<int32_t> &perm) const {
  const int32_t M = (int32_t)perm.size();
  SplitResult r;
  r.cost = std::numeric_limits<float>::max();
  r.segs.clear();
  if (M == 0) {
    r.cost = 0.0f;
    return r;
  }

  // Need prev pointers for backtracking
  std::vector<int32_t> prev(M + 1, -1);
  std::vector<float> dp(M + 1, 1e30f);

  std::vector<int64_t> P_load(M + 1, 0);
  std::vector<float> P_dist(M + 1, 0.0f);
  std::vector<float> D0(M, 0.0f);

  // 1. Precompute
  P_load[0] = 0;
  P_dist[0] = 0;
  D0[0] = d0[perm[0]];
  for (int i = 0; i < M - 1; ++i) {
    P_load[i + 1] = P_load[i] + demand_int[perm[i]];
    P_dist[i + 1] = P_dist[i] + dist(perm[i], perm[i + 1]);
    D0[i + 1] = d0[perm[i + 1]];
  }
  P_load[M] = P_load[M - 1] + demand_int[perm[M - 1]];

  // 2. Monotonic Deque DP
  std::vector<int32_t> dq(M + 1);
  int head = 0, tail = 0;

  dp[0] = 0.0f;

  // Push i=0
  dq[tail++] = 0;

  for (int j = 1; j <= M; ++j) {
    int64_t limit = P_load[j] - capacity_int;

    // Pop front (capacity)
    while (head < tail && P_load[dq[head]] < limit) {
      head++;
    }

    if (head < tail) {
      int best_i = dq[head];
      float val_i = dp[best_i] + D0[best_i] - P_dist[best_i];
      float const_j =
          P_dist[j - 1] + D0[j - 1]; // D0[j-1] is dist(perm[j-1], 0)

      dp[j] = val_i + const_j;
      prev[j] = best_i;
    }

    // Push j as candidate i for next steps
    if (j < M) {
      float val_j = dp[j] + D0[j] - P_dist[j];
      while (head < tail) {
        int back_i = dq[tail - 1];
        float val_back = dp[back_i] + D0[back_i] - P_dist[back_i];
        if (val_back >= val_j)
          tail--;
        else
          break;
      }
      dq[tail++] = j;
    }
  }

  // Backtrack (same as before)
  r.cost = dp[M];
  int32_t curr = M;
  while (curr > 0) {
    int32_t p = prev[curr];
    if (p < 0)
      break; // Should not happen
    r.segs.push_back({p, curr - 1});
    curr = p;
  }
  std::reverse(r.segs.begin(), r.segs.end());
  return r;
}

// -------------------- split_cost_fast Optimized (O(N)) --------------------
float MFACO_CVRP::split_cost_fast(const std::vector<int32_t> &perm) const {
  const int32_t M = (int32_t)perm.size();
  if (M == 0)
    return 0.0f;

  // Use thread_local buffers to avoid allocations in the hot path
  // We need sizes relative to M (max N)
  // Re-using capacity from n is safe as M < n
  thread_local std::vector<float> dp;
  thread_local std::vector<int64_t> cum_load;
  thread_local std::vector<float> cum_dist;
  thread_local std::vector<float> d0_vec;
  thread_local std::vector<int32_t> deque_idx; // Acts as our monotonic queue

  // Ensure sizes
  if (dp.size() <= (size_t)M) {
    size_t sz = (size_t)n + 2; // +2 for safety margins
    dp.resize(sz);
    cum_load.resize(sz);
    cum_dist.resize(sz);
    d0_vec.resize(sz);
    deque_idx.resize(sz);
  }

  // 1. Gather Data & Compute Prefix Sums (O(M))
  // This helps cache locality significantly compared to random access in the
  // loop
  cum_load[0] = 0;
  cum_dist[0] = 0.0f;

  // Prefetch first d0
  d0_vec[0] = d0[perm[0]];

  for (int32_t i = 0; i < M - 1; ++i) {
    int32_t u = perm[i];
    int32_t v = perm[i + 1];

    cum_load[i + 1] = cum_load[i] + demand_int[u];

    // dist calculation might be expensive, doing it linearly here is better for
    // CPU pipelines
    float d = dist(u, v);
    cum_dist[i + 1] = cum_dist[i] + d;

    d0_vec[i + 1] = d0[v];
  }
  // Handle last element load
  cum_load[M] = cum_load[M - 1] + demand_int[perm[M - 1]];

  // 2. Linear DP with Monotonic Deque (O(M))
  // We perform the DP where 'i' is the split point (start of new route)
  // and 'j' is the current end point.
  // Equation: DP[j] = min(Val(i)) + Const(j)
  // Val(i) = DP[i] + d0[perm[i]] - cum_dist[i]

  dp[0] = 0.0f;

  int32_t dq_head = 0;
  int32_t dq_tail = 0;

  // Initialize deque with i=0
  // Val(0) = dp[0] + d0[perm[0]] - cum_dist[0] = 0 + d0_vec[0] - 0
  float val_0 = d0_vec[0];
  deque_idx[0] = 0;
  dq_tail++;

  for (int32_t j = 1; j <= M; ++j) {
    // A. Remove indices from head that violate capacity constraint
    // We need cum_load[j] - cum_load[i] <= capacity
    // => cum_load[i] >= cum_load[j] - capacity
    int64_t min_load = cum_load[j] - capacity_int;

    while (dq_head < dq_tail && cum_load[deque_idx[dq_head]] < min_load) {
      dq_head++;
    }

    // If deque is empty here, it means the single customer j exceeds capacity
    // alone (Should not happen in valid instances, but safety check)
    if (dq_head == dq_tail) {
      // Fallback or large cost
      dp[j] = 1e30f;
    } else {
      // B. Get best i from head
      int32_t best_i = deque_idx[dq_head];

      // Calculate DP[j]
      // Cost term dependent on j: cum_dist[j-1] + d0[perm[j-1]]
      // Note: cum_dist is aligned such that cum_dist[i] is sum of edges up to
      // perm[i] path(i..j-1) distance is cum_dist[j-1] - cum_dist[i]

      // Re-eval Value(best_i) to be safe or store it? Re-calc is cheap.
      float best_val = dp[best_i] + d0_vec[best_i] - cum_dist[best_i];
      float const_j = cum_dist[j - 1] + d0_vec[j - 1];

      dp[j] = best_val + const_j;
    }

    // C. Prepare to push current j as a candidate for future steps (as split
    // point i)
    if (j < M) {
      float new_val = dp[j] + d0_vec[j] - cum_dist[j];

      // Maintain monotonicity (increasing value in deque)
      // While back of deque has Value >= new_val, pop back
      while (dq_tail > dq_head) {
        int32_t back_i = deque_idx[dq_tail - 1];
        float back_val = dp[back_i] + d0_vec[back_i] - cum_dist[back_i];
        if (back_val >= new_val) {
          dq_tail--;
        } else {
          break;
        }
      }
      deque_idx[dq_tail++] = j;
    }
  }

  return dp[M];
}

void MFACO_CVRP::decode_perm_to_route0(const int32_t *perm_ptr,
                                       std::vector<int32_t> &out_route0) const {
  std::vector<int32_t> perm(m);
  for (int32_t i = 0; i < m; ++i)
    perm[i] = perm_ptr[i];

  auto sp = split_dp(perm);
  out_route0.clear();
  out_route0.push_back(0);
  for (auto [i, j] : sp.segs) {
    for (int32_t t = i; t <= j; ++t)
      out_route0.push_back(perm[t]);
    out_route0.push_back(0);
  }
}

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

// -------------------- cycle adjacency check on source_perm
// --------------------
bool MFACO_CVRP::contains_edge(int32_t a, int32_t b) const {
  int32_t ap = source_positions[a];
  int32_t bp = source_positions[b];
  if (ap < 0 || bp < 0)
    return false;
  int32_t diff = std::abs(ap - bp);
  return diff == 1 || diff == (m - 1);
}

int32_t MFACO_CVRP::get_succ(int32_t node, const std::vector<int32_t> &perm,
                             const std::vector<int32_t> &positions) const {
  int32_t pos = positions[node];
  return perm[(pos + 1) % m];
}
int32_t MFACO_CVRP::get_pred(int32_t node, const std::vector<int32_t> &perm,
                             const std::vector<int32_t> &positions) const {
  int32_t pos = positions[node];
  return perm[(pos - 1 + m) % m];
}

// -------------------- select_next_node (same as TSP, but depot is "visited")
// --------------------
std::tuple<int32_t, bool, float>
MFACO_CVRP::select_next_node(int32_t curr, const float *probmat_row,
                             const uint8_t *visited, Xoshiro128Plus &rng,
                             int16_t &out_pick_j, uint64_t &out_valid_mask) {
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
    if (v <= 0)
      continue; // skip depot and invalid
    if (!visited[v]) {
      if (j < 64)
        out_valid_mask |= (1ULL << (uint64_t)j);
      float prod = probmat_row[j];
      cl[cl_size] = v;
      cl_prods[cl_size] = prod;
      cl_jidx[cl_size] = (int16_t)j;
      sum += prod;
      if (prod > max_prod) {
        max_prod = prod;
        max_node = v;
        max_j = (int16_t)j;
      }
      ++cl_size;
    }
  }

  bool is_stochastic = false;
  float log_prob = 0.0f;
  int32_t chosen = max_node;
  out_pick_j = max_j;
  // EPS defined in header

  if (cl_size > 1) {
    is_stochastic = true;
    float r = rng.next_float() * sum;
    float cumsum = 0.0f;
    chosen = cl[cl_size - 1];
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
    if (sum > EPS) {
      log_prob = std::log(chosen_prod / sum);
    }
  } else if (cl_size == 1) {
    // Deterministic choice from CL
    chosen = cl[0];
    out_pick_j = cl_jidx[0];
    log_prob = 0.0f; // prob = 1.0
  } else if (cl_size == 0) {
    // backup list
    bool found = false;
    for (int32_t j = 0; j < bl; ++j) {
      int32_t v = backup_list[curr * bl + j];
      if (v > 0 && !visited[v]) {
        chosen = v;
        found = true;
        break;
      }
    }
    // backup list fallback is treated as deterministic (logp=0) or we ignore
    // its prob contribution

    if (!found) {
      // global scan
      if (chosen == curr) {
        float min_dist = std::numeric_limits<float>::max();
        for (int32_t v = 1; v < n; ++v) {
          float d = dist(curr, v);
          if (!visited[v] && d < min_dist) {
            min_dist = d;
            chosen = v;
          }
        }
      }
    }
  }

  return {chosen, is_stochastic, log_prob};
}

// -------------------- relocate / 2opt / flip: identical to TSP but modulo m
// --------------------
float MFACO_CVRP::relocate_node(int32_t target, int32_t node,
                                std::vector<int32_t> &perm,
                                std::vector<int32_t> &positions) {
  if (node == target)
    return 0.0f;
  int32_t target_succ = get_succ(target, perm, positions);
  if (target_succ == node)
    return 0.0f;

  int32_t node_pos = positions[node];
  int32_t target_pos = positions[target];

  int32_t node_pred = get_pred(node, perm, positions);
  int32_t node_succ = get_succ(node, perm, positions);

  float cost_delta = -dist(node_pred, node) - dist(node, node_succ) -
                     dist(target, target_succ) + dist(node_pred, node_succ) +
                     dist(target, node) + dist(node, target_succ);

  if (target_pos < node_pos) {
    int32_t val = perm[node_pos];
    for (int32_t i = node_pos; i > target_pos + 1; --i)
      perm[i] = perm[i - 1];
    perm[target_pos + 1] = val;
    for (int32_t i = target_pos + 1; i <= node_pos; ++i)
      positions[perm[i]] = i;
  } else {
    int32_t val = perm[node_pos];
    for (int32_t i = node_pos; i < target_pos; ++i)
      perm[i] = perm[i + 1];
    perm[target_pos] = val;
    for (int32_t i = node_pos; i <= target_pos; ++i)
      positions[perm[i]] = i;
  }
  return cost_delta;
}

void MFACO_CVRP::flip_route_section(int32_t start_node, int32_t end_node,
                                    std::vector<int32_t> &perm,
                                    std::vector<int32_t> &positions) {
  int32_t first = positions[start_node];
  int32_t last = positions[end_node];
  if (first > last)
    std::swap(first, last);

  int32_t seg_len = last - first;
  int32_t rem_len = m - seg_len;

  if (seg_len <= rem_len) {
    int32_t l = first, r = last - 1;
    while (l < r) {
      std::swap(perm[l], perm[r]);
      ++l;
      --r;
    }
    for (int32_t i = first; i < last; ++i)
      positions[perm[i]] = i;
  } else {
    int32_t first_adj = (first > 0) ? first - 1 : m - 1;
    int32_t last_adj = last % m;
    std::swap(first_adj, last_adj);

    int32_t l = first_adj;
    int32_t r = last_adj;
    int32_t i = 0;
    int32_t j = m - first_adj + last_adj + 1;

    while (i < j) {
      std::swap(perm[l], perm[r]);
      positions[perm[l]] = l;
      positions[perm[r]] = r;
      l = (l + 1) % m;
      r = (r - 1 + m) % m;
      ++i;
      --j;
    }
  }
}

// -------------------- sampling ants: same MFACO logic, cost via split_dp
// --------------------

float MFACO_CVRP::two_opt_nn(
    std::vector<int32_t> &perm, std::vector<int32_t> &positions,
    std::vector<int32_t> &checklist,
    std::vector<uint8_t> &in_checklist) { // Added in_checklist
  const int32_t max_changes = 1000;
  int32_t changes = 0;
  float total_change = 0.0f;
  size_t cp = 0;

  while (cp < checklist.size() && changes < max_changes) {
    int32_t a = checklist[cp++];
    if (a <= 0 || a >= n)
      continue;

    int32_t a_next = get_succ(a, perm, positions);
    int32_t a_prev = get_pred(a, perm, positions);

    float dist_a_to_next = dist(a, a_next);
    float dist_a_to_prev = dist(a_prev, a);

    float max_diff = 0.0f;
    int32_t best_move[4] = {-1, -1, -1, -1};

    for (int32_t j = 0; j < k; ++j) {
      int32_t b = nn_list[a * k + j];
      if (b <= 0 || b >= n)
        continue;
      float dist_ab = dist(a, b);
      if (dist_a_to_next > dist_ab) {
        int32_t b_next = get_succ(b, perm, positions);
        float diff =
            dist_a_to_next + dist(b, b_next) - dist_ab - dist(a_next, b_next);
        if (diff > max_diff) {
          best_move[0] = a_next;
          best_move[1] = b_next;
          best_move[2] = a;
          best_move[3] = b;
          max_diff = diff;
        }
      }
    }

    for (int32_t j = 0; j < k; ++j) {
      int32_t b = nn_list[a * k + j];
      if (b <= 0 || b >= n)
        continue;
      float dist_ab = dist(a, b);
      if (dist_a_to_prev > dist_ab) {
        int32_t b_prev = get_pred(b, perm, positions);
        float diff =
            dist_a_to_prev + dist(b_prev, b) - dist_ab - dist(a_prev, b_prev);
        if (diff > max_diff) {
          best_move[0] = a;
          best_move[1] = b;
          best_move[2] = a_prev;
          best_move[3] = b_prev;
          max_diff = diff;
        }
      }
    }

    if (max_diff > 0) {
      flip_route_section(best_move[0], best_move[1], perm, positions);
      ++changes;
      total_change -= max_diff;

      // if extend_ls, then add endpoints to checklist
      if (extend_ls) {
        for (int32_t i = 0; i < 4; ++i) {
          int32_t node = best_move[i];
          if (node > 0 && node < n) {
            if (in_checklist[node] == 0) {
              checklist.push_back(node);
              in_checklist[node] = 1;
            }
          }
        }
      }
    }
  }

  return total_change;
}

// -------------------- sampling ants: same MFACO logic, cost via split_dp
// --------------------
float MFACO_CVRP::sample_ant_fast(const float *probmat, int32_t start_node,
                                  std::vector<int32_t> &perm_out,
                                  int32_t &new_edges_out,
                                  std::vector<int32_t> &checklist,
                                  Xoshiro128Plus &rng, const float *prior) {
  // Initialize perm as copy of source
  std::vector<int32_t> perm = source_perm;
  std::vector<int32_t> positions = source_positions; // copy

  std::vector<uint8_t> visited(n, 0);
  int32_t start_customer = (start_node == 0) ? perm[0] : start_node; // guard
  visited[start_customer] = 1;
  visited[0] = 1; // depot always visited

  int32_t visited_count = 1; // customers visited

  checklist.clear();
  checklist.push_back(start_customer);
  std::vector<uint8_t> in_checklist(n, 0);
  in_checklist[start_customer] = 1;

  int32_t new_edges = 0;
  int32_t steps = 0;
  int32_t curr = start_customer;

  while (true) {
    if (fixed_steps > 0) {
      if (steps >= fixed_steps)
        break;
    } else {
      if (new_edges >= min_new_edges || visited_count >= m)
        break;
    }
    if (visited_count >= m)
      break;

    int16_t pick_j = -1;
    uint64_t valid_mask = 0;
    auto [chosen, is_stoch, used_unif] = select_next_node(
        curr, &probmat[curr * k], visited.data(), rng, pick_j, valid_mask);

    if (chosen <= 0) {
      break;
    }

    // Check if this creates a new edge (in cycle)
    if (!contains_edge(curr, chosen)) {
      ++new_edges;
      if (in_checklist[curr] == 0) {
        checklist.push_back(curr);
        in_checklist[curr] = 1;
      }
      if (in_checklist[chosen] == 0) {
        checklist.push_back(chosen);
        in_checklist[chosen] = 1;
      }
      int32_t chosen_pred = get_pred(chosen, perm, positions);
      if (in_checklist[chosen_pred] == 0) {
        checklist.push_back(chosen_pred);
        in_checklist[chosen_pred] = 1;
      }
    }

    relocate_node(curr, chosen, perm, positions);

    visited[chosen] = 1;
    ++visited_count;
    ++steps;
    curr = chosen;
  }

  new_edges_out = new_edges;

  // Local Search
  if (use_local_search && !checklist.empty()) {
    auto start_ls = std::chrono::steady_clock::now();
    if (nls && prior) {
      two_opt_nn(perm, positions, checklist, in_checklist);

      float best_cost = split_cost_fast(perm);
      std::vector<int32_t> best_perm = perm;

      for (int t = 0; t < T_nls; ++t) {
        two_opt_nn_prior(perm, positions, checklist, in_checklist, prior);
        two_opt_nn(perm, positions, checklist, in_checklist);

        float current_cost = split_cost_fast(perm);
        if (current_cost < best_cost) {
          best_cost = current_cost;
          best_perm = perm;
        }
      }
      perm = best_perm;
    } else {
      two_opt_nn(perm, positions, checklist, in_checklist);
    }
    auto end_ls = std::chrono::steady_clock::now();
    time_ls += std::chrono::duration<double>(end_ls - start_ls).count();
  }

  // Split
  auto start_split = std::chrono::steady_clock::now();

  if (use_local_search) {
    inter_route_ls_optimized(perm, positions, checklist, in_checklist);
  }

  float cost = split_cost_fast(perm);
  auto end_split = std::chrono::steady_clock::now();
  time_split += std::chrono::duration<double>(end_split - start_split).count();

  perm_out = perm;
  return cost;
}

float MFACO_CVRP::sample_ant_traced(const float *probmat, int32_t start_node,
                                    std::vector<int32_t> &perm_out,
                                    std::vector<int32_t> &perm_raw_out,
                                    float &cost_raw_out, int32_t &new_edges_out,
                                    std::vector<int32_t> &checklist,
                                    MFACOTrace &trace, Xoshiro128Plus &rng,
                                    float &logp_sum, float &survival_out,
                                    const float *prior) {
  trace.clear();
  trace.start_node = start_node;
  trace.reserve(min_new_edges * 2);

  std::vector<int32_t> perm = source_perm;
  std::vector<int32_t> positions = source_positions;

  std::vector<uint8_t> visited(n, 0);
  int32_t start_customer = (start_node == 0) ? perm[0] : start_node;
  visited[start_customer] = 1;
  visited[0] = 1;

  int32_t visited_count = 1;

  checklist.clear();
  checklist.push_back(start_customer);
  std::vector<uint8_t> in_checklist(n, 0);
  in_checklist[start_customer] = 1;

  int32_t new_edges = 0;
  int32_t steps = 0;
  int32_t curr = start_customer;
  logp_sum = 0.0f;

  while (true) {
    if (fixed_steps > 0) {
      if (steps >= fixed_steps)
        break;
    } else {
      if (new_edges >= min_new_edges || visited_count >= m)
        break;
    }
    if (visited_count >= m)
      break;

    int16_t pick_j = -1;
    uint64_t valid_mask = 0;
    auto [chosen, is_stoch, log_prob] = select_next_node(
        curr, &probmat[curr * k], visited.data(), rng, pick_j, valid_mask);

    if (chosen <= 0)
      break;

    if (is_stoch)
      logp_sum += log_prob;

    trace.curr_nodes.push_back(curr);
    trace.chosen_nodes.push_back(chosen);
    trace.is_stochastic.push_back(is_stoch ? 1 : 0);
    trace.pick_j.push_back(pick_j);
    trace.valid_mask.push_back(valid_mask);

    if (!contains_edge(curr, chosen)) {
      ++new_edges;
      if (in_checklist[curr] == 0) {
        checklist.push_back(curr);
        in_checklist[curr] = 1;
      }
      if (in_checklist[chosen] == 0) {
        checklist.push_back(chosen);
        in_checklist[chosen] = 1;
      }
      int32_t chosen_pred = get_pred(chosen, perm, positions);
      if (in_checklist[chosen_pred] == 0) {
        checklist.push_back(chosen_pred);
        in_checklist[chosen_pred] = 1;
      }
    }

    relocate_node(curr, chosen, perm, positions);

    visited[chosen] = 1;
    ++visited_count;
    ++steps;
    curr = chosen;
  }

  new_edges_out = new_edges;

  // Capture raw
  perm_raw_out = perm;
  cost_raw_out = split_cost_fast(perm);

  // Local Search
  if (use_local_search && !checklist.empty()) {
    auto start_ls = std::chrono::steady_clock::now();
    if (nls && prior) {
      two_opt_nn(perm, positions, checklist, in_checklist);

      float best_cost = split_cost_fast(perm);
      std::vector<int32_t> best_perm = perm;

      for (int t = 0; t < T_nls; ++t) {
        // two_opt_nn_prior signature update pending, passing in_checklist
        // anyway assuming next step fixes it
        two_opt_nn_prior(perm, positions, checklist, in_checklist, prior);
        two_opt_nn(perm, positions, checklist, in_checklist);

        float current_cost = split_cost_fast(perm);
        if (current_cost < best_cost) {
          best_cost = current_cost;
          best_perm = perm;
        }
      }
      perm = best_perm;
    } else {
      two_opt_nn(perm, positions, checklist, in_checklist);
    }
    auto end_ls = std::chrono::steady_clock::now();
    time_ls += std::chrono::duration<double>(end_ls - start_ls).count();
  }

  // Split
  auto start_split = std::chrono::steady_clock::now();

  if (use_local_search) {
    inter_route_ls_optimized(perm, positions, checklist, in_checklist);
  }

  float cost = split_cost_fast(perm);
  auto end_split = std::chrono::steady_clock::now();
  time_split += std::chrono::duration<double>(end_split - start_split).count();

  perm_out = perm;

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
        // Additional check: positions should be valid (>= 0 and < m)
        if (pos_u >= 0 && pos_u < m && pos_v >= 0 && pos_v < m) {
          int32_t diff = std::abs(pos_u - pos_v);
          if (diff == 1 || diff == (m - 1)) {
            sv_num += 1.0f;
          }
        }
      }
    }
  }
  survival_out = (sv_den > 0.5f) ? (sv_num / sv_den) : 0.0f;

  return cost;
}

void MFACO_CVRP::sample(bool require_prob, const float *prior_ptr,
                        SampleResult &result, bool parallel_traced) {
  result.clear();
  result.costs.resize(n_ants);
  result.routes.resize(n_ants); // each is perm length m

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
        result.routes[a].resize(m);
        result.routes_raw[a].resize(m);

        MFACOTrace trace;
        trace.reserve(min_new_edges * 2);

        float logp_sum = 0.0f;
        int32_t mne_out = 0;
        float surv_out = 0.0f;
        float cost = sample_ant_traced(
            probmat.data(), start_nodes[a], result.routes[a],
            result.routes_raw[a], result.costs_raw[a], mne_out, checklist,
            trace, rng_, logp_sum, surv_out, prior_ptr);
        result.costs[a] = cost;
        result.new_edges_count[a] = mne_out;
        result.edge_survival[a] = surv_out;
        result.logps[a] = logp_sum;

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
          result.routes[a].resize(m);
          result.routes_raw[a].resize(m);
          MFACOTrace &trace = traces_per_ant[(size_t)a];
          trace.reserve(min_new_edges * 2);
          Xoshiro128Plus rng_local;
          rng_local.seed(ant_seeds[(size_t)a]);

          float logp_sum = 0.0f;
          int32_t mne_out = 0;
          float surv_out = 0.0f;
          result.costs[a] = sample_ant_traced(
              probmat.data(), start_nodes[a], result.routes[a],
              result.routes_raw[a], result.costs_raw[a], mne_out, checklist,
              trace, rng_local, logp_sum, surv_out, prior_ptr);
          result.new_edges_count[a] = mne_out;
          result.edge_survival[a] = surv_out;
          result.logps[a] = logp_sum;
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
        result.routes[a].resize(m);
        Xoshiro128Plus rng_local;
        rng_local.seed(ant_seeds[(size_t)a]);
        result.costs[a] = sample_ant_fast(
            probmat.data(), start_nodes[a], result.routes[a],
            result.new_edges_count[a], checklist, rng_local, prior_ptr);
      }
    }
  }
}

// -------------------- update pheromone: deposit on decoded VRP edges
// --------------------
void MFACO_CVRP::update_pheromone(const int32_t *iter_best_perm_ptr,
                                  float iter_best_cost) {
  // 1) Update global best-so-far
  if (iter_best_cost < best_cost) {
    best_cost = iter_best_cost;
    std::copy(iter_best_perm_ptr, iter_best_perm_ptr + m, best_perm.begin());
  }

  // 2) Trail limits typically based on global best-so-far (MMAS)
  auto [tmin, tmax] = smooth_mmas ? calc_trail_limits_smooth(best_cost)
                                  : calc_trail_limits_cl(best_cost);
  tau_min = tmin;
  tau_max = tmax;

  // 3) Evaporate
  if (!smooth_mmas) {
    float decay_factor = 1.0f - rho;
    for (int32_t i = 0; i < n * k; ++i) {
      pheromone_sparse[i] *= decay_factor;
      pheromone_sparse[i] =
          std::max(tau_min, std::min(tau_max, pheromone_sparse[i]));
    }

    // 4) Deposit ON ITERATION BEST (consistent with deposit magnitude)
    std::vector<int32_t> perm(iter_best_perm_ptr, iter_best_perm_ptr + m);
    auto sp = split_dp(perm);
    float deposit = 1.0f / (iter_best_cost + EPS);

    // Deposit depot edges + within-segment edges
    for (auto [i, j] : sp.segs) {
      int32_t first = perm[i];
      int32_t last = perm[j];

      // depot -> first
      int32_t jd = find_neighbor_index(0, first);
      if (jd >= 0)
        pheromone_sparse[0 * k + jd] =
            std::min(pheromone_sparse[0 * k + jd] + deposit, tau_max);

      int32_t jdr = find_neighbor_index(first, 0);
      if (jdr >= 0)
        pheromone_sparse[first * k + jdr] =
            std::min(pheromone_sparse[first * k + jdr] + deposit, tau_max);

      // internal edges
      for (int32_t t = i; t < j; ++t) {
        int32_t u = perm[t];
        int32_t v = perm[t + 1];

        int32_t ju = find_neighbor_index(u, v);
        if (ju >= 0)
          pheromone_sparse[u * k + ju] =
              std::min(pheromone_sparse[u * k + ju] + deposit, tau_max);

        int32_t jv = find_neighbor_index(v, u);
        if (jv >= 0)
          pheromone_sparse[v * k + jv] =
              std::min(pheromone_sparse[v * k + jv] + deposit, tau_max);
      }

      // last -> depot
      int32_t jl = find_neighbor_index(last, 0);
      if (jl >= 0)
        pheromone_sparse[last * k + jl] =
            std::min(pheromone_sparse[last * k + jl] + deposit, tau_max);

      int32_t jlr = find_neighbor_index(0, last);
      if (jlr >= 0)
        pheromone_sparse[0 * k + jlr] =
            std::min(pheromone_sparse[0 * k + jlr] + deposit, tau_max);
    }
  } else {
    // Smooth-MMAS
    std::vector<int32_t> perm(iter_best_perm_ptr, iter_best_perm_ptr + m);
    auto sp = split_dp(perm);

    // Identify edges in the best solution
    // Since n is small (up to 2000-5000), we can use a dense adjacency matrix
    // or vector of vectors. Or since we only iterate n*k edges in sparse
    // matrix, we can just check existence. Building a hash set or sorted
    // vector for fast lookup is better. Given sparse graph, maybe just
    // marking nodes? No, edges. Let's use `std::vector<std::vector<int32_t>>
    // adj(n)` just for the solution edges.

    std::vector<std::vector<int32_t>> sol_adj(n);
    auto add_edge = [&](int32_t u, int32_t v) {
      sol_adj[u].push_back(v);
      sol_adj[v].push_back(u);
    };

    for (auto [i, j] : sp.segs) {
      int32_t first = perm[i];
      int32_t last = perm[j];
      add_edge(0, first);
      for (int32_t t = i; t < j; ++t) {
        add_edge(perm[t], perm[t + 1]);
      }
      add_edge(last, 0);
    }

    auto is_in_sol = [&](int32_t u, int32_t v) {
      for (int32_t x : sol_adj[u]) {
        if (x == v)
          return true;
      }
      return false;
    };

    float keep = 1.0f - rho;
    for (int32_t u = 0; u < n; ++u) {
      for (int32_t j = 0; j < k; ++j) {
        int32_t v = nn_list[u * k + j];
        if (v < 0)
          continue; // should check if v is valid node index

        float target = is_in_sol(u, v) ? tau_max : tau_min;
        float &tau = pheromone_sparse[u * k + j];
        tau = keep * tau + rho * target;
      }
    }
  }

  // Update source for next iteration to iteration best
  source_perm.assign(iter_best_perm_ptr, iter_best_perm_ptr + m);
  source_cost = iter_best_cost;

  std::fill(source_positions.begin(), source_positions.end(), -1);
  for (int32_t i = 0; i < m; ++i)
    source_positions[source_perm[i]] = i;
}

float MFACO_CVRP::two_opt_nn_prior(std::vector<int32_t> &perm,
                                   std::vector<int32_t> &positions,
                                   std::vector<int32_t> &checklist,
                                   std::vector<uint8_t> &in_checklist,
                                   const float *prior_ptr) {
  int32_t changes_count = 0;
  float total_gain = 0.0f;
  size_t checklist_pos = 0;

  auto get_prior = [&](int32_t u, int32_t v) -> float {
    // u, v are nodes 1..n-1. Virtual depots (>=n) map to 0.
    int32_t u_real = (u >= n) ? 0 : u;
    int32_t v_real = (v >= n) ? 0 : v;
    int32_t idx = find_neighbor_index(u_real, v_real);
    if (idx >= 0) {
      return prior_ptr[u_real * k + idx];
    }
    return -1e9f;
  };

  while (checklist_pos < checklist.size()) {
    int32_t a = checklist[checklist_pos++];
    if (a <= 0 || a >= n)
      continue;

    int32_t a_next = get_succ(a, perm, positions);
    int32_t a_prev = get_pred(a, perm, positions);

    // Prior values
    float prior_a_next = get_prior(a, a_next);
    float prior_a_prev = get_prior(a_prev, a);

    // printf("DEBUG: a=%d, a_real=%d, a_next=%d, a_prev=%d\n", a, (a >= n) ? 0
    // : a, a_next, a_prev);

    float max_gain = 0.0f;
    int32_t best_move[4] = {-1, -1, -1, -1};

    // Check moves with a -> a_next edge using nn_list
    int32_t a_real = (a >= n) ? 0 : a;
    for (int32_t j = 0; j < k; ++j) {
      int32_t b = nn_list[a_real * k + j];
      if (b <= 0 || b >= n)
        continue;

      float prior_ab = get_prior(a, b);
      int32_t b_next = get_succ(b, perm, positions);
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
      int32_t b = nn_list[a_real * k + j];
      if (b <= 0 || b >= n)
        continue;

      float prior_ab = get_prior(a, b);
      int32_t b_prev = get_pred(b, perm, positions);
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
      flip_route_section(best_move[0], best_move[1], perm, positions);
      ++changes_count;
      total_gain += max_gain;

      if (extend_ls) {
        for (int32_t i = 0; i < 4; ++i) {
          int32_t node = best_move[i];
          if (in_checklist[node] == 0) {
            checklist.push_back(node);
            in_checklist[node] = 1;
          }
        }
      }
    }
  }
  return total_gain;
}

// -------------------- Inter-route LS --------------------

std::vector<std::vector<int32_t>>
MFACO_CVRP::initial_routes_from_perm(const std::vector<int32_t> &perm) const {
  auto res = split_dp(perm);
  std::vector<std::vector<int32_t>> routes;
  routes.reserve(res.segs.size());
  for (auto &seg : res.segs) {
    std::vector<int32_t> r;
    r.reserve(seg.second - seg.first + 3);
    r.push_back(0); // Depot
    for (int i = seg.first; i <= seg.second; ++i) {
      r.push_back(perm[i]);
    }
    r.push_back(0); // Depot
    routes.push_back(r);
  }
  return routes;
}

void MFACO_CVRP::routes_to_perm(const std::vector<std::vector<int32_t>> &routes,
                                std::vector<int32_t> &perm,
                                std::vector<int32_t> &positions) {
  perm.clear();
  perm.reserve(m);
  positions.assign(n, -1);
  int32_t idx = 0;
  for (const auto &r : routes) {
    // skip first (0) and last (0)
    for (size_t i = 1; i < r.size() - 1; ++i) {
      int32_t node = r[i];
      perm.push_back(node);
      positions[node] = idx++;
    }
  }
}

// Helpers for LS
static inline float dist_sq(float x1, float y1, float x2, float y2) {
  float dx = x1 - x2;
  float dy = y1 - y2;
  return dx * dx + dy * dy;
}

bool MFACO_CVRP::ls_relocate(std::vector<std::vector<int32_t>> &routes,
                             std::vector<int64_t> &loads, int32_t u_node,
                             int32_t r_u, int32_t idx_u, int32_t r_v,
                             int32_t idx_v, std::vector<int32_t> &node_pos) {
  // idx_v is the INDEX of the node V. We insert u AFTER V.
  // v_node can be derived from routes[r_v][idx_v].
  int32_t v_node = routes[r_v][idx_v];

  // Try to insert u after v
  if (r_u != r_v) {
    if (loads[r_v] + demand_int[u_node] > capacity_int)
      return false;
  }

  auto &route_u = routes[r_u];
  auto &route_v = routes[r_v];

  // If same route, check standard relocate checks
  if (r_u == r_v) {
    // If v is u's predecessor, no change (insert u after prev(u) is no-op)
    if (idx_v == idx_u - 1)
      return false;
    // If v is u (insert u after u??) -> invalid, no-op
    if (idx_v == idx_u)
      return false;
  }

  // Calculate delta
  int32_t u_prev = route_u[idx_u - 1];
  int32_t u_next = route_u[idx_u + 1];
  int32_t v_next = route_v[idx_v + 1]; // Insert u between v and v_next

  float delta = dist(u_prev, u_next) - dist(u_prev, u_node) -
                dist(u_node, u_next) + dist(v_node, u_node) +
                dist(u_node, v_next) - dist(v_node, v_next);

  if (delta < -1e-6f) {
    // Apply
    // Remove u from r_u
    route_u.erase(route_u.begin() + idx_u);
    // Update indices for r_u elements after idx_u (indices shift left by 1)
    for (size_t i = idx_u; i < route_u.size(); ++i) {
      if (route_u[i] != 0)
        node_pos[route_u[i]] = (int32_t)i;
    }

    // Be careful with indices if r_u == r_v
    if (r_u == r_v) {
      if (idx_u < idx_v) {
        // u removed before v, so v index shifts down by 1
        idx_v--;
      }
      routes[r_u].insert(routes[r_u].begin() + idx_v + 1, u_node);

      // Update indices for elements after insertion (indices shift right by 1)
      // Insertion at idx_v + 1. u is at idx_v + 1.
      for (size_t i = idx_v + 1; i < routes[r_u].size(); ++i) {
        if (routes[r_u][i] != 0)
          node_pos[routes[r_u][i]] = (int32_t)i;
      }
    } else {
      routes[r_v].insert(routes[r_v].begin() + idx_v + 1, u_node);
      loads[r_u] -= demand_int[u_node];
      loads[r_v] += demand_int[u_node];

      // Update indices for r_v elements after insertion
      for (size_t i = idx_v + 1; i < routes[r_v].size(); ++i) {
        if (routes[r_v][i] != 0)
          node_pos[routes[r_v][i]] = (int32_t)i;
      }
    }
    return true;
  }
  return false;
}

bool MFACO_CVRP::ls_swap(std::vector<std::vector<int32_t>> &routes,
                         std::vector<int64_t> &loads, int32_t u_node,
                         int32_t r_u, int32_t idx_u, int32_t r_v, int32_t idx_v,
                         std::vector<int32_t> &node_pos) {
  if (r_u == r_v) {
    return false;
  }

  // Implicitly, idx_v is the node to swap with.
  // v_node might be 0? If so, swapping depot is invalid for ls_swap typically.
  int32_t v_node = routes[r_v][idx_v];
  if (v_node == 0)
    return false; // Cannot swap with depot

  // Swap u and v
  int64_t d_u = demand_int[u_node];
  int64_t d_v = demand_int[v_node];

  if (loads[r_u] - d_u + d_v > capacity_int)
    return false;
  if (loads[r_v] - d_v + d_u > capacity_int)
    return false;

  auto &route_u = routes[r_u];
  auto &route_v = routes[r_v];

  int32_t u_prev = route_u[idx_u - 1];
  int32_t u_next = route_u[idx_u + 1];
  int32_t v_prev = route_v[idx_v - 1];
  int32_t v_next = route_v[idx_v + 1];

  float delta = -dist(u_prev, u_node) - dist(u_node, u_next) -
                dist(v_prev, v_node) - dist(v_node, v_next) +
                dist(u_prev, v_node) + dist(v_node, u_next) +
                dist(v_prev, u_node) + dist(u_node, v_next);

  if (delta < -1e-6f) {
    std::swap(route_u[idx_u], route_v[idx_v]);

    // Update node_pos
    std::swap(node_pos[u_node], node_pos[v_node]);

    loads[r_u] = loads[r_u] - d_u + d_v;
    loads[r_v] = loads[r_v] - d_v + d_u;
    return true;
  }
  return false;
}

bool MFACO_CVRP::ls_2opt_star(std::vector<std::vector<int32_t>> &routes,
                              std::vector<int64_t> &loads, int32_t u_node,
                              int32_t r_u, int32_t idx_u, int32_t r_v,
                              int32_t idx_v, std::vector<int32_t> &node_pos,
                              std::vector<int32_t> &node_to_route) {
  if (r_u == r_v)
    return false;

  auto &route_u = routes[r_u];
  auto &route_v = routes[r_v];

  // We cut AFTER idx_u and AFTER idx_v.
  // v_node is routes[r_v][idx_v]. Can be 0 (cut after depot).
  int32_t v_node = routes[r_v][idx_v];

  // Tail U load: sum demand from idx_u+1 to end-1
  int64_t tail_u_load = 0;
  for (size_t i = idx_u + 1; i < route_u.size() - 1; ++i)
    tail_u_load += demand_int[route_u[i]];

  int64_t tail_v_load = 0;
  for (size_t i = idx_v + 1; i < route_v.size() - 1; ++i)
    tail_v_load += demand_int[route_v[i]];

  int64_t new_load_u = (loads[r_u] - tail_u_load) + tail_v_load;
  int64_t new_load_v = (loads[r_v] - tail_v_load) + tail_u_load;

  if (new_load_u > capacity_int || new_load_v > capacity_int)
    return false;

  int32_t u_next = route_u[idx_u + 1];
  int32_t v_next = route_v[idx_v + 1];

  float delta = -dist(u_node, u_next) - dist(v_node, v_next) +
                dist(u_node, v_next) + dist(v_node, u_next);

  if (delta < -1e-6f) {
    // Perform swap
    std::vector<int32_t> new_tail_u(route_v.begin() + idx_v + 1, route_v.end());
    std::vector<int32_t> new_tail_v(route_u.begin() + idx_u + 1, route_u.end());

    route_u.resize(idx_u + 1);
    route_u.insert(route_u.end(), new_tail_u.begin(), new_tail_u.end());

    route_v.resize(idx_v + 1);
    route_v.insert(route_v.end(), new_tail_v.begin(), new_tail_v.end());

    loads[r_u] = new_load_u;
    loads[r_v] = new_load_v;

    // Update node_pos and node_to_route for moved nodes
    // Nodes from new_tail_u came from route_v, now in route_u
    for (size_t i = idx_u + 1; i < route_u.size() - 1; ++i) {
      int32_t node = route_u[i];
      if (node != 0) {
        node_pos[node] = (int32_t)i;
        node_to_route[node] = r_u;
      }
    }
    // Nodes from new_tail_v came from route_u, now in route_v
    for (size_t i = idx_v + 1; i < route_v.size() - 1; ++i) {
      int32_t node = route_v[i];
      if (node != 0) {
        node_pos[node] = (int32_t)i;
        node_to_route[node] = r_v;
      }
    }
    // Update node_to_route: we processed swapped segments. Depots (start, end)
    // are fine.

    return true;
  }
  return false;
}

void MFACO_CVRP::inter_route_ls(std::vector<int32_t> &perm,
                                std::vector<int32_t> &positions,
                                std::vector<int32_t> &checklist,
                                std::vector<uint8_t> &in_checklist) {
  // Pruning removed by user request
  // Only if best_cost is initialized (>0)

  // 1. Initial Routes
  auto routes = initial_routes_from_perm(perm);

  // Compute loads map and initialize node_pos
  std::vector<int64_t> loads(routes.size(), 0);
  std::vector<int32_t> node_to_route(n, -1);
  std::vector<int32_t> node_pos(n, -1);

  for (size_t r = 0; r < routes.size(); ++r) {
    for (size_t i = 1; i < routes[r].size() - 1; ++i) {
      int32_t node = routes[r][i];
      loads[r] += demand_int[node];
      node_to_route[node] = (int32_t)r;
      node_pos[node] = (int32_t)i;
    }
  }

  // 2. Focused Loop
  size_t checklist_pos = 0;

  while (checklist_pos < checklist.size()) {
    int32_t u = checklist[checklist_pos++];
    if (u <= 0 || u >= n)
      continue;

    int32_t r_u = node_to_route[u];
    if (r_u < 0)
      continue;

    // Check neighbors
    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_list[u * k + j];
      if (v <= 0)
        continue;

      int32_t r_v = node_to_route[v];
      if (r_v < 0)
        continue;

      int32_t move_type = 0;

      int32_t idx_u = node_pos[u];
      int32_t idx_v = node_pos[v];

      // Capture neighbors before move for checklist update
      int32_t p_u = routes[r_u][idx_u - 1];
      int32_t s_u = routes[r_u][idx_u + 1];
      int32_t p_v = routes[r_v][idx_v - 1];
      int32_t s_v = routes[r_v][idx_v + 1];

      // Try Relocate u -> v (insert u after v)
      if (use_relocate &&
          ls_relocate(routes, loads, u, r_u, idx_u, r_v, idx_v, node_pos)) {
        move_type = 1; // Relocate u after v
      }
      // Try Relocate u -> prev(v) (insert u before v)
      else if (use_relocate && ls_relocate(routes, loads, u, r_u, idx_u, r_v,
                                           idx_v - 1, node_pos)) {
        move_type = 2; // Relocate u before v
      }
      // Try Relocate v -> u (insert v after u)
      else if (use_relocate && ls_relocate(routes, loads, v, r_v, idx_v, r_u,
                                           idx_u, node_pos)) {
        move_type = 3; // Relocate v after u
      }
      // Try Relocate v -> prev(u) (insert v before u)
      else if (use_relocate && ls_relocate(routes, loads, v, r_v, idx_v, r_u,
                                           idx_u - 1, node_pos)) {
        move_type = 4; // Relocate v before u
      }
      // Try Swap
      else if (use_swap &&
               ls_swap(routes, loads, u, r_u, idx_u, r_v, idx_v, node_pos)) {
        move_type = 5; // Swap
      }
      // Try 2-opt* (swap tails: u->v_next)
      else if (use_2opt_star && ls_2opt_star(routes, loads, u, r_u, idx_u, r_v,
                                             idx_v, node_pos, node_to_route)) {
        move_type = 6; // 2-opt* u, v
      }
      // Try 2-opt* (swap tails: u->v) (Cut after u, Cut after prev(v))
      else if (use_2opt_star &&
               ls_2opt_star(routes, loads, u, r_u, idx_u, r_v, idx_v - 1,
                            node_pos, node_to_route)) {
        move_type = 7; // 2-opt* u, v-1
      }
      // Try 2-opt* (swap tails: v->u_next)
      else if (use_2opt_star && ls_2opt_star(routes, loads, v, r_v, idx_v, r_u,
                                             idx_u, node_pos, node_to_route)) {
        move_type = 8; // 2-opt* v, u
      } else if (use_2opt_star &&
                 ls_2opt_star(routes, loads, v, r_v, idx_v, r_u, idx_u - 1,
                              node_pos, node_to_route)) {
        move_type = 9; // 2-opt* v, u-1
      }

      if (move_type > 0) {
        // Update node_to_route
        if (move_type >= 1 && move_type <= 4) { // Relocate
          if (move_type <= 2) {                 // u moved
            node_to_route[u] = r_v;
            r_u = r_v;
          } else { // v moved
            node_to_route[v] = r_u;
          }
        } else if (move_type == 5) { // Swap
          node_to_route[u] = r_v;
          node_to_route[v] = r_u;
          int32_t tmp = r_u;
          r_u = r_v;
          r_v = tmp;
        }
        // 2-opt* updates node_to_route internally

        // Extend LS: Add involved nodes to checklist
        if (extend_ls) {
          auto add_chk = [&](int32_t node) {
            if (node > 0 && node < n && in_checklist[node] == 0) {
              checklist.push_back(node);
              in_checklist[node] = 1;
            }
          };

          // Always check u and v
          add_chk(u);
          add_chk(v);

          // Specific neighbors based on move type
          if (move_type == 1) { // Relocate u after v
            add_chk(p_u);
            add_chk(s_u);
            add_chk(s_v);
          } else if (move_type == 2) { // Relocate u before v (after p_v)
            add_chk(p_u);
            add_chk(s_u);
            add_chk(p_v);
          } else if (move_type == 3) { // Relocate v after u
            add_chk(p_v);
            add_chk(s_v);
            add_chk(s_u);
          } else if (move_type == 4) { // Relocate v before u (after p_u)
            add_chk(p_v);
            add_chk(s_v);
            add_chk(p_u);
          } else if (move_type == 5) { // Swap
            add_chk(p_u);
            add_chk(s_u);
            add_chk(p_v);
            add_chk(s_v);
          } else if (move_type == 6) { // 2-opt* u, v
            add_chk(s_u);
            add_chk(s_v);
          } else if (move_type == 7) { // 2-opt* u, v-1
            add_chk(s_u);
            add_chk(p_v);
          } else if (move_type == 8) { // 2-opt* v, u
            add_chk(s_v);
            add_chk(s_u);
          } else if (move_type == 9) { // 2-opt* v, u-1
            add_chk(s_v);
            add_chk(p_u);
          }
        }
        break;
      }
    }
  }

  routes_to_perm(routes, perm, positions);
}

// ============================================================================
// Optimized Inter-Route Local Search (Linked List + DLB + O(1) Delta)
// ============================================================================

void MFACO_CVRP::inter_route_ls_optimized(std::vector<int32_t> &perm,
                                          std::vector<int32_t> &positions,
                                          std::vector<int32_t> &checklist,
                                          std::vector<uint8_t> &in_checklist) {
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

  if (checklist.empty()) {
    for (int32_t i = 1; i < n; ++i)
      touch(i);
  }

  // Get current routes
  auto res = split_dp(perm);
  int32_t num_routes = res.segs.size();

  if (num_routes > n) {
    // Just in case, though guaranteed by split logic usually
    next_node.resize(n + num_routes);
    prev_node.resize(n + num_routes);
    node_route.resize(n + num_routes);
    cum_demand.resize(n + num_routes);
  }
  route_loads.resize(num_routes);

  // Build Linked List
  for (int r = 0; r < num_routes; ++r) {
    int32_t depot = n + r;
    int32_t start_idx = res.segs[r].first;
    int32_t end_idx = res.segs[r].second;

    int32_t prev = depot;
    int64_t load = 0;

    node_route[depot] = r;
    cum_demand[depot] = 0;

    for (int i = start_idx; i <= end_idx; ++i) {
      int32_t u = perm[i];
      next_node[prev] = u;
      prev_node[u] = prev;
      node_route[u] = r;
      load += demand_int[u];
      cum_demand[u] = load;
      prev = u;
    }
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
            break;
          }
        }
      }
    }
    if (!improved)
      dlb[u] = true;
  }

  // 3. Reconstruct Permutation
  perm.clear();
  perm.reserve(m);
  for (int r = 0; r < num_routes; ++r) {
    int32_t depot = n + r;
    int32_t curr = next_node[depot];
    while (curr < n) {
      perm.push_back(curr);
      curr = next_node[curr];
    }
  }
}
} // namespace mfaco
