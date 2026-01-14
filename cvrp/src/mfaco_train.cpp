// mfaco_cvrp.cpp (or inside mfaco_train.cpp)
#include "mfaco_train.h"
#include "kd_tree.h"
#include <omp.h>
#include <cstring>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace mfaco {

// You already have EPS / MAX_CAND_LIST_SIZE in your project.
// If not, define them appropriately.
#ifndef EPS
#define EPS 1e-12f
#endif

MFACO_CVRP::MFACO_CVRP(
    const float* coords_ptr,
    const float* demand_ptr,
    int32_t n_,
    float capacity_,
    int32_t n_ants_,
    int32_t cand_list_size,
    int32_t backup_list_size,
    int32_t min_new_edges_,
    float decay,
    float alpha_,
    float p_best_,
    bool use_local_search_,
    bool disable_heuristic_
) : n(n_),
    m(n_ - 1),
    n_ants(n_ants_),
    k(std::min(cand_list_size, n_ - 1)),
    bl(std::min(backup_list_size, std::max(0, n_ - 1 - k))),
    min_new_edges(min_new_edges_),
    rho(decay),
    alpha(alpha_),
    p_best(p_best_),
    use_local_search(use_local_search_),
    disable_heuristic(disable_heuristic_),
    capacity(capacity_)
{
    if (!coords_ptr || !demand_ptr) {
        throw std::runtime_error("coords_ptr and demand_ptr must not be null");
    }
    if (n < 2) throw std::runtime_error("n must be >= 2 (depot + at least one customer)");
    if (capacity <= 0) throw std::runtime_error("capacity must be > 0");

    coords.resize(static_cast<size_t>(n) * 2);
    std::memcpy(coords.data(), coords_ptr, sizeof(float) * static_cast<size_t>(n) * 2);

    demand.resize(static_cast<size_t>(n));
    std::memcpy(demand.data(), demand_ptr, sizeof(float) * static_cast<size_t>(n));
    demand[0] = 0.0f; // enforce

    build_nn_lists();
    build_nn_pos();
    build_heuristic();

    source_perm.resize(m);
    best_perm.resize(m);
    source_positions.assign(n, -1);

    build_initial_perm();

    auto [tmin, tmax] = calc_trail_limits_cl(source_cost);
    tau_min = tmin;
    tau_max = tmax;
    pheromone_sparse.assign(n * k, tau_max);

    rng_.seed(42);
}

void MFACO_CVRP::seed_rng(uint64_t seed) { rng_.seed(seed); }

// -------------------- distance --------------------
float MFACO_CVRP::dist(int32_t u, int32_t v) const {
    const size_t ou = static_cast<size_t>(u) * 2;
    const size_t ov = static_cast<size_t>(v) * 2;
    float dx = coords[ou] - coords[ov];
    float dy = coords[ou + 1] - coords[ov + 1];
    return std::sqrt(dx*dx + dy*dy);
}

// -------------------- NN lists (same as TSP) --------------------
void MFACO_CVRP::build_nn_lists() {
    nn_list.resize(n * k);
    backup_list.resize(n * bl);

    const int32_t total = k + bl;
    if (total <= 0) return;

    std::vector<Vec2d> pts(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        const size_t off = static_cast<size_t>(i) * 2;
        pts[static_cast<size_t>(i)] = Vec2d{ (double)coords[off], (double)coords[off+1] };
    }

    KDTree shared_kdtree(pts, /*round_distances=*/false);

    #pragma omp parallel default(none) shared(shared_kdtree, nn_list, backup_list) firstprivate(n, k, bl, total)
    {
        KDTree kdtree = shared_kdtree;

        #pragma omp for schedule(static)
        for (int32_t u = 0; u < n; ++u) {
            for (int32_t j = 0; j < total; ++j) {
                uint32_t pt_idx = kdtree.nn_bottom_up(static_cast<uint32_t>(u));
                if (j < k) nn_list[u * k + j] = (int32_t)pt_idx;
                else       backup_list[u * bl + (j - k)] = (int32_t)pt_idx;
                kdtree.delete_point(pt_idx);
            }
            for (int32_t j = 0; j < k; ++j) kdtree.undelete_point((uint32_t)nn_list[u * k + j]);
            for (int32_t j = 0; j < bl; ++j) kdtree.undelete_point((uint32_t)backup_list[u * bl + j]);
        }
    }
}

void MFACO_CVRP::build_nn_pos() {
    nn_pos.assign(n * n, -1);
    for (int32_t u = 0; u < n; ++u) {
        for (int32_t j = 0; j < k; ++j) {
            int32_t v = nn_list[u * k + j];
            nn_pos[u * n + v] = j;
        }
    }
}

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

// -------------------- initial perm (greedy NN on customers, score by split) --------------------
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
                if (v > 0 && !visited[v]) { nxt = v; break; }
            }
            if (nxt < 0) {
                float md = std::numeric_limits<float>::max();
                for (int32_t v = 1; v < n; ++v) {
                    if (visited[v]) continue;
                    float d = dist(curr, v);
                    if (d < md) { md = d; nxt = v; }
                }
            }
            perm[i] = nxt;
            visited[nxt] = 1;
        }

        float cost = split_dp(perm).cost;
        if (cost < best) { best = cost; best_p = perm; }
    }

    source_perm = best_p;
    best_perm = best_p;
    source_cost = best;
    best_cost = best;

    std::fill(source_positions.begin(), source_positions.end(), -1);
    for (int32_t i = 0; i < m; ++i) source_positions[source_perm[i]] = i;
}

// -------------------- split DP: permutation -> optimal route partition --------------------
MFACO_CVRP::SplitResult MFACO_CVRP::split_dp(const std::vector<int32_t>& perm) const {
    const int32_t M = (int32_t)perm.size();
    SplitResult r;
    r.cost = std::numeric_limits<float>::max();
    r.segs.clear();
    if (M == 0) { r.cost = 0.0f; return r; }

    std::vector<float> pref_d(M + 1, 0.0f);
    for (int32_t i = 0; i < M; ++i) pref_d[i + 1] = pref_d[i] + demand[perm[i]];

    std::vector<float> pref_e(M, 0.0f);
    for (int32_t i = 1; i < M; ++i) pref_e[i] = pref_e[i - 1] + dist(perm[i - 1], perm[i]);

    auto seg_load = [&](int32_t i, int32_t j) {
        return pref_d[j + 1] - pref_d[i];
    };
    auto seg_cost = [&](int32_t i, int32_t j) {
        float internal = (i == j) ? 0.0f : (pref_e[j] - pref_e[i]);
        return dist(0, perm[i]) + internal + dist(perm[j], 0);
    };

    const float INF = 1e30f;
    std::vector<float> dp(M + 1, INF);
    std::vector<int32_t> prev(M + 1, -1);
    dp[0] = 0.0f;

    for (int32_t t = 1; t <= M; ++t) {
        int32_t j = t - 1;
        float bestv = INF;
        int32_t besti = -1;
        for (int32_t i = 0; i < t; ++i) {
            if (seg_load(i, j) <= capacity + 1e-12f) {
                float v = dp[i] + seg_cost(i, j);
                if (v < bestv) { bestv = v; besti = i; }
            }
        }
        dp[t] = bestv;
        prev[t] = besti;
    }

    // backtrack
    std::vector<std::pair<int32_t,int32_t>> segs;
    int32_t t = M;
    while (t > 0) {
        int32_t i = prev[t];
        if (i < 0) {
            // infeasible permutation given capacity (should not happen if instance feasible)
            r.cost = dp[M];
            r.segs.clear();
            return r;
        }
        segs.push_back({i, t - 1});
        t = i;
    }
    std::reverse(segs.begin(), segs.end());

    r.cost = dp[M];
    r.segs = std::move(segs);
    return r;
}

void MFACO_CVRP::decode_perm_to_route0(const int32_t* perm_ptr, std::vector<int32_t>& out_route0) const {
    std::vector<int32_t> perm(m);
    for (int32_t i = 0; i < m; ++i) perm[i] = perm_ptr[i];

    auto sp = split_dp(perm);
    out_route0.clear();
    out_route0.push_back(0);
    for (auto [i,j] : sp.segs) {
        for (int32_t t = i; t <= j; ++t) out_route0.push_back(perm[t]);
        out_route0.push_back(0);
    }
}

// -------------------- pheromone bounds (same formula as TSP) --------------------
std::pair<float,float> MFACO_CVRP::calc_trail_limits_cl(float solution_cost) const {
    float tau_max_ = 1.0f / (solution_cost * (1.0f - rho) + EPS);
    float avg = static_cast<float>(std::max(2, k));
    float p = std::pow(p_best, 1.0f / avg);
    float tau_min_ = std::min(tau_max_, tau_max_ * (1.0f - p) / ((avg - 1.0f) * p + EPS));
    return {tau_min_, tau_max_};
}

// -------------------- probmat (same as TSP) --------------------
void MFACO_CVRP::compute_probmat(const float* residual_logits, std::vector<float>& probmat) {
    probmat.resize((size_t)n * (size_t)k);
    for (int32_t u = 0; u < n; ++u) {
        for (int32_t j = 0; j < k; ++j) {
            int32_t idx = u * k + j;
            float tau = pheromone_sparse[idx];
            float eta = heuristic_sparse[idx];
            float w = std::pow(tau + EPS, alpha) * eta;
            if (residual_logits) {
                float r = residual_logits[idx];
                r = std::max(-10.0f, std::min(10.0f, r));
                w *= std::exp(r);
            }
            probmat[idx] = std::max(w, EPS);
        }
    }
}

// -------------------- cycle adjacency check on source_perm --------------------
bool MFACO_CVRP::contains_edge(int32_t a, int32_t b) const {
    int32_t ap = source_positions[a];
    int32_t bp = source_positions[b];
    if (ap < 0 || bp < 0) return false;
    int32_t diff = std::abs(ap - bp);
    return diff == 1 || diff == (m - 1);
}

int32_t MFACO_CVRP::get_succ(int32_t node, const std::vector<int32_t>& perm, const std::vector<int32_t>& positions) const {
    int32_t pos = positions[node];
    return perm[(pos + 1) % m];
}
int32_t MFACO_CVRP::get_pred(int32_t node, const std::vector<int32_t>& perm, const std::vector<int32_t>& positions) const {
    int32_t pos = positions[node];
    return perm[(pos - 1 + m) % m];
}

// -------------------- select_next_node (same as TSP, but depot is "visited") --------------------
std::tuple<int32_t, bool, bool> MFACO_CVRP::select_next_node(
    int32_t curr,
    const float* probmat_row,
    const uint8_t* visited,
    Xoshiro128Plus& rng,
    int16_t& out_pick_j,
    uint64_t& out_valid_mask
) {
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
        if (v <= 0) continue;          // skip depot and invalid
        if (!visited[v]) {
            if (j < 64) out_valid_mask |= (1ULL << (uint64_t)j);
            float prod = probmat_row[j];
            cl[cl_size] = v;
            cl_prods[cl_size] = prod;
            cl_jidx[cl_size] = (int16_t)j;
            sum += prod;
            if (prod > max_prod) { max_prod = prod; max_node = v; max_j = (int16_t)j; }
            ++cl_size;
        }
    }

    bool is_stochastic = false;
    bool used_uniform = false;
    int32_t chosen = max_node;
    out_pick_j = max_j;

    if (cl_size > 1) {
        is_stochastic = true;
        float r = rng.next_float() * sum;
        float cumsum = 0.0f;
        chosen = cl[cl_size - 1];
        out_pick_j = cl_jidx[cl_size - 1];
        for (int32_t i = 0; i < cl_size; ++i) {
            cumsum += cl_prods[i];
            if (r <= cumsum) { chosen = cl[i]; out_pick_j = cl_jidx[i]; break; }
        }
    } else if (cl_size == 0) {
        // backup list
        for (int32_t j = 0; j < bl; ++j) {
            int32_t v = backup_list[curr * bl + j];
            if (v > 0 && !visited[v]) { chosen = v; break; }
        }
        if (chosen == curr) {
            // global fallback
            float md = std::numeric_limits<float>::max();
            for (int32_t v = 1; v < n; ++v) {
                if (visited[v]) continue;
                float d = dist(curr, v);
                if (d < md) { md = d; chosen = v; }
            }
        }
    }

    return {chosen, is_stochastic, used_uniform};
}

// -------------------- relocate / 2opt / flip: identical to TSP but modulo m --------------------
float MFACO_CVRP::relocate_node(
    int32_t target, int32_t node,
    std::vector<int32_t>& perm,
    std::vector<int32_t>& positions
) {
    if (node == target) return 0.0f;
    int32_t target_succ = get_succ(target, perm, positions);
    if (target_succ == node) return 0.0f;

    int32_t node_pos = positions[node];
    int32_t target_pos = positions[target];

    int32_t node_pred = get_pred(node, perm, positions);
    int32_t node_succ = get_succ(node, perm, positions);

    float cost_delta =
        - dist(node_pred, node) - dist(node, node_succ) - dist(target, target_succ)
        + dist(node_pred, node_succ) + dist(target, node) + dist(node, target_succ);

    if (target_pos < node_pos) {
        int32_t val = perm[node_pos];
        for (int32_t i = node_pos; i > target_pos + 1; --i) perm[i] = perm[i - 1];
        perm[target_pos + 1] = val;
        for (int32_t i = target_pos + 1; i <= node_pos; ++i) positions[perm[i]] = i;
    } else {
        int32_t val = perm[node_pos];
        for (int32_t i = node_pos; i < target_pos; ++i) perm[i] = perm[i + 1];
        perm[target_pos] = val;
        for (int32_t i = node_pos; i <= target_pos; ++i) positions[perm[i]] = i;
    }
    return cost_delta;
}

void MFACO_CVRP::flip_route_section(
    int32_t start_node, int32_t end_node,
    std::vector<int32_t>& perm,
    std::vector<int32_t>& positions
) {
    int32_t first = positions[start_node];
    int32_t last  = positions[end_node];
    if (first > last) std::swap(first, last);

    int32_t seg_len = last - first;
    int32_t rem_len = m - seg_len;

    if (seg_len <= rem_len) {
        int32_t l = first, r = last - 1;
        while (l < r) { std::swap(perm[l], perm[r]); ++l; --r; }
        for (int32_t i = first; i < last; ++i) positions[perm[i]] = i;
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
            ++i; --j;
        }
    }
}

float MFACO_CVRP::two_opt_nn(
    std::vector<int32_t>& perm,
    std::vector<int32_t>& positions,
    std::vector<int32_t>& checklist
) {
    const int32_t max_changes = 1000;
    int32_t changes = 0;
    float total_change = 0.0f;
    size_t cp = 0;

    while (cp < checklist.size() && changes < max_changes) {
        int32_t a = checklist[cp++];
        if (a <= 0 || a >= n) continue;

        int32_t a_next = get_succ(a, perm, positions);
        int32_t a_prev = get_pred(a, perm, positions);

        float dist_a_to_next = dist(a, a_next);
        float dist_a_to_prev = dist(a_prev, a);

        float max_diff = 0.0f;
        int32_t best_move[4] = {-1,-1,-1,-1};

        for (int32_t j = 0; j < k; ++j) {
            int32_t b = nn_list[a * k + j];
            if (b <= 0 || b >= n) continue;
            float dist_ab = dist(a, b);
            if (dist_a_to_next > dist_ab) {
                int32_t b_next = get_succ(b, perm, positions);
                float diff = dist_a_to_next + dist(b, b_next) - dist_ab - dist(a_next, b_next);
                if (diff > max_diff) {
                    best_move[0]=a_next; best_move[1]=b_next; best_move[2]=a; best_move[3]=b;
                    max_diff = diff;
                }
            }
        }

        for (int32_t j = 0; j < k; ++j) {
            int32_t b = nn_list[a * k + j];
            if (b <= 0 || b >= n) continue;
            float dist_ab = dist(a, b);
            if (dist_a_to_prev > dist_ab) {
                int32_t b_prev = get_pred(b, perm, positions);
                float diff = dist_a_to_prev + dist(b_prev, b) - dist_ab - dist(a_prev, b_prev);
                if (diff > max_diff) {
                    best_move[0]=a; best_move[1]=b; best_move[2]=a_prev; best_move[3]=b_prev;
                    max_diff = diff;
                }
            }
        }

        if (max_diff > 0) {
            flip_route_section(best_move[0], best_move[1], perm, positions);
            ++changes;
            total_change -= max_diff;
        }
    }

    return total_change;
}

// -------------------- sampling ants: same MFACO logic, cost via split_dp --------------------
float MFACO_CVRP::sample_ant_fast(
    const float* probmat,
    int32_t start_node,
    std::vector<int32_t>& perm_out,
    std::vector<int32_t>& checklist,
    Xoshiro128Plus& rng
) {
    std::vector<int32_t> perm = source_perm;
    std::vector<int32_t> positions(n, -1);
    for (int32_t i = 0; i < m; ++i) positions[perm[i]] = i;

    std::vector<uint8_t> visited(n, 0);
    visited[0] = 1;
    visited[start_node] = 1;
    int32_t visited_count = 1;

    checklist.clear();
    checklist.push_back(start_node);

    int32_t new_edges = 0;
    int32_t curr = start_node;

    while (new_edges < min_new_edges && visited_count < m) {
        int16_t pick_j = -1;
        uint64_t valid_mask = 0;
        auto [chosen, is_stoch, used_unif] =
            select_next_node(curr, &probmat[curr * k], visited.data(), rng, pick_j, valid_mask);

        if (!contains_edge(curr, chosen)) {
            ++new_edges;
            if (std::find(checklist.begin(), checklist.end(), curr) == checklist.end()) checklist.push_back(curr);
            if (std::find(checklist.begin(), checklist.end(), chosen) == checklist.end()) checklist.push_back(chosen);
            int32_t chosen_pred = get_pred(chosen, perm, positions);
            if (std::find(checklist.begin(), checklist.end(), chosen_pred) == checklist.end()) checklist.push_back(chosen_pred);
        }

        relocate_node(curr, chosen, perm, positions);

        visited[chosen] = 1;
        ++visited_count;
        curr = chosen;
    }

    if (use_local_search && !checklist.empty()) {
        two_opt_nn(perm, positions, checklist);
    }

    perm_out = perm;
    return split_dp(perm).cost;
}

float MFACO_CVRP::sample_ant_traced(
    const float* probmat,
    int32_t start_node,
    std::vector<int32_t>& perm_out,
    std::vector<int32_t>& checklist,
    MFACOTrace& trace,
    Xoshiro128Plus& rng
) {
    trace.clear();
    trace.start_node = start_node;

    std::vector<int32_t> perm = source_perm;
    std::vector<int32_t> positions(n, -1);
    for (int32_t i = 0; i < m; ++i) positions[perm[i]] = i;

    std::vector<uint8_t> visited(n, 0);
    visited[0] = 1;
    visited[start_node] = 1;
    int32_t visited_count = 1;

    checklist.clear();
    checklist.push_back(start_node);

    int32_t new_edges = 0;
    int32_t curr = start_node;

    while (new_edges < min_new_edges && visited_count < m) {
        int16_t pick_j = -1;
        uint64_t valid_mask = 0;
        auto [chosen, is_stoch, used_unif] =
            select_next_node(curr, &probmat[curr * k], visited.data(), rng, pick_j, valid_mask);

        trace.curr_nodes.push_back(curr);
        trace.chosen_nodes.push_back(chosen);
        trace.is_stochastic.push_back(is_stoch ? 1 : 0);
        trace.used_uniform.push_back(used_unif ? 1 : 0);
        trace.pick_j.push_back(pick_j);
        trace.valid_mask.push_back(valid_mask);

        if (!contains_edge(curr, chosen)) {
            ++new_edges;
            if (std::find(checklist.begin(), checklist.end(), curr) == checklist.end()) checklist.push_back(curr);
            if (std::find(checklist.begin(), checklist.end(), chosen) == checklist.end()) checklist.push_back(chosen);
            int32_t chosen_pred = get_pred(chosen, perm, positions);
            if (std::find(checklist.begin(), checklist.end(), chosen_pred) == checklist.end()) checklist.push_back(chosen_pred);
        }

        relocate_node(curr, chosen, perm, positions);

        visited[chosen] = 1;
        ++visited_count;
        curr = chosen;
    }

    if (use_local_search && !checklist.empty()) {
        two_opt_nn(perm, positions, checklist);
    }

    perm_out = perm;
    return split_dp(perm).cost;
}

void MFACO_CVRP::sample(bool require_prob, const float* residual_logits, SampleResult& result, bool parallel_traced) {
    result.clear();
    result.costs.resize(n_ants);
    result.routes.resize(n_ants); // each is perm length m

    std::vector<float> probmat;
    compute_probmat(residual_logits, probmat);

    std::vector<int32_t> start_nodes(n_ants);
    for (int32_t a = 0; a < n_ants; ++a) {
        start_nodes[a] = 1 + (int32_t)rng_.next_uint((uint32_t)m);
    }

    std::vector<uint64_t> ant_seeds;
    auto ensure_ant_seeds = [&]() {
        if (!ant_seeds.empty()) return;
        ant_seeds.resize((size_t)n_ants);
        for (int32_t a = 0; a < n_ants; ++a) {
            uint64_t hi = (uint64_t)rng_.next_u32();
            uint64_t lo = (uint64_t)rng_.next_u32();
            ant_seeds[(size_t)a] = (hi << 32) ^ lo ^ (0x9e3779b97f4a7c15ULL + (uint64_t)a);
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
                MFACOTrace trace;
                trace.reserve(min_new_edges * 2);

                float cost = sample_ant_traced(
                    probmat.data(), start_nodes[a], result.routes[a], checklist, trace, rng_);
                result.costs[a] = cost;

                result.traces.start_nodes.push_back(trace.start_node);
                for (size_t i = 0; i < trace.curr_nodes.size(); ++i) {
                    result.traces.curr_nodes.push_back(trace.curr_nodes[i]);
                    result.traces.chosen_nodes.push_back(trace.chosen_nodes[i]);
                    result.traces.is_stochastic.push_back(trace.is_stochastic[i]);
                    result.traces.used_uniform.push_back(trace.used_uniform[i]);
                    result.traces.pick_j.push_back(trace.pick_j[i]);
                    result.traces.valid_mask.push_back(trace.valid_mask[i]);
                }
                result.traces.starts.push_back((int32_t)result.traces.curr_nodes.size());
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
                    MFACOTrace& trace = traces_per_ant[(size_t)a];
                    trace.reserve(min_new_edges * 2);
                    Xoshiro128Plus rng_local;
                    rng_local.seed(ant_seeds[(size_t)a]);

                    result.costs[a] = sample_ant_traced(
                        probmat.data(), start_nodes[a], result.routes[a], checklist, trace, rng_local);
                }
            }

            result.traces.clear();
            result.traces.starts.resize((size_t)n_ants + 1);
            result.traces.start_nodes.resize((size_t)n_ants);
            result.traces.starts[0] = 0;

            for (int32_t a = 0; a < n_ants; ++a) {
                const auto& t = traces_per_ant[(size_t)a];
                result.traces.start_nodes[(size_t)a] = t.start_node;
                result.traces.starts[(size_t)a + 1] =
                    result.traces.starts[(size_t)a] + (int32_t)t.curr_nodes.size();
            }

            int32_t total = result.traces.starts[(size_t)n_ants];
            result.traces.curr_nodes.resize((size_t)total);
            result.traces.chosen_nodes.resize((size_t)total);
            result.traces.is_stochastic.resize((size_t)total);
            result.traces.used_uniform.resize((size_t)total);
            result.traces.pick_j.resize((size_t)total);
            result.traces.valid_mask.resize((size_t)total);

            for (int32_t a = 0; a < n_ants; ++a) {
                const auto& t = traces_per_ant[(size_t)a];
                int32_t off = result.traces.starts[(size_t)a];
                for (size_t i = 0; i < t.curr_nodes.size(); ++i) {
                    result.traces.curr_nodes[(size_t)off + i] = t.curr_nodes[i];
                    result.traces.chosen_nodes[(size_t)off + i] = t.chosen_nodes[i];
                    result.traces.is_stochastic[(size_t)off + i] = t.is_stochastic[i];
                    result.traces.used_uniform[(size_t)off + i] = t.used_uniform[i];
                    result.traces.pick_j[(size_t)off + i] = t.pick_j[i];
                    result.traces.valid_mask[(size_t)off + i] = t.valid_mask[i];
                }
            }
        }
    } else {
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
                    probmat.data(), start_nodes[a], result.routes[a], checklist, rng_local);
            }
        }
    }
}

// -------------------- update pheromone: deposit on decoded VRP edges --------------------
void MFACO_CVRP::update_pheromone(const int32_t* best_perm_ptr, float new_best_cost) {
    if (new_best_cost < best_cost) {
        best_cost = new_best_cost;
        std::copy(best_perm_ptr, best_perm_ptr + m, best_perm.begin());
    }

    auto [tmin, tmax] = calc_trail_limits_cl(best_cost);
    tau_min = tmin;
    tau_max = tmax;

    float decay_factor = 1.0f - rho;
    for (int32_t i = 0; i < n * k; ++i) {
        pheromone_sparse[i] *= decay_factor;
        pheromone_sparse[i] = std::max(tau_min, std::min(tau_max, pheromone_sparse[i]));
    }

    // Decode segments for deposit
    std::vector<int32_t> perm(best_perm.begin(), best_perm.end());
    auto sp = split_dp(perm);

    float deposit = 1.0f / (new_best_cost + EPS);

    // Deposit depot edges + within-segment edges
    for (auto [i,j] : sp.segs) {
        int32_t first = perm[i];
        int32_t last  = perm[j];

        // depot -> first
        int32_t jd = nn_pos[0 * n + first];
        if (jd >= 0) pheromone_sparse[0 * k + jd] = std::min(pheromone_sparse[0 * k + jd] + deposit, tau_max);

        int32_t jdr = nn_pos[first * n + 0];
        if (jdr >= 0) pheromone_sparse[first * k + jdr] = std::min(pheromone_sparse[first * k + jdr] + deposit, tau_max);

        // internal edges
        for (int32_t t = i; t < j; ++t) {
            int32_t u = perm[t];
            int32_t v = perm[t + 1];

            int32_t ju = nn_pos[u * n + v];
            if (ju >= 0) pheromone_sparse[u * k + ju] = std::min(pheromone_sparse[u * k + ju] + deposit, tau_max);

            int32_t jv = nn_pos[v * n + u];
            if (jv >= 0) pheromone_sparse[v * k + jv] = std::min(pheromone_sparse[v * k + jv] + deposit, tau_max);
        }

        // last -> depot
        int32_t jl = nn_pos[last * n + 0];
        if (jl >= 0) pheromone_sparse[last * k + jl] = std::min(pheromone_sparse[last * k + jl] + deposit, tau_max);

        int32_t jlr = nn_pos[0 * n + last];
        if (jlr >= 0) pheromone_sparse[0 * k + jlr] = std::min(pheromone_sparse[0 * k + jlr] + deposit, tau_max);
    }

    // Update source
    source_perm = best_perm;
    source_cost = new_best_cost;

    std::fill(source_positions.begin(), source_positions.end(), -1);
    for (int32_t i = 0; i < m; ++i) source_positions[source_perm[i]] = i;
}

} // namespace mfaco
