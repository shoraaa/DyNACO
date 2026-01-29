/**
 * pybind11 bindings for Unified MFACO Training Module
 *
 * Exposes MFACO_TSP, MFACO_CVRP, and MFACOTrace to Python.
 * Module name: faco_opt
 */

#include "mfaco_train.h"
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace mfaco;

// ============================================================================
// Helper: create numpy array view from vector (no copy)
// ============================================================================

template <typename T>
py::array_t<T> make_view(T *data, std::vector<py::ssize_t> shape) {
  // Compute strides (row-major)
  std::vector<py::ssize_t> strides(shape.size());
  py::ssize_t stride = sizeof(T);
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return py::array_t<T>(shape, strides, data, py::none());
}

template <typename T> py::array_t<T> make_1d_view(T *data, py::ssize_t len) {
  return make_view<T>(data, {len});
}

template <typename T>
py::array_t<T> make_2d_view(T *data, py::ssize_t rows, py::ssize_t cols) {
  return make_view<T>(data, {rows, cols});
}

// ============================================================================
// MFACOTraceBatch Python wrapper (Shared)
// ============================================================================

class PyMFACOTrace {
public:
  // The trace batch from C++
  MFACOTraceBatch batch;
  int32_t n_ants;

  PyMFACOTrace() : n_ants(0) {}

  // Property accessors returning numpy views
  py::array_t<int32_t> get_starts() {
    return make_1d_view(batch.starts.data(), batch.starts.size());
  }

  py::array_t<int32_t> get_curr_nodes() {
    return make_1d_view(batch.curr_nodes.data(), batch.curr_nodes.size());
  }

  py::array_t<int32_t> get_chosen_nodes() {
    return make_1d_view(batch.chosen_nodes.data(), batch.chosen_nodes.size());
  }

  py::array_t<uint8_t> get_is_stochastic() {
    return make_1d_view(batch.is_stochastic.data(), batch.is_stochastic.size());
  }

  py::array_t<int16_t> get_pick_j() {
    return make_1d_view(batch.pick_j.data(), batch.pick_j.size());
  }

  py::array_t<uint64_t> get_valid_mask() {
    return make_1d_view(batch.valid_mask.data(), batch.valid_mask.size());
  }

  py::array_t<uint8_t> get_is_new_edge() {
    return make_1d_view(batch.is_new_edge.data(), batch.is_new_edge.size());
  }

  py::array_t<int32_t> get_start_nodes() {
    return make_1d_view(batch.start_nodes.data(), batch.start_nodes.size());
  }

  int32_t n_decisions() const {
    return static_cast<int32_t>(batch.curr_nodes.size());
  }

  // Convert batch to list of Python-compatible trace dicts for
  // replay_logp_batch
  py::list to_trace_list() {
    py::list result;
    for (int32_t a = 0; a < n_ants; ++a) {
      py::dict trace;
      trace["start_node"] = batch.start_nodes[a];

      int32_t start = batch.starts[a];
      int32_t end = batch.starts[a + 1];

      py::list curr, chosen, is_stoch;
      for (int32_t i = start; i < end; ++i) {
        curr.append(batch.curr_nodes[i]);
        chosen.append(batch.chosen_nodes[i]);
        is_stoch.append(batch.is_stochastic[i] != 0);
      }

      trace["curr_nodes"] = curr;
      trace["chosen_nodes"] = chosen;
      trace["is_stochastic"] = is_stoch;
      trace["is_new_edge"] =
          make_1d_view(batch.is_new_edge.data() + start, end - start);

      result.append(trace);
    }
    return result;
  }
};

// ============================================================================
// MFACO_TSP Python wrapper
// ============================================================================

class PyMFACO_TSP {
public:
  std::unique_ptr<MFACO_TSP> solver;

  PyMFACO_TSP(
      py::array_t<float, py::array::c_style | py::array::forcecast> coords,
      int32_t n_ants, int32_t cand_list_size = 32,
      int32_t backup_list_size = 32, int32_t min_new_edges = 8,
      float decay = 0.9f, float alpha = 1.0f, float p_best = 0.05f,
      bool use_local_search = true, bool disable_heuristic = false,
      bool extend_ls = false, bool smooth_mmas = false,
      int32_t fixed_steps = 0) {
    auto buf = coords.request();
    if (buf.ndim != 2 || buf.shape[1] != 2) {
      throw std::runtime_error("coords must have shape (n, 2)");
    }
    int32_t n = static_cast<int32_t>(buf.shape[0]);
    const float *coords_ptr = static_cast<const float *>(buf.ptr);

    solver = std::make_unique<MFACO_TSP>(
        coords_ptr, n, n_ants, cand_list_size, backup_list_size, min_new_edges,
        decay, alpha, p_best, use_local_search, disable_heuristic, extend_ls,
        smooth_mmas, fixed_steps);
  }

  // Properties
  int32_t get_n() const { return solver->n; }
  int32_t get_n_ants() const { return solver->n_ants; }
  int32_t get_k() const { return solver->k; }
  int32_t get_bl() const { return solver->bl; }
  int32_t get_min_new_edges() const { return solver->min_new_edges; }
  int32_t get_fixed_steps() const { return solver->fixed_steps; }
  float get_rho() const { return solver->rho; }
  float get_alpha() const { return solver->alpha; }
  float get_p_best() const { return solver->p_best; }
  bool get_use_local_search() const { return solver->use_local_search; }
  bool get_extend_ls() const { return solver->extend_ls; }
  bool get_smooth_mmas() const { return solver->smooth_mmas; }
  float get_source_cost() const { return solver->source_cost; }
  float get_best_cost() const { return solver->best_cost; }
  float get_tau_min() const { return solver->tau_min; }
  float get_tau_max() const { return solver->tau_max; }

  py::array_t<float> get_pheromone_sparse_np() {
    return make_2d_view(solver->pheromone_data(), solver->n, solver->k);
  }
  py::array_t<int32_t> get_nn_list() {
    return make_2d_view(solver->nn_list_data(), solver->n, solver->k);
  }
  py::array_t<int32_t> get_backup_list() {
    return make_2d_view(solver->backup_list_data(), solver->n, solver->bl);
  }
  py::array_t<float> get_heuristic_sparse_np() {
    return make_2d_view(solver->heuristic_data(), solver->n, solver->k);
  }
  py::array_t<int32_t> get_source_route() {
    return make_1d_view(solver->source_route_data(), solver->n);
  }
  py::array_t<int32_t> get_best_route() {
    return make_1d_view(solver->best_route_data(), solver->n);
  }
  py::array_t<int32_t> get_nn_pos() {
    if (solver->nn_pos.empty())
      return make_2d_view<int32_t>(nullptr, 0, 0);
    return make_2d_view(solver->nn_pos_data(), solver->n, solver->n);
  }
  py::array_t<int32_t> get_source_positions() {
    return make_1d_view(solver->source_positions.data(), solver->n);
  }

  void seed_rng(uint64_t seed) { solver->seed_rng(seed); }

  py::tuple
  sample(float invtemp = 1.0f, // unused for now, heuristic is pre-baked
         bool require_prob = false, py::object prior_obj = py::none(),
         bool parallel_traced = false) {
    const float *prior_ptr = nullptr;
    py::array_t<float> prior_arr;

    if (!prior_obj.is_none()) {
      prior_arr = prior_obj.cast<
          py::array_t<float, py::array::c_style | py::array::forcecast>>();
      auto buf = prior_arr.request();
      if (buf.ndim != 2 || buf.shape[0] != solver->n ||
          buf.shape[1] != solver->k) {
        throw std::runtime_error("prior must be shape (n, k)");
      }
      prior_ptr = static_cast<const float *>(buf.ptr);
    }

    SampleResult result;
    {
      py::gil_scoped_release release;
      solver->sample(require_prob, prior_ptr, result, parallel_traced);
    }

    py::array_t<float> costs(solver->n_ants);
    auto costs_buf = costs.mutable_unchecked<1>();
    for (int32_t a = 0; a < solver->n_ants; ++a)
      costs_buf(a) = result.costs[a];

    py::list flats;
    for (int32_t a = 0; a < solver->n_ants; ++a) {
      py::array_t<int32_t> flat(solver->n + 1);
      auto flat_buf = flat.mutable_unchecked<1>();
      for (int32_t i = 0; i < solver->n; ++i)
        flat_buf(i) = result.routes[a][i];
      flat_buf(solver->n) = result.routes[a][0];
      flats.append(flat);
    }

    py::list touched_list;
    for (int32_t a = 0; a < solver->n_ants; ++a)
      touched_list.append(make_1d_view<int32_t>(nullptr, 0));

    py::object traces_obj = py::none();
    if (require_prob) {
      auto traces = std::make_unique<PyMFACOTrace>();
      traces->batch = std::move(result.traces);
      traces->n_ants = solver->n_ants;
      traces_obj = py::cast(std::move(traces));
    }

    py::array_t<float> logps_arr(solver->n_ants);
    if (!result.logps.empty()) {
      auto logps_buf = logps_arr.mutable_unchecked<1>();
      for (int32_t a = 0; a < solver->n_ants; ++a)
        logps_buf(a) = result.logps[a];
    }

    py::object costs_raw_obj = py::none();
    if (!result.costs_raw.empty()) {
      py::array_t<float> costs_raw_arr(solver->n_ants);
      auto buf = costs_raw_arr.mutable_unchecked<1>();
      for (int32_t a = 0; a < solver->n_ants; ++a)
        buf(a) = result.costs_raw[a];
      costs_raw_obj = costs_raw_arr;
    }

    py::object flats_raw_obj = py::none();
    if (!result.routes_raw.empty()) {
      py::list flats_raw;
      for (int32_t a = 0; a < solver->n_ants; ++a) {
        py::array_t<int32_t> flat(solver->n + 1);
        auto flat_buf = flat.mutable_unchecked<1>();
        for (int32_t i = 0; i < solver->n; ++i)
          flat_buf(i) = result.routes_raw[a][i];
        flat_buf(solver->n) = result.routes_raw[a][0];
        flats_raw.append(flat);
      }
      flats_raw_obj = flats_raw;
    }

    py::array_t<int32_t> new_edges_arr(solver->n_ants);
    auto ne_buf = new_edges_arr.mutable_unchecked<1>();
    if (!result.new_edges_count.empty()) {
      for (int32_t a = 0; a < solver->n_ants; ++a)
        ne_buf(a) = result.new_edges_count[a];
    } else {
      for (int32_t a = 0; a < solver->n_ants; ++a)
        ne_buf(a) = 0;
    }

    return py::make_tuple(costs, flats, touched_list, logps_arr, traces_obj,
                          costs_raw_obj, flats_raw_obj, new_edges_arr);
  }

  void update_pheromone_from_flat(
      py::array_t<int32_t, py::array::c_style | py::array::forcecast> best_flat,
      float best_cost) {
    auto buf = best_flat.request();
    if (buf.ndim != 1 || buf.shape[0] < solver->n) {
      throw std::runtime_error("best_flat must be at least length n");
    }
    const int32_t *flat_ptr = static_cast<const int32_t *>(buf.ptr);
    py::gil_scoped_release release;
    solver->update_pheromone(flat_ptr, best_cost);
  }

  void load_snapshot(
      py::array_t<float, py::array::c_style | py::array::forcecast> pheromone,
      py::array_t<int32_t, py::array::c_style | py::array::forcecast>
          source_route,
      float source_cost,
      py::array_t<int32_t, py::array::c_style | py::array::forcecast>
          best_route,
      float best_cost, float tau_min, float tau_max,
      py::array_t<int32_t, py::array::c_style | py::array::forcecast> nn_list,
      py::array_t<int32_t, py::array::c_style | py::array::forcecast>
          backup_list) {
    auto phe_buf = pheromone.request();
    auto src_buf = source_route.request();
    auto best_buf = best_route.request();
    auto nn_buf = nn_list.request();
    auto bl_buf = backup_list.request();

    solver->load_snapshot(
        static_cast<const float *>(phe_buf.ptr),
        static_cast<const int32_t *>(src_buf.ptr), source_cost,
        static_cast<const int32_t *>(best_buf.ptr), best_cost, tau_min, tau_max,
        static_cast<const int32_t *>(nn_buf.ptr),
        static_cast<const int32_t *>(bl_buf.ptr));
  }

  void set_pheromone(
      py::array_t<float, py::array::c_style | py::array::forcecast> pheromone) {
    auto buf = pheromone.request();
    solver->set_pheromone(static_cast<const float *>(buf.ptr));
  }

  void sync_pheromone_to_torch() {
    // No-op
  }
};

// ============================================================================
// MFACO_CVRP Python wrapper
// ============================================================================

class PyMFACO_CVRP {
public:
  std::unique_ptr<MFACO_CVRP> solver;

  PyMFACO_CVRP(py::array_t<float, py::array::c_style | py::array::forcecast>
                   coords, // (n,2)
               py::array_t<float, py::array::c_style | py::array::forcecast>
                   demand, // (n,)
               float capacity, int32_t n_ants, int32_t cand_list_size = 32,
               int32_t backup_list_size = 32, int32_t min_new_edges = 8,
               float decay = 0.9f, float alpha = 1.0f, float p_best = 0.05f,
               bool use_local_search = true, bool disable_heuristic = false,
               bool extend_ls = false, bool smooth_mmas = false,
               int32_t fixed_steps = 0) {
    auto cbuf = coords.request();
    if (cbuf.ndim != 2 || cbuf.shape[1] != 2) {
      throw std::runtime_error("coords must be shape (n,2)");
    }
    int32_t n = (int32_t)cbuf.shape[0];

    auto dbuf = demand.request();
    if (dbuf.ndim != 1 || (int32_t)dbuf.shape[0] != n) {
      throw std::runtime_error("demand must be shape (n,) matching coords");
    }

    solver = std::make_unique<MFACO_CVRP>(
        (const float *)cbuf.ptr, (const float *)dbuf.ptr, n, capacity, n_ants,
        cand_list_size, backup_list_size, min_new_edges, decay, alpha, p_best,
        use_local_search, disable_heuristic, extend_ls, smooth_mmas,
        fixed_steps);
  }

  // properties
  int32_t get_n() const { return solver->n; }
  int32_t get_m() const { return solver->m; }
  int32_t get_n_ants() const { return solver->n_ants; }
  int32_t get_k() const { return solver->k; }
  int32_t get_bl() const { return solver->bl; }
  float get_source_cost() const { return solver->source_cost; }
  float get_best_cost() const { return solver->best_cost; }
  float get_tau_min() const { return solver->tau_min; }
  float get_tau_max() const { return solver->tau_max; }

  py::array_t<float> get_pheromone_sparse_np() {
    return make_2d_view(solver->pheromone_data(), solver->n, solver->k);
  }
  py::array_t<int32_t> get_nn_list() {
    return make_2d_view(solver->nn_list_data(), solver->n, solver->k);
  }
  py::array_t<int32_t> get_backup_list() {
    return make_2d_view(solver->backup_list_data(), solver->n, solver->bl);
  }
  py::array_t<float> get_heuristic_sparse_np() {
    return make_2d_view(solver->heuristic_data(), solver->n, solver->k);
  }
  py::array_t<int32_t> get_nn_pos() {
    if (solver->nn_pos.empty())
      return make_2d_view<int32_t>(nullptr, 0, 0);
    return make_2d_view(solver->nn_pos_data(), solver->n, solver->n);
  }

  py::array_t<int32_t> get_source_perm() {
    return make_1d_view(solver->source_perm_data(), solver->m);
  }
  py::array_t<int32_t> get_best_perm() {
    return make_1d_view(solver->best_perm_data(), solver->m);
  }

  void seed_rng(uint64_t seed) { solver->seed_rng(seed); }

  py::tuple sample(bool require_prob = false, py::object prior = py::none(),
                   bool parallel_traced = false, bool return_decoded = false) {
    const float *prior_ptr = nullptr;
    py::array_t<float> prior_arr;

    if (!prior.is_none()) {
      prior_arr = prior.cast<
          py::array_t<float, py::array::c_style | py::array::forcecast>>();
      auto rbuf = prior_arr.request();
      if (rbuf.ndim != 2 || rbuf.shape[0] != solver->n ||
          rbuf.shape[1] != solver->k) {
        throw std::runtime_error("prior must be shape (n,k)");
      }
      prior_ptr = (const float *)rbuf.ptr;
    }

    SampleResult result;
    {
      py::gil_scoped_release release;
      solver->sample(require_prob, prior_ptr, result, parallel_traced);
    }

    py::array_t<float> costs(solver->n_ants);
    auto cb = costs.mutable_unchecked<1>();
    for (int32_t a = 0; a < solver->n_ants; ++a)
      cb(a) = result.costs[a];

    py::list perms;
    for (int32_t a = 0; a < solver->n_ants; ++a) {
      py::array_t<int32_t> perm(solver->m + 1);
      auto pb = perm.mutable_unchecked<1>();
      for (int32_t i = 0; i < solver->m; ++i)
        pb(i) = result.routes[a][i];
      pb(solver->m) = result.routes[a][0];
      perms.append(perm);
    }

    py::object decoded_obj = py::none();
    if (return_decoded) {
      py::list decoded;
      for (int32_t a = 0; a < solver->n_ants; ++a) {
        std::vector<int32_t> r0;
        solver->decode_perm_to_route0(result.routes[a].data(), r0);
        py::array_t<int32_t> rr((py::ssize_t)r0.size());
        auto rb = rr.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < (py::ssize_t)r0.size(); ++i)
          rb(i) = r0[(size_t)i];
        decoded.append(rr);
      }
      decoded_obj = decoded;
    }

    py::object traces_obj = py::none();
    if (require_prob) {
      auto t = std::make_unique<PyMFACOTrace>();
      t->batch = std::move(result.traces);
      t->n_ants = solver->n_ants;
      traces_obj = py::cast(std::move(t));
    }

    py::array_t<float> logps_arr(solver->n_ants);
    if (!result.logps.empty()) {
      auto logps_buf = logps_arr.mutable_unchecked<1>();
      for (int32_t a = 0; a < solver->n_ants; ++a) {
        logps_buf(a) = result.logps[a];
      }
    }

    py::array_t<int32_t> new_edges_arr(solver->n_ants);
    auto ne_buf = new_edges_arr.mutable_unchecked<1>();
    if (!result.new_edges_count.empty()) {
      for (int32_t a = 0; a < solver->n_ants; ++a) {
        ne_buf(a) = result.new_edges_count[a];
      }
    } else {
      for (int32_t a = 0; a < solver->n_ants; ++a)
        ne_buf(a) = 0;
    }

    return py::make_tuple(costs, perms, decoded_obj, logps_arr, traces_obj,
                          new_edges_arr);
  }

  void update_pheromone_from_perm(
      py::array_t<int32_t, py::array::c_style | py::array::forcecast> best_perm,
      float best_cost) {
    auto buf = best_perm.request();
    if (buf.ndim != 1 || buf.shape[0] < solver->m) {
      throw std::runtime_error("best_perm must be length >= m");
    }
    const int32_t *p = (const int32_t *)buf.ptr;
    py::gil_scoped_release release;
    solver->update_pheromone(p, best_cost);
  }

  void reset_timings() { solver->reset_timings(); }

  py::dict get_timings() {
    py::dict d;
    d["time_ant"] = solver->time_ant;
    d["time_ls"] = solver->time_ls;
    d["time_split"] = solver->time_split;
    return d;
  }
};

// ============================================================================
// Module definition
// ============================================================================

PYBIND11_MODULE(faco_opt, m) {
  m.doc() = "Unified C++ MFACO Training Module for TSP and CVRP";

  // OpenMP controls
  m.def(
      "set_num_threads",
      [](int n_threads) {
        if (n_threads <= 0)
          throw std::runtime_error("n_threads must be > 0");
        omp_set_num_threads(n_threads);
      },
      py::arg("n_threads"), "Set OpenMP thread count");

  m.def("get_max_threads", []() { return omp_get_max_threads(); });
  m.def("get_num_procs", []() { return omp_get_num_procs(); });
  m.def(
      "set_dynamic", [](bool enabled) { omp_set_dynamic(enabled ? 1 : 0); },
      py::arg("enabled"));
  m.def("get_dynamic", []() { return omp_get_dynamic() != 0; });

  // MFACOTrace
  py::class_<PyMFACOTrace>(m, "MFACOTrace")
      .def(py::init<>())
      .def_property_readonly("starts", &PyMFACOTrace::get_starts)
      .def_property_readonly("curr_nodes", &PyMFACOTrace::get_curr_nodes)
      .def_property_readonly("chosen_nodes", &PyMFACOTrace::get_chosen_nodes)
      .def_property_readonly("is_stochastic", &PyMFACOTrace::get_is_stochastic)
      .def_property_readonly("pick_j", &PyMFACOTrace::get_pick_j)
      .def_property_readonly("valid_mask", &PyMFACOTrace::get_valid_mask)
      .def_property_readonly("is_new_edge", &PyMFACOTrace::get_is_new_edge)
      .def_property_readonly("start_nodes", &PyMFACOTrace::get_start_nodes)
      .def_property_readonly("n_decisions", &PyMFACOTrace::n_decisions)
      .def_property_readonly("n_ants",
                             [](const PyMFACOTrace &t) { return t.n_ants; })
      .def("to_trace_list", &PyMFACOTrace::to_trace_list);

  // MFACO_TSP
  py::class_<PyMFACO_TSP>(m, "MFACO_TSP")
      .def(py::init<
               py::array_t<float, py::array::c_style | py::array::forcecast>,
               int32_t, int32_t, int32_t, int32_t, float, float, float, bool,
               bool, bool, bool, int32_t>(),
           py::arg("coords"), py::arg("n_ants"), py::arg("cand_list_size") = 32,
           py::arg("backup_list_size") = 32, py::arg("min_new_edges") = 8,
           py::arg("decay") = 0.9f, py::arg("alpha") = 1.0f,
           py::arg("p_best") = 0.05f, py::arg("use_local_search") = true,
           py::arg("disable_heuristic") = false, py::arg("extend_ls") = false,
           py::arg("smooth_mmas") = false, py::arg("fixed_steps") = 0)
      .def_property_readonly("n", &PyMFACO_TSP::get_n)
      .def_property_readonly("n_ants", &PyMFACO_TSP::get_n_ants)
      .def_property_readonly("k", &PyMFACO_TSP::get_k)
      .def_property_readonly("bl", &PyMFACO_TSP::get_bl)
      .def_property_readonly("min_new_edges", &PyMFACO_TSP::get_min_new_edges)
      .def_property_readonly("fixed_steps", &PyMFACO_TSP::get_fixed_steps)
      .def_property_readonly("rho", &PyMFACO_TSP::get_rho)
      .def_property_readonly("alpha", &PyMFACO_TSP::get_alpha)
      .def_property_readonly("p_best", &PyMFACO_TSP::get_p_best)
      .def_property_readonly("use_local_search",
                             &PyMFACO_TSP::get_use_local_search)
      .def_property_readonly("extend_ls", &PyMFACO_TSP::get_extend_ls)
      .def_property_readonly("smooth_mmas", &PyMFACO_TSP::get_smooth_mmas)
      .def_property_readonly("source_cost", &PyMFACO_TSP::get_source_cost)
      .def_property_readonly("best_cost", &PyMFACO_TSP::get_best_cost)
      .def_property_readonly("tau_min", &PyMFACO_TSP::get_tau_min)
      .def_property_readonly("tau_max", &PyMFACO_TSP::get_tau_max)
      .def_property_readonly("pheromone_sparse_np",
                             &PyMFACO_TSP::get_pheromone_sparse_np)
      .def_property_readonly("nn_list", &PyMFACO_TSP::get_nn_list)
      .def_property_readonly("backup_list", &PyMFACO_TSP::get_backup_list)
      .def_property_readonly("heuristic_sparse_np",
                             &PyMFACO_TSP::get_heuristic_sparse_np)
      .def_property_readonly("source_route", &PyMFACO_TSP::get_source_route)
      .def_property_readonly("best_route", &PyMFACO_TSP::get_best_route)
      .def_property_readonly("nn_pos", &PyMFACO_TSP::get_nn_pos)
      .def_property_readonly("source_positions",
                             &PyMFACO_TSP::get_source_positions)
      .def("seed_rng", &PyMFACO_TSP::seed_rng, py::arg("seed"))
      .def("sample", &PyMFACO_TSP::sample, py::arg("invtemp") = 1.0f,
           py::arg("require_prob") = false, py::arg("prior") = py::none(),
           py::arg("parallel_traced") = false)
      .def("_update_pheromone_from_flat",
           &PyMFACO_TSP::update_pheromone_from_flat)
      .def("load_snapshot", &PyMFACO_TSP::load_snapshot)
      .def("set_pheromone", &PyMFACO_TSP::set_pheromone)
      .def("sync_pheromone_to_torch", &PyMFACO_TSP::sync_pheromone_to_torch);

  // MFACO_CVRP
  py::class_<PyMFACO_CVRP>(m, "MFACO_CVRP")
      .def(py::init<
               py::array_t<float, py::array::c_style | py::array::forcecast>,
               py::array_t<float, py::array::c_style | py::array::forcecast>,
               float, int32_t, int32_t, int32_t, int32_t, float, float, float,
               bool, bool, bool, bool, int32_t>(),
           py::arg("coords"), py::arg("demand"), py::arg("capacity"),
           py::arg("n_ants"), py::arg("cand_list_size") = 32,
           py::arg("backup_list_size") = 32, py::arg("min_new_edges") = 8,
           py::arg("decay") = 0.9f, py::arg("alpha") = 1.0f,
           py::arg("p_best") = 0.05f, py::arg("use_local_search") = true,
           py::arg("disable_heuristic") = false, py::arg("extend_ls") = false,
           py::arg("smooth_mmas") = false, py::arg("fixed_steps") = 0)
      .def_property_readonly("n", &PyMFACO_CVRP::get_n)
      .def_property_readonly("m", &PyMFACO_CVRP::get_m)
      .def_property_readonly("n_ants", &PyMFACO_CVRP::get_n_ants)
      .def_property_readonly("k", &PyMFACO_CVRP::get_k)
      .def_property_readonly("bl", &PyMFACO_CVRP::get_bl)
      .def_property_readonly("source_cost", &PyMFACO_CVRP::get_source_cost)
      .def_property_readonly("best_cost", &PyMFACO_CVRP::get_best_cost)
      .def_property_readonly("tau_min", &PyMFACO_CVRP::get_tau_min)
      .def_property_readonly("tau_max", &PyMFACO_CVRP::get_tau_max)
      .def_property_readonly("pheromone_sparse_np",
                             &PyMFACO_CVRP::get_pheromone_sparse_np)
      .def_property_readonly("nn_list", &PyMFACO_CVRP::get_nn_list)
      .def_property_readonly("backup_list", &PyMFACO_CVRP::get_backup_list)
      .def_property_readonly("heuristic_sparse_np",
                             &PyMFACO_CVRP::get_heuristic_sparse_np)
      .def_property_readonly("nn_pos", &PyMFACO_CVRP::get_nn_pos)
      .def_property_readonly("source_perm", &PyMFACO_CVRP::get_source_perm)
      .def_property_readonly("best_perm", &PyMFACO_CVRP::get_best_perm)
      .def("seed_rng", &PyMFACO_CVRP::seed_rng)
      .def("sample", &PyMFACO_CVRP::sample, py::arg("require_prob") = false,
           py::arg("prior") = py::none(), py::arg("parallel_traced") = false,
           py::arg("return_decoded") = false)
      .def("update_pheromone_from_perm",
           &PyMFACO_CVRP::update_pheromone_from_perm)
      .def("reset_timings", &PyMFACO_CVRP::reset_timings)
      .def("get_timings", &PyMFACO_CVRP::get_timings);
}
