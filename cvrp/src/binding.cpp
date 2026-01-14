/**
 * pybind11 bindings for MFACO_CVRP
 * Module name: faco_cvrp
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>

#include "mfaco_train.h"   // must define MFACO_CVRP + SampleResult + Trace types

namespace py = pybind11;
using namespace mfaco;

// ---------- numpy view helpers (no-copy) ----------
template<typename T>
py::array_t<T> make_view(T* data, std::vector<py::ssize_t> shape) {
    std::vector<py::ssize_t> strides(shape.size());
    py::ssize_t stride = sizeof(T);
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return py::array_t<T>(shape, strides, data, py::none());
}
template<typename T>
py::array_t<T> make_1d_view(T* data, py::ssize_t len) {
    return make_view<T>(data, {len});
}
template<typename T>
py::array_t<T> make_2d_view(T* data, py::ssize_t rows, py::ssize_t cols) {
    return make_view<T>(data, {rows, cols});
}

// ---------- (optional) Trace wrapper ----------
class PyMFACOTrace {
public:
    MFACOTraceBatch batch;
    int32_t n_ants = 0;

    py::array_t<int32_t> starts()      { return make_1d_view(batch.starts.data(), batch.starts.size()); }
    py::array_t<int32_t> curr_nodes()  { return make_1d_view(batch.curr_nodes.data(), batch.curr_nodes.size()); }
    py::array_t<int32_t> chosen_nodes(){ return make_1d_view(batch.chosen_nodes.data(), batch.chosen_nodes.size()); }
    py::array_t<uint8_t> is_stochastic(){ return make_1d_view(batch.is_stochastic.data(), batch.is_stochastic.size()); }
    py::array_t<uint8_t> used_uniform(){ return make_1d_view(batch.used_uniform.data(), batch.used_uniform.size()); }
    py::array_t<int16_t> pick_j()      { return make_1d_view(batch.pick_j.data(), batch.pick_j.size()); }
    py::array_t<uint64_t> valid_mask() { return make_1d_view(batch.valid_mask.data(), batch.valid_mask.size()); }
    py::array_t<int32_t> start_nodes() { return make_1d_view(batch.start_nodes.data(), batch.start_nodes.size()); }

    int32_t n_decisions() const { return (int32_t)batch.curr_nodes.size(); }
};

// ---------- MFACO_CVRP wrapper ----------
class PyMFACO_CVRP {
public:
    std::unique_ptr<MFACO_CVRP> solver;

    PyMFACO_CVRP(
        py::array_t<float, py::array::c_style | py::array::forcecast> coords,   // (n,2)
        py::array_t<float, py::array::c_style | py::array::forcecast> demand,   // (n,)
        float capacity,
        int32_t n_ants,
        int32_t cand_list_size = 32,
        int32_t backup_list_size = 32,
        int32_t min_new_edges = 8,
        float decay = 0.9f,
        float alpha = 1.0f,
        float p_best = 0.05f,
        bool use_local_search = true,
        bool disable_heuristic = false
    ) {
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
            (const float*)cbuf.ptr,
            (const float*)dbuf.ptr,
            n,
            capacity,
            n_ants,
            cand_list_size,
            backup_list_size,
            min_new_edges,
            decay,
            alpha,
            p_best,
            use_local_search,
            disable_heuristic
        );
    }

    // properties
    int32_t n() const { return solver->n; }        // includes depot
    int32_t m() const { return solver->m; }        // customers only
    int32_t n_ants() const { return solver->n_ants; }
    int32_t k() const { return solver->k; }
    int32_t bl() const { return solver->bl; }
    float source_cost() const { return solver->source_cost; }
    float best_cost() const { return solver->best_cost; }
    float tau_min() const { return solver->tau_min; }
    float tau_max() const { return solver->tau_max; }

    py::array_t<float> pheromone_sparse_np() { return make_2d_view(solver->pheromone_data(), solver->n, solver->k); }
    py::array_t<int32_t> nn_list()          { return make_2d_view(solver->nn_list_data(), solver->n, solver->k); }
    py::array_t<int32_t> backup_list()      { return make_2d_view(solver->backup_list_data(), solver->n, solver->bl); }
    py::array_t<float> heuristic_sparse_np(){ return make_2d_view(solver->heuristic_data(), solver->n, solver->k); }
    py::array_t<int32_t> nn_pos()           { return make_2d_view(solver->nn_pos_data(), solver->n, solver->n); }

    py::array_t<int32_t> source_perm() { return make_1d_view(solver->source_perm_data(), solver->m); }
    py::array_t<int32_t> best_perm()   { return make_1d_view(solver->best_perm_data(), solver->m); }

    void seed_rng(uint64_t seed) { solver->seed_rng(seed); }

    py::tuple sample(
        bool require_prob = false,
        py::object residual_logits = py::none(),
        bool parallel_traced = false,
        bool return_decoded = false
    ) {
        const float* residual_ptr = nullptr;
        py::array_t<float> residual_arr;

        if (!residual_logits.is_none()) {
            residual_arr = residual_logits.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            auto rbuf = residual_arr.request();
            if (rbuf.ndim != 2 || rbuf.shape[0] != solver->n || rbuf.shape[1] != solver->k) {
                throw std::runtime_error("residual_logits must be shape (n,k)");
            }
            residual_ptr = (const float*)rbuf.ptr;
        }

        SampleResult result;
        {
            py::gil_scoped_release release;
            solver->sample(require_prob, residual_ptr, result, parallel_traced);
        }

        // costs
        py::array_t<float> costs(solver->n_ants);
        auto cb = costs.mutable_unchecked<1>();
        for (int32_t a = 0; a < solver->n_ants; ++a) cb(a) = result.costs[a];

        // perms (customers-only cycle, length m+1 with last == first)
        py::list perms;
        for (int32_t a = 0; a < solver->n_ants; ++a) {
            py::array_t<int32_t> perm(solver->m + 1);
            auto pb = perm.mutable_unchecked<1>();
            for (int32_t i = 0; i < solver->m; ++i) pb(i) = result.routes[a][i];
            pb(solver->m) = result.routes[a][0];
            perms.append(perm);
        }

        // decoded routes (optional) : [0,...,0,...,0]
        py::object decoded_obj = py::none();
        if (return_decoded) {
            py::list decoded;
            for (int32_t a = 0; a < solver->n_ants; ++a) {
                std::vector<int32_t> r0;
                solver->decode_perm_to_route0(result.routes[a].data(), r0);
                py::array_t<int32_t> rr((py::ssize_t)r0.size());
                auto rb = rr.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < (py::ssize_t)r0.size(); ++i) rb(i) = r0[(size_t)i];
                decoded.append(rr);
            }
            decoded_obj = decoded;
        }

        // traces
        py::object traces_obj = py::none();
        if (require_prob) {
            auto t = std::make_unique<PyMFACOTrace>();
            t->batch = std::move(result.traces);
            t->n_ants = solver->n_ants;
            traces_obj = py::cast(std::move(t));
        }

        return py::make_tuple(costs, perms, decoded_obj, traces_obj);
    }

    void update_pheromone_from_perm(
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> best_perm,
        float best_cost
    ) {
        auto buf = best_perm.request();
        if (buf.ndim != 1 || buf.shape[0] < solver->m) {
            throw std::runtime_error("best_perm must be length >= m");
        }
        const int32_t* p = (const int32_t*)buf.ptr;
        py::gil_scoped_release release;
        solver->update_pheromone(p, best_cost);
    }
};

PYBIND11_MODULE(faco_cvrp, m) {
    m.doc() = "C++ MFACO CVRP module (giant tour + split)";

    // OpenMP controls
    m.def("set_num_threads", [](int n){ omp_set_num_threads(n); });
    m.def("get_max_threads", [](){ return omp_get_max_threads(); });

    py::class_<PyMFACOTrace>(m, "MFACOTrace")
        .def(py::init<>())
        .def_property_readonly("starts", &PyMFACOTrace::starts)
        .def_property_readonly("curr_nodes", &PyMFACOTrace::curr_nodes)
        .def_property_readonly("chosen_nodes", &PyMFACOTrace::chosen_nodes)
        .def_property_readonly("is_stochastic", &PyMFACOTrace::is_stochastic)
        .def_property_readonly("used_uniform", &PyMFACOTrace::used_uniform)
        .def_property_readonly("pick_j", &PyMFACOTrace::pick_j)
        .def_property_readonly("valid_mask", &PyMFACOTrace::valid_mask)
        .def_property_readonly("start_nodes", &PyMFACOTrace::start_nodes)
        .def_property_readonly("n_decisions", &PyMFACOTrace::n_decisions)
        .def_property_readonly("n_ants", [](const PyMFACOTrace& t){ return t.n_ants; });

    py::class_<PyMFACO_CVRP>(m, "MFACO_CVRP")
        .def(py::init<
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            float, int32_t,
            int32_t, int32_t, int32_t,
            float, float, float,
            bool, bool
        >(),
            py::arg("coords"),
            py::arg("demand"),
            py::arg("capacity"),
            py::arg("n_ants"),
            py::arg("cand_list_size") = 32,
            py::arg("backup_list_size") = 32,
            py::arg("min_new_edges") = 8,
            py::arg("decay") = 0.9f,
            py::arg("alpha") = 1.0f,
            py::arg("p_best") = 0.05f,
            py::arg("use_local_search") = true,
            py::arg("disable_heuristic") = false
        )
        .def_property_readonly("n", &PyMFACO_CVRP::n)
        .def_property_readonly("m", &PyMFACO_CVRP::m)
        .def_property_readonly("n_ants", &PyMFACO_CVRP::n_ants)
        .def_property_readonly("k", &PyMFACO_CVRP::k)
        .def_property_readonly("bl", &PyMFACO_CVRP::bl)
        .def_property_readonly("source_cost", &PyMFACO_CVRP::source_cost)
        .def_property_readonly("best_cost", &PyMFACO_CVRP::best_cost)
        .def_property_readonly("tau_min", &PyMFACO_CVRP::tau_min)
        .def_property_readonly("tau_max", &PyMFACO_CVRP::tau_max)
        .def_property_readonly("pheromone_sparse_np", &PyMFACO_CVRP::pheromone_sparse_np)
        .def_property_readonly("nn_list", &PyMFACO_CVRP::nn_list)
        .def_property_readonly("backup_list", &PyMFACO_CVRP::backup_list)
        .def_property_readonly("heuristic_sparse_np", &PyMFACO_CVRP::heuristic_sparse_np)
        .def_property_readonly("nn_pos", &PyMFACO_CVRP::nn_pos)
        .def_property_readonly("source_perm", &PyMFACO_CVRP::source_perm)
        .def_property_readonly("best_perm", &PyMFACO_CVRP::best_perm)
        .def("seed_rng", &PyMFACO_CVRP::seed_rng)
        .def("sample", &PyMFACO_CVRP::sample,
            py::arg("require_prob") = false,
            py::arg("residual_logits") = py::none(),
            py::arg("parallel_traced") = false,
            py::arg("return_decoded") = false
        )
        .def("_update_pheromone_from_perm", &PyMFACO_CVRP::update_pheromone_from_perm);
}
