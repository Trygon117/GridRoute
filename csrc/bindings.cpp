#include "router.h"

PYBIND11_MODULE(_C, m) {
    py::class_<gridroute::Router>(m, "Router")
        .def(py::init<>())
        .def("reset_memory", &gridroute::Router::reset_memory);
}