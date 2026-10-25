# GridRoute: Dynamic C++ Memory Routing Engine for Recurrent Transformer Architectures

GridRoute is a high-performance C++ routing engine coupled with a declarative Python interface and PyTorch JIT compilation, designed to optimize structured grid-based memory dependencies and dynamic execution schedules for recurrent transformer architectures (such as the FLB-Transformer).

## Build Instructions

Ensure your Conda environment is active with PyTorch, numpy, and ninja installed, then install the package in editable mode with the C++ extension compiled:

```bash
conda activate gridroute
pip install --no-build-isolation -e .
```

## Run Instructions & Testing

To verify the build and run the test suite (comparing the routing engine against a pure Python sequential baseline), execute:

```bash
PYTHONPATH=. pytest -v
```

## Code Layout
* README.md - Project overview, build/run instructions, and code organization guide.
* AI_USAGE.md - Required disclosure log of AI tooling, generated components, and verification methods.
* setup.py - Build configuration script compiling the C++ extension (gridroute._C) via PyTorch cpp_extension.
* requirements.txt - Python package dependencies for the Conda environment.
* gridroute/ - Main Python package directory.
    * __init__.py - Exposes public API classes (Schema, Engine).
    * schema.py - Declarative classes defining memory slots, scalar variables, and execution blocks.
    * engine.py - Handles JIT tracing/scripting of user modules and executes the state-machine routing loop.
* csrc/ - C++ backend source directory.
    * router.h & router.cpp - C++ routing engine managing contiguous memory views and buffer storage.
    * bindings.cpp - Pybind11 bridge exposing C++ classes to Python.
* tests/ - Test suite directory.
    * test_flb_routing.py - Unit tests verifying memory contiguity, equilibrium convergence, recurrence parity, and scalar variable routing.

