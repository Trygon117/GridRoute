# AI Usage Disclosure

In compliance with DSCI-650 project guidelines, this document outlines the usage of AI coding assistants during the development of GridRoute.

## Tools Used
* **Gemini (Personal AI Collaborator):** Utilized for system architecture design, structuring AST-based type annotation injection for TorchScript compilation, and designing C++ extension bindings.

## Code Generation Breakdown
* **Mostly AI-Assisted / Generated:** 
  * The AST parsing and unparsing logic in `gridroute/engine.py` that automatically injects TorchScript type hints into user-defined condition functions.
  * The initial template structure of `setup.py` and `csrc/bindings.cpp`.
* **Human-Reviewed & Custom-Built:**
  * Core state-machine execution logic and contiguous buffer slicing views.
  * Test cases in `tests/test_flb_routing.py` validating math parity against sequential Python loops.

## Verification Methods
* **Automated Testing:** All code is verified using `pytest`, ensuring bit-for-bit mathematical parity between GridRoute execution and standard sequential PyTorch reference loops (`torch.allclose`).
* **Line-by-Line Review:** Every generated function was reviewed, tested against edge cases (such as scalar variables and equilibrium convergence checks), and verified to compile under C++20.