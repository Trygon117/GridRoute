from typing import Tuple, List, Dict, Callable
import torch
import torch.nn as nn
import pytest
import gridroute


class DummyFLBBlock(nn.Module):
    """Sample user-defined block combining Forward, Lateral, and Backward streams."""
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, fwd: torch.Tensor, lat: torch.Tensor, bwd: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(fwd + 0.5 * lat + 0.5 * bwd))


def test_contiguous_memory_allocation():
    """Verify that multiple slots are carved out of a single contiguous memory buffer."""
    schema = gridroute.Schema()
    schema.add_slot("a", shape=(2, 4))
    schema.add_slot("b", shape=(2, 4))
    schema.add_var("flag", dtype=torch.float32)

    mod = nn.Identity()
    schema.add_block(mod, lambda curr_block, step, out: -1, inputs="a", outputs="b")

    engine = gridroute.Engine(schema, use_cpp=False)
    pool = engine.storage_Buffers[torch.float32]

    # Total elements should be 8 + 8 + 1 = 17
    assert pool.numel() == 17
    assert pool.is_contiguous()

    # Modifying a slot view in-place must directly modify the backing contiguous pool
    engine.slot_views["b"].fill_(3.14)
    assert torch.allclose(pool[8:16], torch.full((8,), 3.14))


def test_dynamic_equilibrium_convergence():
    """Verify that a block can loop dynamically until a tensor metric converges."""
    schema = gridroute.Schema()
    schema.add_slot("state", shape=(1, 4))

    class DecayModule(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 0.5

    # Loop Block 0 until the norm of 'state' drops below 0.2, then halt (-1)
    def until_converged(current_block, step, state):
        if state.norm() > 0.2 and step < 10:
            return current_block
        return -1

    schema.add_block(DecayModule(), until_converged, inputs="state", outputs="state")
    engine = gridroute.Engine(schema, use_cpp=False)

    x_init = torch.ones((1, 4))  # Initial norm is sqrt(4) = 2.0
    out = engine(inputs={"state": x_init}, outputs="state")

    # After 4 halvings: 1.0 -> 0.5 -> 0.25 -> 0.125 -> 0.0625 (norm = 0.125 < 0.2)
    assert out.norm().item() < 0.2
    assert torch.allclose(out, torch.full((1, 4), 0.0625))


def test_flb_recurrent_sweep_parity():
    """
    Verify that a 2-layer FLB recurrent sweep in GridRoute matches a manual
    PyTorch loop bit-for-bit.
    """
    torch.manual_seed(42)
    shape = (2, 8, 16)
    layer0 = DummyFLBBlock(16)
    layer1 = DummyFLBBlock(16)

    schema = gridroute.Schema()
    schema.add_slot("x", shape=shape)
    schema.add_slot("h0", shape=shape)
    schema.add_slot("h1", shape=shape)

    # Run 3 full sweeps: Block 0 -> Block 1 -> Block 0 -> ... -> Halt after 3 sweeps
    def cond_l0(current_block, step, h0):
        return 1  # Always proceed to Layer 1

    def cond_l1(current_block, step, h1):
        if step < 3:
            return 0  # Sweep back to Layer 0
        return -1     # Done after 3 sweeps

    schema.add_block(layer0, cond_l0, inputs=["x", "h0", "h1"], outputs="h0")
    schema.add_block(layer1, cond_l1, inputs=["h0", "h1", "h1"], outputs="h1")

    engine = gridroute.Engine(schema, use_cpp=False)

    x_val = torch.randn(shape)
    gridroute_out = engine(inputs={"x": x_val}, outputs="h1")

    # Reference sequential Python loop
    h0_ref = torch.zeros(shape)
    h1_ref = torch.zeros(shape)
    for _ in range(3):
        h0_ref = layer0(x_val, h0_ref, h1_ref)
        h1_ref = layer1(h0_ref, h1_ref, h1_ref)

    assert torch.allclose(gridroute_out, h1_ref, atol=1e-6)


def test_scalar_variable_routing():
    """Verify that scalar variables passed as slots can be incremented and routed."""
    class IncrementModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)

        def forward(self, x: torch.Tensor, var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.proj(x), var + 1.0

    schema = gridroute.Schema()
    schema.add_slot("input", shape=(1, 2))
    schema.add_slot("output", shape=(1, 2))
    schema.add_var("var_name", dtype=torch.float32)

    def cond(current_block, step, out, var):
        if var.item() < 3.0:
            return current_block
        return -1

    mod = IncrementModule()
    schema.add_block(mod, cond, inputs=["input", "var_name"], outputs=["output", "var_name"])

    engine = gridroute.Engine(schema, use_cpp=False)
    
    x_init = torch.ones((1, 2))
    var_init = torch.zeros((1,))
    
    final_out, final_var = engine(
        inputs={"input": x_init, "var_name": var_init},
        outputs=["output", "var_name"]
    )
    
    assert final_var.item() == 3.0