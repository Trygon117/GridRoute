import ast
import inspect
import math
import textwrap
import warnings
from typing import Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
from .schema import Schema

# Attempt to import the C++ extension if built
try:
    from . import _C
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False


def _compile_condition_fn(fn: Callable, num_outputs: int) -> torch.jit.ScriptFunction:
    """
    Parses the user's condition function (def or lambda), automatically injects
    TorchScript type annotations (step: int, *outputs: torch.Tensor) -> int,
    and compiles it via torch.jit.CompilationUnit.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    expected_args = 1 + num_outputs

    if len(params) != expected_args:
        raise ValueError(
            f"Condition function '{fn.__name__}' expects {len(params)} arguments, "
            f"but must accept {expected_args}: (step, *outputs)."
        )

    source_lines, _ = inspect.getsourcelines(fn)
    source = textwrap.dedent("".join(source_lines))
    tree = ast.parse(source)

    # Find either a 'def' FunctionDef or an inline 'lambda'
    func_def = None
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
        func_def = tree.body[0]
    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                func_def = ast.FunctionDef(
                    name="_lambda_cond",
                    args=node.args,
                    body=[ast.Return(value=node.body)],
                    decorator_list=[],
                )
                break

    if func_def is None:
        raise RuntimeError(f"Could not parse condition function '{fn.__name__}'.")

    # Inject TorchScript type annotations into the AST: (step: int, out0: torch.Tensor, ...) -> int
    func_def.args.args[0].annotation = ast.Name(id="int", ctx=ast.Load())
    for arg in func_def.args.args[1:]:
        arg.annotation = ast.Name(id="Tensor", ctx=ast.Load())
    func_def.returns = ast.Name(id="int", ctx=ast.Load())

    ast.fix_missing_locations(func_def)
    annotated_src = ast.unparse(func_def) + "\n"

    cu = torch.jit.CompilationUnit(annotated_src)
    return getattr(cu, func_def.name)


class Engine(nn.Module):
    """
    Compiles user-defined PyTorch modules and condition functions into TorchScript,
    allocates a contiguous memory buffer for all slots, and executes the dynamic
    state-machine routing graph.
    """

    def __init__(self, schema: Schema, device: Union[str, torch.device] = "cpu", use_cpp: bool = True, ) -> None:
        super().__init__()
        if len(schema.blocks) == 0:
            raise ValueError("Schema must contain at least one block before initializing Engine.")

        self.schema = schema
        self.device = torch.device(device)
        self.use_cpp = use_cpp and HAS_CPP_BACKEND

        # Hold references to compiled TorchScript modules and condition functions
        self.compiled_modules: List[torch.jit.ScriptModule] = []
        self.compiled_conditions: List[torch.jit.ScriptFunction] = []

        # Contiguous backing buffers grouped by dtype, and named tensor views into them
        self.storage_Buffers: Dict[torch.dtype, torch.Tensor] = {}
        self.slot_views: Dict[str, torch.Tensor] = {}

        self._allocate_contiguous_pool()
        self._compile_blocks_and_conditions()

        # Initialize C++ router if available
        self._cpp_router = None
        if self.use_cpp:
            self._init_cpp_backend()

    def _allocate_contiguous_pool(self) -> None:
        """
        Allocates one contiguous 1D tensor per dtype and slices out shaped tensor
        views for every slot registered in the schema.
        """
        dtype_groups: Dict[torch.dtype, List[str]] = {}
        for name, spec in self.schema.slots.items():
            dtype_groups.setdefault(spec.dtype, []).append(name)

        for dtype, slot_names in dtype_groups.items():
            total_elements = sum(math.prod(self.schema.slots[name].shape) for name in slot_names)
            # Allocate a single contiguous 1D block on the target device
            contiguous_buffer = torch.zeros(total_elements, dtype=dtype, device=self.device)
            self.storage_Buffers[dtype] = contiguous_buffer

            # Carve out non-copying views for each slot
            offset = 0
            for name in slot_names:
                shape = self.schema.slots[name].shape
                numel = math.prod(shape)
                self.slot_views[name] = contiguous_buffer[offset : offset + numel].view(shape)
                offset += numel

    def _compile_blocks_and_conditions(self) -> None:
        """
        Dynamically compiles each block's nn.Module (via trace) and condition_fn
        (via script) into TorchScript so they can run without Python overhead.
        """
        module_cache: Dict[int, torch.jit.ScriptModule] = {}

        for block in self.schema.blocks:
            block.module.to(self.device)
            mod_id = id(block.module)

            # Reuse compiled TorchScript module if the same nn.Module instance is shared
            if mod_id not in module_cache:
                dummy_inputs = tuple(
                    torch.zeros(
                        self.schema.slots[slot_name].shape,
                        dtype=self.schema.slots[slot_name].dtype,
                        device=self.device,
                    )
                    for slot_name in block.inputs
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    try:
                        compiled_mod = torch.jit.trace(block.module, dummy_inputs)
                    except Exception:
                        compiled_mod = torch.jit.script(block.module)
                module_cache[mod_id] = compiled_mod

            self.compiled_modules.append(module_cache[mod_id])

            # Compile the condition function with auto-injected AST type annotations
            compiled_cond = _compile_condition_fn(block.condition_fn, len(block.outputs))
            self.compiled_conditions.append(compiled_cond)

    def _init_cpp_backend(self) -> None:
        """Initializes the C++ routing engine with the compiled TorchScript objects."""
        # Stubbed until csrc/router.cpp is compiled
        if HAS_CPP_BACKEND and hasattr(_C, "Router"):
            self._cpp_router = _C.Router()

    def reset_memory(self) -> None:
        """Zeros out all contiguous memory buffers in a single operation per dtype."""
        for buf in self.storage_Buffers.values():
            buf.zero_()

    def forward(self,inputs: Dict[str, torch.Tensor], outputs: Union[str, List[str]], start_block: int = 0, max_steps: int = 1000, ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Executes the dynamic routing graph starting from start_block until a
        block's condition function returns -1.
        """
        self.reset_memory()

        # Copy initial inputs directly into their contiguous slot views
        for slot_name, tensor in inputs.items():
            if slot_name not in self.slot_views:
                raise KeyError(f"Unknown input slot '{slot_name}'.")
            if tensor.shape != self.slot_views[slot_name].shape:
                raise ValueError(
                    f"Input tensor for slot '{slot_name}' has shape {tuple(tensor.shape)}, "
                    f"expected {self.slot_views[slot_name].shape}."
                )
            self.slot_views[slot_name].copy_(tensor)

        # Execute state machine
        current_block = start_block
        block_exec_counts = [0] * len(self.schema.blocks)
        total_steps = 0

        while current_block != -1:
            if current_block < 0 or current_block >= len(self.schema.blocks):
                raise RuntimeError(
                    f"Condition returned invalid block ID {current_block}. "
                    f"Valid IDs are 0..{len(self.schema.blocks) - 1} or -1 to halt."
                )
            if total_steps >= max_steps:
                raise RuntimeError(
                    f"Routing exceeded max_steps ({max_steps}) without halting (-1)."
                )

            spec = self.schema.blocks[current_block]
            mod = self.compiled_modules[current_block]
            cond = self.compiled_conditions[current_block]

            # 1. Gather input slot views
            in_tensors = [self.slot_views[name] for name in spec.inputs]

            # 2. Execute JIT-compiled block
            raw_out = mod(*in_tensors)

            # 3. Write outputs back into contiguous memory slots in-place
            out_tuples = raw_out if isinstance(raw_out, (tuple, list)) else (raw_out,)
            if len(out_tuples) != len(spec.outputs):
                raise RuntimeError(
                    f"Block {current_block} returned {len(out_tuples)} tensors, "
                    f"but schema specifies {len(spec.outputs)} output slots."
                )

            for slot_name, val in zip(spec.outputs, out_tuples):
                self.slot_views[slot_name].copy_(val)

            # 4. Update step counter for this block and evaluate JIT condition
            block_exec_counts[current_block] += 1
            total_steps += 1

            out_views = [self.slot_views[name] for name in spec.outputs]
            next_block = cond(block_exec_counts[current_block], *out_views)
            current_block = int(next_block)

        # Return requested output slot(s)
        if isinstance(outputs, str):
            return self.slot_views[outputs].clone()
        return [self.slot_views[name].clone() for name in outputs]