from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union
import torch
import torch.nn as nn


@dataclass
class SlotSpec:
    """Blueprint for a single tensor slot in the contiguous memory pool."""
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype


@dataclass
class BlockSpec:
    """Blueprint for an executable block and its state-transition condition."""
    block_id: int
    module: nn.Module
    condition_fn: Callable
    inputs: List[str]
    outputs: List[str]


class Schema:
    """
    Declarative routing schema defining memory slots, scalar variables,
    and execution blocks with attached dynamic routing conditions.
    """

    def __init__(self) -> None:
        self.slots: Dict[str, SlotSpec] = {}
        self.blocks: List[BlockSpec] = []

    def add_slot(self, name: str, shape: Union[Tuple[int, ...], List[int]], dtype: torch.dtype = torch.float32, ) -> None:
        """Register a named tensor memory slot with a fixed shape."""
        if name in self.slots:
            raise ValueError(f"Memory slot '{name}' is already registered.")
        if len(shape) == 0:
            raise ValueError(f"Slot '{name}' must have at least 1 dimension. Use add_var() for scalars.")

        self.slots[name] = SlotSpec(
            name=name,
            shape=tuple(shape),
            dtype=dtype,
        )

    def add_var(self, name: str, dtype: torch.dtype = torch.float32, ) -> None:
        """Register a 1-element tensor slot to act as a scalar flag or metric."""
        self.add_slot(name=name, shape=(1,), dtype=dtype)

    def add_block(self, module: nn.Module, condition_fn: Callable, inputs: Union[str, List[str]], outputs: Union[str, List[str]], ) -> int:
        """
        Register a PyTorch nn.Module along with its transition condition function
        and the names of the input and output memory slots it routes between.
        Returns the integer block_id assigned to this block.
        """
        if not isinstance(module, nn.Module):
            raise TypeError("block module must be an instance of torch.nn.Module.")
        if not callable(condition_fn):
            raise TypeError("condition_fn must be a callable function.")

        input_list = [inputs] if isinstance(inputs, str) else list(inputs)
        output_list = [outputs] if isinstance(outputs, str) else list(outputs)

        for slot_name in input_list:
            if slot_name not in self.slots:
                raise KeyError(f"Input slot '{slot_name}' has not been registered via add_slot() or add_var().")

        for slot_name in output_list:
            if slot_name not in self.slots:
                raise KeyError(f"Output slot '{slot_name}' has not been registered via add_slot() or add_var().")

        block_id = len(self.blocks)
        self.blocks.append(
            BlockSpec(
                block_id=block_id,
                module=module,
                condition_fn=condition_fn,
                inputs=input_list,
                outputs=output_list,
            )
        )
        return block_id