import torch
import torch.nn as nn
import gridroute

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 4, bias=False)

    def forward(self, input, var_name):
        var_name += 1.0  # Increment the scalar variable in-place (I dont think this works right now, so we need to pass it back as an output to the next block)
        return self.proj(input), var_name

schema = gridroute.Schema()
schema.add_slot("input", shape=(2, 4))
schema.add_slot("output", shape=(2, 4))
schema.add_var("var_name", dtype=torch.float32)

def condition1(current_block, step, output, var_name): # Add ID of previous block so we know where we want to go next
    if step < 10 and output.norm() > 0.2: 
        return current_block        # re-run the block
    else:
        return current_block + 1    # Go to next block

def condition2(current_block, step, output, var_name):
    if step < 10 and output.norm() > 0.2: 
        return current_block    # re-run the block
    else:
        return -1               # Halt the execution

layer1 = MyModule()
layer2 = MyModule()

schema.add_block(layer1, condition1, inputs=["input", "var_name"], outputs=["output", "var_name"])
schema.add_block(layer2, condition2, inputs=["input", "var_name"], outputs=["output", "var_name"])