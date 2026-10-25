import torch
import torch.nn as nn
import gridroute


# Any torch module can be used
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, input, var_name):
        var_name = var_name + 1.0
        output = self.proj(input)
        return output, var_name # every variable you want to update needs to be returned, even if you don't need it in the next block.

# Conditions define what blocks run next based on the index in the schema. -1 exits the engine making this the final block.
# The condition always receives the current block index, the step count, and then the current block's outputs.
# You can use different conditions for each block, and they can be as complex as you want. The only requirement is that they return an integer index of the next block to run, exits engine on invalid index.
def condition (current_block, step, output, var_name):
    if step < 10 and output.norm() > 0.2: 
        return current_block        # re-run the block
    else:
        return current_block + 1    # Go to next block (exits if this is the last block, or if it was set to -1)

# The schema is where the user defines the entire execution schedule, and the memory slots that will be used to route data between blocks.
schema = gridroute.Schema()
schema.add_slot("input", shape=(2, 4))
schema.add_slot("output", shape=(2, 4))
schema.add_var("var_name", dtype=torch.float32, default=0.0) # A var is a non-tensor (int, float, bool), but it is stored in a 1-element tensor on the backend.

# The user can dynamically create as many blocks as they want, and the schema will assign them an index in the order they are added.
for i in range(5):        
    # A block can be any torch module, and needs a condition function that defines which block runs next. The inputs and outputs are the names of the slots that will be used to route data between blocks.
    schema.add_block(MyModule(), condition, inputs=["input", "var_name"], outputs=["output", "var_name"])

# The engine is used to execute the schedule and route data between blocks.
engine = gridroute.Engine(schema)

# The final output and var will be the result of the execution schedule. Notice we can set an initial value to var_name even though it has a default value.
final_out, final_var = engine(inputs={"input": torch.ones((2, 4)), "var_name": 0.0}, outputs=["output", "var_name"])