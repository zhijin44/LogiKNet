import LTNtorch.LTN_IoMT.custom_fuzzy_ops as cfo
import torch
import ltn

# Instantiate the AndProd operator
And = ltn.Connective(cfo.AndProd())
# And = cfo.AndProd()

# Example predicates and variables
p = ltn.Predicate(func=lambda x: torch.nn.Sigmoid()(torch.sum(x, dim=1)))
x = ltn.Variable('x', torch.tensor([[0.56], [0.9]]))
y = ltn.Variable('y', torch.tensor([[0.7], [0.2]]))
z = ltn.Variable('z', torch.tensor([[0.8], [0.5]]))

# Apply the operator to a list of operands
result = And((p(x), p(y)))  # Change from list to tuple
# result = And([p(x), p(y), p(z)])
print(result.value)
