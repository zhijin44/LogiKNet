import ltn
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Nodes information
raw_nodes = {
    "Hannover": (9.80, 52.39),
    "Frankfurt": (8.66, 50.14),
    "Hamburg": (10.08, 53.55),
    "Norden": (7.21, 53.60),
    "Bremen": (8.80, 53.08),
    "Berlin": (13.48, 52.52),
    "Muenchen": (11.55, 48.15),
    "Ulm": (9.99, 48.40),
    "Nuernberg": (11.08, 49.45),
    "Stuttgart": (9.12, 48.73),
    "Karlsruhe": (8.41, 49.01),
    "Mannheim": (8.49, 49.49),
    "Essen": (7.00, 51.44),
    "Dortmund": (7.48, 51.51),
    "Duesseldorf": (6.78, 51.22),
    "Koeln": (7.01, 50.92),
    "Leipzig": (12.38, 51.34)
}
# Convert raw_nodes to LTN constants
embedding_size = 2
nodes = {k: ltn.Constant(torch.rand((embedding_size,)), trainable=True) for k, v in raw_nodes.items()}
# nodes = {k: ltn.Constant(torch.tensor(v, device=device)) for k, v in raw_nodes.items()}


# Links information
raw_links = {
    "L1": ("Berlin", "Hamburg"),
    "L2": ("Berlin", "Hannover"),
    "L3": ("Berlin", "Leipzig"),
    "L4": ("Bremen", "Hamburg"),
    "L5": ("Bremen", "Hannover"),
    "L6": ("Bremen", "Norden"),
    "L7": ("Dortmund", "Essen"),
    "L8": ("Dortmund", "Hannover"),
    "L9": ("Dortmund", "Koeln"),
    "L10": ("Dortmund", "Norden"),
    "L11": ("Duesseldorf", "Essen"),
    "L12": ("Duesseldorf", "Koeln"),
    "L13": ("Frankfurt", "Hannover"),
    "L14": ("Frankfurt", "Koeln"),
    "L15": ("Frankfurt", "Leipzig"),
    "L16": ("Frankfurt", "Mannheim"),
    "L17": ("Frankfurt", "Nuernberg"),
    "L18": ("Hamburg", "Hannover"),
    "L19": ("Hannover", "Leipzig"),
    "L20": ("Karlsruhe", "Mannheim"),
    "L21": ("Karlsruhe", "Stuttgart"),
    "L22": ("Leipzig", "Nuernberg"),
    "L23": ("Muenchen", "Nuernberg"),
    "L24": ("Muenchen", "Ulm"),
    "L25": ("Nuernberg", "Stuttgart"),
    "L26": ("Stuttgart", "Ulm")
}
# define the link relation
links = list(raw_links.values())




class MLP(torch.nn.Module):
    """
    Simple MLP model used for defining the predicates of our problem.
    """
    def __init__(self, layer_sizes=(10, 16, 16, 1)):
        super(MLP, self).__init__()
        self.elu = torch.nn.ELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                                                  for i in range(1, len(layer_sizes))])
        self.to(device)

    def forward(self, *x):
        """
        Given an individual x, the forward phase of this MLP returns the probability that the individual x is a smoker,
        or has cancer, or is friend of y (if given and predicate is F).

        :param x: individuals for which we have to compute the probability
        :return: the probability that individual x is a smoker, or has cancer, or is friend of y (if given)
        """
        x = list(x)
        if len(x) == 1:
            x = x[0]
        else:
            x = torch.cat(x, dim=1)
        x = x.to(device)
        for layer in self.linear_layers[:-1]:
            x = self.elu(layer(x))
        out = self.sigmoid(self.linear_layers[-1](x))
        return out

L = ltn.Predicate(MLP(layer_sizes=(4, 16, 16, 1)))

# define connectives, quantifiers, and SatAgg
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()


# define logical formula phi


criterion = torch.nn.CrossEntropyLoss()
params = list(L.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)

for epoch in range(200):
    if epoch <= 200:
        p_exists = 2
    else:
        p_exists = 6
    optimizer.zero_grad()

    x_ = ltn.Variable("x", torch.stack([i.value for i in nodes.values()]))
    y_ = ltn.Variable("y", torch.stack([i.value for i in nodes.values()]))
    
    sat_agg = SatAgg(
        *[L(nodes[source], nodes[target]) for (source, target) in links],

        # Link is anti-reflexive
        Forall([x_], Not(L(x_, x_)), p = 5),
        # Link is symmetric
        Forall([x_, y_], Implies(L(x_, y_), L(y_, x_)), p = 5)
    )


    loss = 1. - sat_agg
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(" epoch %d | loss %.4f | Train Sat %.3f" % (epoch, loss, sat_agg))

source = "Berlin"
target = "Stuttgart"
print(L(nodes[source], nodes[target]).value)
# print the output of L when inputs links are not in the links
not_in_links = [("Berlin", "Stuttgart"), ("Stuttgart", "Berlin"), ("Berlin", "Berlin"), ("Stuttgart", "Stuttgart")]
invalid_link_sat = SatAgg(*[L(nodes[source], nodes[target]) for (source, target) in links])





# for (x, y) in links:
# print(L(one_node, another_node))
# print(L(nodes["Berlin"], nodes["Hamburg"]))
# print(L(nodes["Berlin"], nodes["Stuttgart"]))
# print(L(nodes["Berlin"], nodes["Hannover"]))
# print(SatAgg(L(nodes["Berlin"], nodes["Hamburg"]), L(nodes["Berlin"], nodes["Hannover"]), L(nodes["Berlin"], nodes["Stuttgart"])))
# for (source, target) in links:
#     print(L(nodes[source], nodes[target]).value)
#     print(SatAgg(L(nodes[source], nodes[target])).data)
