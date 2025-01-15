import ltn
import torch

embedding_size = 5

# first group of people
g1 = {person: ltn.Constant(torch.rand((embedding_size,)), trainable=True) for person in 'abcdefgh'}
# second group of people
g2 = {person: ltn.Constant(torch.rand((embedding_size,)), trainable=True) for person in 'ijklmn'}
# group of all people
g = {**g1, **g2}

# we define friendship relations, who has cancer and who is a smoker
friends = [('a', 'b'), ('a', 'e'), ('a', 'f'), ('a', 'g'), ('b', 'c'), ('c', 'd'), ('e', 'f'), ('g', 'h'),
               ('i', 'j'), ('j', 'm'), ('k', 'l'), ('m', 'n')]
smokes = ['a', 'e', 'f', 'g', 'j', 'n']
cancer = ['a', 'e']

# we define predicates F, C, and S
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
        for layer in self.linear_layers[:-1]:
            x = self.elu(layer(x))
        out = self.sigmoid(self.linear_layers[-1](x))
        return out

C = ltn.Predicate(MLP(layer_sizes=(5, 16, 16, 1)))
S = ltn.Predicate(MLP(layer_sizes=(5, 16, 16, 1)))
F = ltn.Predicate(MLP(layer_sizes=(10, 16, 16, 1)))

# we define connectives, quantifiers, and SatAgg
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()


# functions which compute phi1 and phi2
# we need disjunction connective for phi2
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
# this function returns the satisfaction level of the logical formula phi 1
def phi1():
    p = ltn.Variable("p", torch.stack([i.value for i in g.values()]))
    return Forall(p, Implies(C(p), S(p)), p=5).value

# this function returns the satisfaction level of the logical formula phi2
def phi2():
    p = ltn.Variable("p", torch.stack([i.value for i in g.values()]))
    q = ltn.Variable("q", torch.stack([i.value for i in g.values()]))
    return Forall([p, q], Implies(Or(C(p), C(q)), F(p, q)), p=5).value


# we have to optimize the parameters of the three predicates and also of the embeddings
params = list(S.parameters()) + list(F.parameters()) + list(C.parameters()) + [i.value for i in g.values()]
optimizer = torch.optim.Adam(params, lr=0.001)

for epoch in range(1000):
    if epoch <= 200:
        p_exists = 1
    else:
        p_exists = 6
    optimizer.zero_grad()

    # ground the variables
    """
    NOTE: we update the embeddings at each step
        -> we should re-compute the variables.
    """
    x_ = ltn.Variable("x", torch.stack([i.value for i in g.values()]))
    y_ = ltn.Variable("y", torch.stack([i.value for i in g.values()]))

    sat_agg = SatAgg(
        # Friends: knowledge incomplete in that
        #     Friend(x,y) with x<y may be known
        #     but Friend(y,x) may not be known
        SatAgg(*[F(g[x], g[y]) for (x, y) in friends]),
        SatAgg(*[Not(F(g[x], g[y])) for x in g1 for y in g1 if (x, y) not in friends and x < y] +
                [Not(F(g[x], g[y])) for x in g2 for y in g2 if (x, y) not in friends and x < y]),

        # Smokes: knowledge complete
        SatAgg(*[S(g[x]) for x in smokes]),
        SatAgg(*[Not(S(g[x])) for x in g if x not in smokes]),

        # Cancer: knowledge complete in g1 only
        SatAgg(*[C(g[x]) for x in cancer]),
        SatAgg(*[Not(C(g[x])) for x in g1 if x not in cancer]),

        # friendship is anti-reflexive (note that p=5)
        Forall(x_, Not(F(x_, x_)), p=5),

        # friendship is symmetric (note that p=5)
        Forall([x_, y_], Implies(F(x_, y_), F(y_, x_)), p=5),

        # everyone has a friend
        Forall(x_, Exists(y_, F(x_, y_), p=p_exists)),

        # smoking propagates among friends
        Forall([x_, y_], Implies(And(F(x_, y_), S(x_)), S(y_))),

        # smoking causes cancer + not smoking causes not cancer
        Forall(x_, Implies(S(x_), C(x_))),
        Forall(x_, Implies(Not(S(x_)), Not(C(x_))))
    )
    loss = 1. - sat_agg
    loss.backward()
    optimizer.step()

    # we print metrics every 20 epochs of training
    if epoch % 20 == 0:
        print(" epoch %d | loss %.4f | Train Sat %.3f | Phi1 Sat %.3f | Phi2 Sat %.3f" % (epoch, loss,
                    sat_agg, phi1(), phi2()))