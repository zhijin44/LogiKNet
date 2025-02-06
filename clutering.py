import torch
import numpy as np
import matplotlib.pyplot as plt

nr_of_clusters = 4
nr_of_points_x_cluster = 50

close_threshold = 0.2
distant_threshold = 1.0

margin = .2
mean = [np.random.uniform([-1 + margin, -1 + margin], [0 - margin, 0 - margin], 2),
        np.random.uniform([0 + margin, -1 + margin], [1 - margin, 0 - margin], 2),
        np.random.uniform([-1 + margin, 0 + margin], [0 - margin, 1 - margin], 2),
        np.random.uniform([0 + margin, 0 + margin], [1 - margin, 1 - margin], 2)]

cov = np.array([[[.01, 0], [0, .01]]] * nr_of_clusters)

cluster_data = {}
for i in range(nr_of_clusters):
    cluster_data[i] = np.random.multivariate_normal(mean=mean[i], cov=cov[i], size=nr_of_points_x_cluster)

data = np.concatenate([cluster_data[i] for i in range(nr_of_clusters)]).astype(np.float32)

for i in range(nr_of_clusters):
    plt.scatter(cluster_data[i][:, 0], cluster_data[i][:, 1])

import ltn

# we define predicate C
class MLP(torch.nn.Module):
    """
    Here, the problem of clustering is organized as a classification task, where the classifier outputs the probability
    that the point given in input belongs to a specific cluster. The clusters are the classes of the classification
    problem.
    """
    def __init__(self, layer_sizes=(2, 16, 16, 16, 4)):
        super(MLP, self).__init__()
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                                                  for i in range(1, len(layer_sizes))])

    def forward(self, x, c):
        """
        Given a point x and a cluster c, the forward phase of this MLP returns the probability that the point x belongs
        to cluster c.

        :param x: point that has to be assigned to a cluster
        :param c: cluster for which we want to compute the probability
        :return: the probability that point x belong to cluster c
        """
        for layer in self.linear_layers[:-1]:
            x = self.elu(layer(x))
        x = self.softmax(self.linear_layers[-1](x))
        out = torch.sum(x * c, dim=1)
        return out

C = ltn.Predicate(MLP())

# we define the variables
c = ltn.Variable("c", torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
x = ltn.Variable("x", torch.tensor(data))
y = ltn.Variable("y", torch.tensor(data))

# we define the constants
th_close = ltn.Constant(torch.tensor(close_threshold))
th_distant = ltn.Constant(torch.tensor(distant_threshold))

# we define connectives, quantifiers, and the SatAgg operator
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=1), quantifier="e")
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=4), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()

# lambda function for Euclidean distance
dist = lambda x, y: torch.unsqueeze(torch.norm(x - y, dim=1), dim=1)

optimizer = torch.optim.Adam(C.parameters(), lr=0.001)

for epoch in range(150):
    if epoch <= 100:
        p_exists = 1
    else:
        p_exists = 6
    optimizer.zero_grad()
    sat_agg = SatAgg(
        Forall(x, Exists(c, C(x, c), p=p_exists)),
        Forall(c, Exists(x, C(x, c), p=p_exists)),
        Forall([c, x, y], Equiv(C(x, c), C(y, c)),
               cond_vars=[x, y],
               cond_fn=lambda x, y: torch.lt(dist(x.value, y.value), th_close.value)),
        Forall([c, x, y], Not(And(C(x, c), C(y, c))),
               cond_vars=[x, y],
               cond_fn=lambda x, y: torch.gt(dist(x.value, y.value), th_distant.value))
    )
    loss = 1. - sat_agg
    loss.backward()
    optimizer.step()

    # we print metrics every 100 epochs of training
    if epoch % 100 == 0:
        print(" epoch %d | loss %.4f | Train Sat %.3f " % (epoch, loss, sat_agg))