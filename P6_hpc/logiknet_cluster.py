# ============================================================================
# logiknet_cluster.py -- LogiK-Net logic-guided clustering for the 3D torus.
#
# Cluster membership C(x, c) is a soft assignment produced by an MLP that
# embeds structural (topology) + congestion knowledge, trained *only* to
# maximise the satisfaction of the four logical axioms of Section V-B:
#
#   A1  forall x exists c  C(x, c)                        (coverage)
#   A2  forall c exists x  C(x, c)                        (non-empty clusters)
#   A3  forall (c, x, y : dist_coord(x,y) <= th_close)
#           C(x,c) <-> C(y,c)                             (close -> same cluster)
#   A4  forall (c, x, y : dist_PTs(x,y)  > th_similarity)
#           not (C(x,c) and C(y,c))                       (dissimilar -> not same)
#
# The design mirrors the project's existing LTNtorch usage
# (P3_knowlegeembed/knowledge_embedding.py and P1_structurelevel/utils.py):
# an MLP produces logits over K clusters, LogitsToPredicate turns them into
# C(x, onehot_c), and SatAgg aggregates the axioms into 1 - loss.
#
# Requires: torch, LTNtorch (`pip install LTNtorch`) -- the same stack the rest
# of the repo uses. numpy-only baselines live in baselines.py.
# ============================================================================
from __future__ import annotations
import numpy as np

from torus import neighbor_pairs


class _MLP:
    pass  # placeholder so the module imports even without torch (see _lazy_torch)


def _lazy_torch():
    import torch
    import ltn
    return torch, ltn


def _build_models(torch, ltn, in_dim, k, hidden, device):
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.elu = torch.nn.ELU()
            sizes = [in_dim] + list(hidden) + [k]
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(sizes[i - 1], sizes[i]) for i in range(1, len(sizes))])

        def forward(self, x):
            for layer in self.layers[:-1]:
                x = self.elu(layer(x))
            return self.layers[-1](x)          # logits over clusters

    class LogitsToPredicate(torch.nn.Module):
        """C(x, l): probability that node x belongs to cluster encoded by one-hot l."""
        def __init__(self, logits_model):
            super().__init__()
            self.logits_model = logits_model
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, x, l):
            probs = self.softmax(self.logits_model(x))
            return torch.sum(probs * l, dim=1)

    logits = MLP().to(device)
    C = ltn.Predicate(LogitsToPredicate(logits)).to(device)
    return logits, C


def cluster(tab, k=16, dims=(24, 24, 24), th_close=2, th_similarity=4.0,
            pts_col="PTs", hidden=(64, 64), lr=1e-3, epochs=300,
            p_forall=2, p_exists=2, seed=0, device=None, verbose=False):
    """Run LogiK-Net logic-guided clustering on one snapshot.

    Returns an integer label per node (argmax of the learned soft assignment;
    clusters that end up empty simply do not appear).
    """
    torch, ltn = _lazy_torch()
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    coords = np.stack([tab["x"].values, tab["y"].values, tab["z"].values], axis=1).astype(float)
    pts = tab[pts_col].values.astype(float)

    # node feature = normalised coords + normalised PTs (structural + congestion)
    feats = np.concatenate([coords / np.array(dims, dtype=float),
                            (pts / 100.0)[:, None]], axis=1).astype(np.float32)
    N, in_dim = feats.shape

    # local pair sets for the guarded axioms (restricted to spatial neighbours)
    pairs = neighbor_pairs(coords.astype(int), dims, th_close)
    if len(pairs):
        dpts = np.abs(pts[pairs[:, 0]] - pts[pairs[:, 1]])
        close_similar = pairs[dpts <= th_similarity]        # A3 pairs
        close_dissim = pairs[dpts > th_similarity]          # A4 pairs
    else:
        close_similar = close_dissim = np.empty((0, 2), dtype=int)

    logits, C = _build_models(torch, ltn, in_dim, k, hidden, device)

    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

    def Iff(a, b):
        # biconditional via existing connectives (portable across LTNtorch versions)
        return And(Implies(a, b), Implies(b, a))

    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=p_forall), quantifier="f")
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=p_exists), quantifier="e")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    x_all = torch.tensor(feats, device=device)
    onehots = torch.eye(k, device=device)
    cluster_var = ltn.Variable("c", onehots)

    def sample(idx, size=4096):
        if len(idx) > size:
            sel = np.random.choice(len(idx), size, replace=False)
            return idx[sel]
        return idx

    opt = torch.optim.Adam(C.parameters(), lr=lr)
    for epoch in range(epochs):
        opt.zero_grad()
        x_var = ltn.Variable("x", x_all)

        # A1 coverage: forall x exists c C(x,c)
        a1 = Forall(x_var, Exists(cluster_var, C(x_var, cluster_var)))
        # A2 non-empty: forall c exists x C(x,c)
        a2 = Forall(cluster_var, Exists(x_var, C(x_var, cluster_var)))

        formulae = [a1, a2]

        # A3 closeness: for sampled close+similar pairs, C(xi,c) <-> C(xj,c).
        # ltn.diag ties xi and xj so they iterate pairwise (the i-th xi with the
        # i-th xj), while the cluster variable is quantified fully.
        if len(close_similar):
            p = sample(close_similar)
            xi = ltn.Variable("xi", x_all[p[:, 0]])
            xj = ltn.Variable("xj", x_all[p[:, 1]])
            ltn.diag(xi, xj)
            formulae.append(
                Forall(cluster_var,
                       Forall(ltn.diag(xi, xj),
                              Iff(C(xi, cluster_var), C(xj, cluster_var)))))
            ltn.undiag(xi, xj)

        # A4 similarity: for sampled close+dissimilar pairs, not(C(xi,c) and C(xj,c))
        if len(close_dissim):
            p = sample(close_dissim)
            xi = ltn.Variable("xi2", x_all[p[:, 0]])
            xj = ltn.Variable("xj2", x_all[p[:, 1]])
            formulae.append(
                Forall(cluster_var,
                       Forall(ltn.diag(xi, xj),
                              Not(And(C(xi, cluster_var), C(xj, cluster_var))))))
            ltn.undiag(xi, xj)

        sat = SatAgg(*formulae)
        loss = 1.0 - sat
        loss.backward()
        opt.step()
        if verbose and epoch % 25 == 0:
            print(f"epoch {epoch:4d} | sat {sat.item():.3f}")

    with torch.no_grad():
        assign = torch.softmax(logits(x_all), dim=1).argmax(dim=1).cpu().numpy() + 1
    return assign.astype(int)
