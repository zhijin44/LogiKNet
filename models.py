import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_add_pool

class GINModel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, dim_h, layers):
        super(GINModel, self).__init__()

        self.layers = layers

        # 初始化多个卷积层
        self.convs = torch.nn.ModuleList()
        for _ in range(self.layers):
            self.convs.append(
                GINConv(
                    Sequential(Linear(num_node_features if _ == 0 else dim_h, dim_h),
                               BatchNorm1d(dim_h), ReLU(),
                               Linear(dim_h, dim_h), ReLU())
                )
            )

        self.lin1 = Linear(dim_h * layers, dim_h * layers)
        self.lin2 = Linear(dim_h * layers, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node embeddings
        node_embeddings = []
        for i in range(self.layers):
            if i == 0:
                node_embeddings.append(self.convs[i](x, edge_index))
            else:
                node_embeddings.append(self.convs[i](node_embeddings[-1], edge_index))

        # Graph-level readout for each layer's embedding
        graph_embeddings = [global_add_pool(h, batch) for h in node_embeddings]

        # Concatenate graph embeddings
        h = torch.cat(graph_embeddings, dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h


