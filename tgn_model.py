import torch
from torch import nn
from torch_geometric.nn.models.tgn import TGNMemory, LastAggregator

class IdentityMessage(nn.Module):
    def __init__(self, raw_msg_dim):
        super().__init__()
        self.out_channels = raw_msg_dim

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        return raw_msg

class TGNNodePredictor(nn.Module):
    def __init__(self, num_nodes, num_node_features, memory_dim, time_dim):
        super().__init__()
        self.num_nodes = num_nodes

        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=num_node_features,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(num_node_features),
            aggregator_module=LastAggregator(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(memory_dim + num_node_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, n_id, node_features_at_t):
        last_memory = self.memory.memory[n_id]
        combined_features = torch.cat([last_memory, node_features_at_t], dim=1)
        return self.classifier(combined_features)

    def update_memory(self, src, dst, t, msg):
        self.memory.update_state(src, dst, t, msg)

    def reset_memory(self):
        self.memory.reset_state()
        device = self.memory.memory.device
        self.memory.last_update = self.memory.last_update.float().to(device)
