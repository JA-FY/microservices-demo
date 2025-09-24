import torch
from torch import nn
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from torch_geometric.data import TemporalData
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

DEVICE = torch.device('cpu')
BATCH_SIZE = 200
EPOCHS = 15
LR = 0.0001
MEMORY_DIM = 64
TRAIN_SPLIT_RATIO = 0.03
VAL_SPLIT_RATIO = 0.06

class SimpleTemporalGNN(nn.Module):
    def __init__(self, num_nodes, num_node_features, memory_dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_nodes, memory_dim), requires_grad=False)
        self.gru = nn.GRUCell(input_size=num_node_features, hidden_size=memory_dim)
        self.classifier = nn.Sequential(nn.Linear(memory_dim + num_node_features, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, n_id, node_features_at_t):
        node_memory = self.memory[n_id]
        combined_features = torch.cat([node_memory, node_features_at_t], dim=1)
        return self.classifier(combined_features)
    def update_state(self, src, dst, msg):
        unique_dst, inverse_indices = torch.unique(dst, return_inverse=True)
        if unique_dst.numel() == 0: return
        num_dst = unique_dst.size(0)
        aggregated_msg = torch.zeros(num_dst, msg.size(1), device=DEVICE)
        aggregated_msg.index_add_(0, inverse_indices, msg)
        current_memory = self.memory[unique_dst]
        new_memory = self.gru(aggregated_msg, current_memory)
        self.memory[unique_dst] = new_memory.detach()
    def reset_memory(self):
        self.memory.data.normal_(0, 0.1)

def get_data():
    df = pd.read_csv('processed_temporal_data.csv')
    df.dropna(inplace=True)
    MICROSERVICES = ['frontend', 'cartservice', 'productcatalogservice', 'currencyservice','paymentservice', 'shippingservice', 'emailservice', 'checkoutservice','recommendationservice', 'adservice']
    service_to_id = {name: i for i, name in enumerate(MICROSERVICES)}
    dependencies = [('frontend', 'currencyservice'), ('frontend', 'productcatalogservice'),('frontend', 'cartservice'), ('frontend', 'recommendationservice'),('frontend', 'checkoutservice'), ('frontend', 'adservice'),('checkoutservice', 'productcatalogservice'), ('checkoutservice', 'shippingservice'),('checkoutservice', 'paymentservice'), ('checkoutservice', 'emailservice'),('checkoutservice', 'currencyservice'), ('checkoutservice', 'cartservice'),('cartservice', 'productcatalogservice'), ('recommendationservice', 'productcatalogservice'),('shippingservice', 'currencyservice'),]
    src = [service_to_id[dep[0]] for dep in dependencies if dep[0] in service_to_id and dep[1] in service_to_id]
    dst = [service_to_id[dep[1]] for dep in dependencies if dep[0] in service_to_id and dep[1] in service_to_id]
    edge_index = torch.tensor([src, dst])
    timestamps = torch.tensor(df['unix_timestamp'].values, dtype=torch.float)
    feature_cols = [f'{s}_{m}' for s in MICROSERVICES for m in ['cpu_usage', 'memory_usage', 'p99_latency', 'error_rate']]
    all_possible_features = [f'{s}_{m}' for s in MICROSERVICES for m in ['cpu_usage', 'memory_usage', 'p99_latency', 'error_rate']]
    feature_cols = [col for col in all_possible_features if col in df.columns]
    label_cols = [f'{s}_label' for s in MICROSERVICES]
    scaler = MinMaxScaler()
    features_np = df[feature_cols].values
    features_scaled = scaler.fit_transform(features_np)
    num_snapshots, num_nodes = len(df), len(MICROSERVICES)
    num_node_features = len(feature_cols) // num_nodes
    node_features = torch.tensor(features_scaled, dtype=torch.float).view(num_snapshots, num_nodes, num_node_features)
    labels = torch.tensor(df[label_cols].values, dtype=torch.long)
    num_events_per_snapshot = edge_index.shape[1]
    src_nodes, dst_nodes = edge_index[0].repeat(num_snapshots), edge_index[1].repeat(num_snapshots)
    edge_timestamps = timestamps.repeat_interleave(num_events_per_snapshot)
    src_node_indices_for_edges = src_nodes % num_nodes
    snapshot_indices_for_edges = torch.arange(num_snapshots).repeat_interleave(num_events_per_snapshot)
    msg = node_features[snapshot_indices_for_edges, src_node_indices_for_edges, :]
    return TemporalData(src=src_nodes, dst=dst_nodes, t=edge_timestamps, msg=msg, t_nodes=timestamps, x=node_features, y=labels)

data = get_data()
num_nodes = data.x.size(1)
num_node_features = data.x.size(2)
print(f"Using device: {DEVICE}")
print(f"Graph has {num_nodes} nodes with {num_node_features} features each.")

num_snapshots = data.x.size(0)
train_snap_idx = int(num_snapshots * TRAIN_SPLIT_RATIO)
val_snap_idx = int(num_snapshots * VAL_SPLIT_RATIO)

model = SimpleTemporalGNN(num_nodes=num_nodes, num_node_features=num_node_features, memory_dim=MEMORY_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
y_all = data.y.flatten()
num_positives = max((y_all == 1).sum(), 1)
num_negatives = (y_all == 0).sum()
pos_weight = min(num_negatives / num_positives, 25.0)
print(f"Calculated pos_weight: {num_negatives / num_positives:.2f}, Using capped value: {pos_weight:.2f}")
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight)).to(DEVICE)

def train(max_snap_idx):
    model.train()
    model.reset_memory()
    total_loss, num_batches = 0, 0
    train_event_end_idx = torch.searchsorted(data.t, data.t_nodes[max_snap_idx-1]).item()
    for i in range(0, train_event_end_idx, BATCH_SIZE):
        optimizer.zero_grad()
        src, dst, msg, t = data.src[i:i+BATCH_SIZE].to(DEVICE), data.dst[i:i+BATCH_SIZE].to(DEVICE), data.msg[i:i+BATCH_SIZE].to(DEVICE), data.t[i:i+BATCH_SIZE].to(DEVICE)
        batch_nodes = torch.cat([src, dst]).unique()
        snap_idx_end = torch.searchsorted(data.t_nodes, t[-1].cpu()).item()
        batch_node_features = data.x[snap_idx_end, batch_nodes.cpu()].to(DEVICE)
        batch_labels = data.y[snap_idx_end, batch_nodes.cpu()].float().to(DEVICE)
        pred_logits = model(batch_nodes, batch_node_features)
        loss = criterion(pred_logits.squeeze(), batch_labels)
        loss.backward()
        optimizer.step()
        model.update_state(src, dst, msg)
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0

@torch.no_grad()
def evaluate(start_snap_idx, end_snap_idx):
    model.eval()
    model.reset_memory()
    all_preds, all_true = [], []

    print(f"\n  Building memory state up to snapshot {start_snap_idx}...")

    train_event_end_idx = torch.searchsorted(data.t, data.t_nodes[start_snap_idx-1]).item()
    for i in range(0, train_event_end_idx, BATCH_SIZE):
        src = data.src[i:i+BATCH_SIZE].to(DEVICE)
        dst = data.dst[i:i+BATCH_SIZE].to(DEVICE)
        msg = data.msg[i:i+BATCH_SIZE].to(DEVICE)
        model.update_state(src, dst, msg)

    print(f"  Evaluating from snapshot {start_snap_idx} to {end_snap_idx}...")

    for i in range(start_snap_idx, end_snap_idx):
        all_nodes = torch.arange(num_nodes)
        node_features = data.x[i, all_nodes].to(DEVICE)
        labels = data.y[i, all_nodes]


        pred_logits = model(all_nodes.to(DEVICE), node_features)
        preds_binary = (torch.sigmoid(pred_logits).squeeze().cpu() > 0.5).long()

        all_preds.append(preds_binary.numpy())
        all_true.append(labels.numpy())


        t_start = data.t_nodes[i-1] if i > 0 else 0
        t_end = data.t_nodes[i]
        events_mask = (data.t >= t_start) & (data.t < t_end)
        src = data.src[events_mask].to(DEVICE)
        dst = data.dst[events_mask].to(DEVICE)
        msg = data.msg[events_mask].to(DEVICE)
        if src.numel() > 0:
            model.update_state(src, dst, msg)

    if not all_preds: return "No data in this split."
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    return classification_report(all_true, all_preds, target_names=['Healthy', 'Faulty'], zero_division=0)

for epoch in range(1, EPOCHS + 1):
    loss = train(train_snap_idx)
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}")

print("\n--- Validation Set Evaluation ---")
val_report = evaluate(train_snap_idx, val_snap_idx)
print(val_report)

print("\n--- Test Set Evaluation ---")
test_report = evaluate(val_snap_idx, num_snapshots)
print(test_report)
