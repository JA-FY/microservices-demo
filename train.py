import torch
from sklearn.metrics import classification_report
import numpy as np
from tgn_model import TGNNodePredictor

DEVICE = torch.device('cpu')
BATCH_SIZE = 200
EPOCHS = 15
LR = 0.0001
MEMORY_DIM = 64
TIME_DIM = 64

data = torch.load('temporal_graph_data.pt', weights_only=False)
num_nodes = data.x.size(1)
num_node_features = data.x.size(2)
print(f"Using device: {DEVICE}")
print(f"Graph has {num_nodes} nodes with {num_node_features} features each.")

train_end_idx = int(len(data.src) * 0.7)
val_end_idx = int(len(data.src) * 0.85)

model = TGNNodePredictor(
    num_nodes=num_nodes, num_node_features=num_node_features,
    memory_dim=MEMORY_DIM, time_dim=TIME_DIM
).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
y_all = data.y.flatten()
pos_weight = (y_all == 0).sum() / max((y_all == 1).sum(), 1)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(DEVICE)

def train():
    model.train()
    model.reset_memory()
    total_loss = 0
    for i in range(0, train_end_idx, BATCH_SIZE):
        optimizer.zero_grad()
        src = data.src[i:i+BATCH_SIZE].to(DEVICE)
        dst = data.dst[i:i+BATCH_SIZE].to(DEVICE)
        t = data.t[i:i+BATCH_SIZE].to(DEVICE)
        msg = data.msg[i:i+BATCH_SIZE].to(DEVICE)
        batch_nodes = torch.cat([src, dst]).unique()
        time_idx = torch.searchsorted(data.t_nodes, t[-1].cpu()).item()
        batch_node_features = data.x[time_idx, batch_nodes.cpu()].to(DEVICE)
        batch_labels = data.y[time_idx, batch_nodes.cpu()].float().to(DEVICE)
        pred_logits = model(batch_nodes, batch_node_features)
        loss = criterion(pred_logits.squeeze(), batch_labels)
        loss.backward()
        optimizer.step()
        model.update_memory(src, dst, t, msg)
        total_loss += loss.item()
    return total_loss / (train_end_idx / BATCH_SIZE)

@torch.no_grad()
def evaluate(start_idx, end_idx):
    model.eval()
    all_preds, all_true = [], []
    for i in range(0, end_idx, BATCH_SIZE):
        src = data.src[i:i+BATCH_SIZE].to(DEVICE)
        dst = data.dst[i:i+BATCH_SIZE].to(DEVICE)
        t = data.t[i:i+BATCH_SIZE].to(DEVICE)
        msg = data.msg[i:i+BATCH_SIZE].to(DEVICE)
        if i >= start_idx:
            batch_nodes = torch.cat([src, dst]).unique()
            time_idx = torch.searchsorted(data.t_nodes, t[-1].cpu()).item()
            batch_node_features = data.x[time_idx, batch_nodes.cpu()].to(DEVICE)
            batch_labels = data.y[time_idx, batch_nodes.cpu()]
            pred_logits = model(batch_nodes, batch_node_features)
            preds_binary = (torch.sigmoid(pred_logits).squeeze().cpu() > 0.5).long()
            all_preds.append(preds_binary.numpy())
            all_true.append(batch_labels.numpy())
        model.update_memory(src, dst, t, msg)
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    return classification_report(all_true, all_preds, target_names=['Healthy', 'Faulty'], zero_division=0)

for epoch in range(1, EPOCHS + 1):
    loss = train()
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}")

print("\n--- Validation Set Evaluation ---")
model.reset_memory()
val_report = evaluate(train_end_idx, val_end_idx)
print(val_report)

print("\n--- Test Set Evaluation ---")
test_report = evaluate(val_end_idx, len(data.src))
print(test_report)
num_nodes = data.x.size(1)
num_node_features = data.x.size(2)
print(f"Using device: {DEVICE}")
print(f"Graph has {num_nodes} nodes with {num_node_features} features each.")

train_end_idx = int(len(data.src) * 0.7)
val_end_idx = int(len(data.src) * 0.85)

model = TGNNodePredictor(
    num_nodes=num_nodes, num_node_features=num_node_features,
    memory_dim=MEMORY_DIM, time_dim=TIME_DIM
).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
y_all = data.y.flatten()
pos_weight = (y_all == 0).sum() / max((y_all == 1).sum(), 1)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(DEVICE)

def train():
    model.train()
    model.reset_memory()
    total_loss = 0
    for i in range(0, train_end_idx, BATCH_SIZE):
        optimizer.zero_grad()
        src = data.src[i:i+BATCH_SIZE].to(DEVICE)
        dst = data.dst[i:i+BATCH_SIZE].to(DEVICE)
        t = data.t[i:i+BATCH_SIZE].to(DEVICE)
        msg = data.msg[i:i+BATCH_SIZE].to(DEVICE)
        batch_nodes = torch.cat([src, dst]).unique()
        time_idx = torch.searchsorted(data.t_nodes, t[-1].cpu()).item()
        batch_node_features = data.x[time_idx, batch_nodes.cpu()].to(DEVICE)
        batch_labels = data.y[time_idx, batch_nodes.cpu()].float().to(DEVICE)
        pred_logits = model(batch_nodes, batch_node_features)
        loss = criterion(pred_logits.squeeze(), batch_labels)
        loss.backward()
        optimizer.step()
        model.update_memory(src, dst, t, msg)
        total_loss += loss.item()
    return total_loss / (train_end_idx / BATCH_SIZE)

@torch.no_grad()
def evaluate(start_idx, end_idx):
    model.eval()
    all_preds, all_true = [], []
    for i in range(0, end_idx, BATCH_SIZE):
        src = data.src[i:i+BATCH_SIZE].to(DEVICE)
        dst = data.dst[i:i+BATCH_SIZE].to(DEVICE)
        t = data.t[i:i+BATCH_SIZE].to(DEVICE)
        msg = data.msg[i:i+BATCH_SIZE].to(DEVICE)
        if i >= start_idx:
            batch_nodes = torch.cat([src, dst]).unique()
            time_idx = torch.searchsorted(data.t_nodes, t[-1].cpu()).item()
            batch_node_features = data.x[time_idx, batch_nodes.cpu()].to(DEVICE)
            batch_labels = data.y[time_idx, batch_nodes.cpu()]
            pred_logits = model(batch_nodes, batch_node_features)
            preds_binary = (torch.sigmoid(pred_logits).squeeze().cpu() > 0.5).long()
            all_preds.append(preds_binary.numpy())
            all_true.append(batch_labels.numpy())
        model.update_memory(src, dst, t, msg)
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    return classification_report(all_true, all_preds, target_names=['Healthy', 'Faulty'], zero_division=0)

for epoch in range(1, EPOCHS + 1):
    loss = train()
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}")

print("\n--- Validation Set Evaluation ---")
model.reset_memory()
val_report = evaluate(train_end_idx, val_end_idx)
print(val_report)

print("\n--- Test Set Evaluation ---")
test_report = evaluate(val_end_idx, len(data.src))
print(test_report)
