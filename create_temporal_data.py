import pandas as pd
import numpy as np
import torch
from torch_geometric.data import TemporalData
from sklearn.preprocessing import MinMaxScaler

def create_graph_structure(services):
    service_to_id = {name: i for i, name in enumerate(services)}
    id_to_service = {i: name for i, name in enumerate(services)}
    dependencies = [
        ('frontend', 'currencyservice'), ('frontend', 'productcatalogservice'),
        ('frontend', 'cartservice'), ('frontend', 'recommendationservice'),
        ('frontend', 'checkoutservice'), ('frontend', 'adservice'),
        ('checkoutservice', 'productcatalogservice'), ('checkoutservice', 'shippingservice'),
        ('checkoutservice', 'paymentservice'), ('checkoutservice', 'emailservice'),
        ('checkoutservice', 'currencyservice'), ('checkoutservice', 'cartservice'),
        ('cartservice', 'productcatalogservice'), ('recommendationservice', 'productcatalogservice'),
        ('shippingservice', 'currencyservice'),
    ]
    src = [service_to_id[dep[0]] for dep in dependencies if dep[0] in service_to_id and dep[1] in service_to_id]
    dst = [service_to_id[dep[1]] for dep in dependencies if dep[0] in service_to_id and dep[1] in service_to_id]
    edge_index = torch.tensor([src, dst])
    return service_to_id, id_to_service, edge_index

def preprocess_for_tgn(data_df, services, service_to_id, edge_index):
    timestamps = torch.tensor(data_df['unix_timestamp'].values, dtype=torch.float)
    feature_cols = [f'{s}_{m}' for s in services for m in ['cpu_usage', 'memory_usage']]
    label_cols = [f'{s}_label' for s in services]
    scaler = MinMaxScaler()
    features_np = data_df[feature_cols].values
    features_scaled = scaler.fit_transform(features_np)
    num_snapshots = len(data_df)
    num_nodes = len(services)
    num_node_features = len(feature_cols) // num_nodes
    node_features = torch.tensor(features_scaled, dtype=torch.float).view(num_snapshots, num_nodes, num_node_features)
    labels = torch.tensor(data_df[label_cols].values, dtype=torch.long)
    num_events_per_snapshot = edge_index.shape[1]
    src_nodes = edge_index[0].repeat(num_snapshots)
    dst_nodes = edge_index[1].repeat(num_snapshots)
    edge_timestamps = timestamps.repeat_interleave(num_events_per_snapshot)
    src_node_indices_for_edges = src_nodes % num_nodes
    snapshot_indices_for_edges = torch.arange(num_snapshots).repeat_interleave(num_events_per_snapshot)
    msg = node_features[snapshot_indices_for_edges, src_node_indices_for_edges, :]
    data = TemporalData(src=src_nodes, dst=dst_nodes, t=edge_timestamps, msg=msg,
                        t_nodes=timestamps, x=node_features, y=labels)
    return data

if __name__ == "__main__":
    MICROSERVICES = [
        'frontend', 'cartservice', 'productcatalogservice', 'currencyservice',
        'paymentservice', 'shippingservice', 'emailservice', 'checkoutservice',
        'recommendationservice', 'adservice'
    ]
    df = pd.read_csv('processed_temporal_data.csv')
    df.dropna(inplace=True)
    service_to_id, _, edge_index = create_graph_structure(MICROSERVICES)
    temporal_data = preprocess_for_tgn(df, MICROSERVICES, service_to_id, edge_index)
    print("\n--- TemporalData Object ---")
    print(temporal_data)
    torch.save(temporal_data, 'temporal_graph_data.pt')
