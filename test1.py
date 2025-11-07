# src/data.py

import os
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import MinMaxScaler

def load_nodes_edges(data_dir):
    nodes_path = os.path.join(data_dir, "Raw Dataset", "Nodes", "Nodes.csv")
    edges_path = os.path.join(data_dir, "Raw Dataset", "Edges", "Edges (Plant).csv")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    return nodes, edges

def load_temporal_data(data_dir, subfolder="Unit"):
    base_path = os.path.join(data_dir, "Raw Dataset", "Temporal Data", subfolder)
    prod = pd.read_csv(os.path.join(base_path, "Production .csv"))
    sales = pd.read_csv(os.path.join(base_path, "Sales Order.csv"))
    issues = pd.read_csv(os.path.join(base_path, "Factory Issue.csv"))
    delivery = pd.read_csv(os.path.join(base_path, "Delivery To distributor.csv"))
    return prod, sales, issues, delivery

def build_graph_from_edges(edges):
    G = nx.Graph()
    G.add_edges_from(edges[['node1', 'node2']].values.tolist())
    data = from_networkx(G)
    return data

def aggregate_stats_with_rolling(df, prefix):
    df_values = df.drop(columns=['Date'])
    rolling_mean_3 = df_values.rolling(window=3).mean().mean(axis=0)
    rolling_std_3 = df_values.rolling(window=3).std().mean(axis=0)
    rolling_mean_7 = df_values.rolling(window=7).mean().mean(axis=0)
    rolling_std_7 = df_values.rolling(window=7).std().mean(axis=0)
    return pd.DataFrame({
        'Node': df_values.columns,
        f'{prefix}_mean': df_values.mean(axis=0).values,
        f'{prefix}_std': df_values.std(axis=0).values,
        f'{prefix}_max': df_values.max(axis=0).values,
        f'{prefix}_min': df_values.min(axis=0).values,
        f'{prefix}_roll3_mean': rolling_mean_3.values,
        f'{prefix}_roll3_std': rolling_std_3.values,
        f'{prefix}_roll7_mean': rolling_mean_7.values,
        f'{prefix}_roll7_std': rolling_std_7.values
    })

def prepare_features(nodes, prod, sales, issues, delivery, data):
    prod_stats = aggregate_stats_with_rolling(prod, 'prod')
    sales_stats = aggregate_stats_with_rolling(sales, 'sales')
    issues_stats = aggregate_stats_with_rolling(issues, 'issues')
    delivery_stats = aggregate_stats_with_rolling(delivery, 'delivery')

    features = nodes[['Node']].merge(prod_stats, on='Node', how='left') \
                              .merge(sales_stats, on='Node', how='left') \
                              .merge(issues_stats, on='Node', how='left') \
                              .merge(delivery_stats, on='Node', how='left') \
                              .fillna(0)

    feature_cols = [c for c in features.columns if c != 'Node']

    scaler_x = MinMaxScaler()
    features_scaled = scaler_x.fit_transform(features[feature_cols])[:data.num_nodes]

    return features, features_scaled, scaler_x

def prepare_targets(nodes, prod, data):
    prod_vals = prod.drop(columns=['Date'])
    threshold = np.percentile(prod_vals.mean(axis=0), 25)
    bottleneck_labels = (prod_vals.mean(axis=0) < threshold).astype(float)[:data.num_nodes]

    future_prod = prod_vals.shift(-1).iloc[0]
    target_df = pd.DataFrame({'Node': prod_vals.columns, 'next_prod': future_prod.values})
    target_merged = nodes.merge(target_df, on='Node').fillna(0)
    targets = target_merged['next_prod'].values[:data.num_nodes].reshape(-1, 1)
    scaler_y = MinMaxScaler()
    targets_scaled = scaler_y.fit_transform(targets)

    return bottleneck_labels, targets_scaled, scaler_y

def create_train_val_masks(data, train_frac=0.8, seed=42):
    import torch
    torch.manual_seed(seed)
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    train_idx = perm[:int(train_frac*num_nodes)]
    val_idx = perm[int(train_frac*num_nodes):]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True

    return data
