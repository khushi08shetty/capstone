# # import torch
# # import torch.nn.functional as F
# # from torch_geometric.nn import GATConv, BatchNorm

# # class MultiTaskGNN(torch.nn.Module):
# #     """
# #     Graph Attention Network for multitask regression (forecasting) and classification (bottleneck)
# #     """
# #     def __init__(self, input_dim, hidden_dim=64, heads=4):
# #         super().__init__()
# #         self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.3)
# #         self.bn1 = BatchNorm(hidden_dim * heads)
# #         self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=0.3)
# #         self.bn2 = BatchNorm(hidden_dim)
# #         self.lin_reg = torch.nn.Linear(hidden_dim, 1)
# #         self.lin_clf = torch.nn.Linear(hidden_dim, 1)

# #     def forward(self, x, edge_index):
# #         x, _ = self.conv1(x, edge_index, return_attention_weights=True)
# #         x = self.bn1(x)
# #         x = F.elu(x)
# #         x = F.dropout(x, 0.4, training=self.training)
# #         x, _ = self.conv2(x, edge_index, return_attention_weights=True)
# #         x = self.bn2(x)
# #         x = F.elu(x)
# #         x = F.dropout(x, 0.4, training=self.training)
# #         return self.lin_reg(x), torch.sigmoid(self.lin_clf(x))


# # def train_multitask_model(model, data, epochs=200, lr=0.005, patience=30):
# #     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
# #     criterion_reg = torch.nn.MSELoss()
# #     criterion_clf = torch.nn.BCELoss()

# #     best_val_loss = float('inf')
# #     counter = 0
# #     for epoch in range(epochs):
# #         model.train()
# #         optimizer.zero_grad()
# #         reg_out, clf_out = model(data.x, data.edge_index)
# #         loss_reg = criterion_reg(reg_out[data.train_mask], data.y_reg[data.train_mask])
# #         loss_clf = criterion_clf(clf_out[data.train_mask], data.y_clf[data.train_mask])
# #         loss = loss_reg + loss_clf
# #         loss.backward()
# #         optimizer.step()

# #         model.eval()
# #         with torch.no_grad():
# #             val_reg, val_clf = model(data.x, data.edge_index)
# #             val_loss_reg = criterion_reg(val_reg[data.val_mask], data.y_reg[data.val_mask])
# #             val_loss_clf = criterion_clf(val_clf[data.val_mask], data.y_clf[data.val_mask])
# #             val_loss = val_loss_reg + val_loss_clf

# #         if val_loss < best_val_loss:
# #             best_val_loss = val_loss
# #             torch.save(model.state_dict(), 'best_multitask_gnn.pth')
# #             counter = 0
# #             print(f"Epoch {epoch}: New best val loss {val_loss:.6f}")
# #         else:
# #             counter += 1
# #             print(f"Epoch {epoch}: No improvement, patience {counter}/{patience}")

# #         if counter >= patience:
# #             print(f"Early stopping at epoch {epoch}")
# #             break

# # def load_trained_model(input_dim):
# #     model = MultiTaskGNN(input_dim)
# #     model.load_state_dict(torch.load('best_multitask_gnn.pth'))
# #     model.eval()
# #     return model

# # def predict_with_uncertainty(model, data, mc_samples=20):
# #     import numpy as np
# #     model.train()  # Enable dropout for MC Dropout
# #     mc_preds = []

# #     for _ in range(mc_samples):
# #         pred_reg, pred_clf = model(data.x, data.edge_index)
# #         mc_preds.append(pred_clf.detach().cpu().numpy())

# #     mc_preds = np.array(mc_preds)
# #     mean_preds = mc_preds.mean(axis=0).flatten()
# #     variance = mc_preds.var(axis=0).flatten()

# #     # Define zones by uncertainty variance thresholds
# #     thresholds = {'green': 0.02, 'amber': 0.05}
# #     zones = ['green' if var < thresholds['green'] else 'amber' if var < thresholds['amber'] else 'red' for var in variance]

# #     return mean_preds, variance, zones

# # if __name__ == "__main__":
# #     import pandas as pd
# #     from src import data as data_module

# #     # Load your CSVs (edit file paths as needed)
# #     nodes = pd.read_csv(r"C:\Users\khushi shetty\Downloads\project_root\project_root\data\Raw Dataset\Nodes\Nodes.csv")
# #     edges = pd.read_csv(r"C:\Users\khushi shetty\Downloads\project_root\project_root\data\Raw Dataset\Edges\Edges (Plant).csv")
# #     prod = pd.read_csv(r"C:\Users\khushi shetty\Downloads\project_root\project_root\data\Raw Dataset\Temporal Data\Unit\Production .csv")
# #     sales = pd.read_csv(r"C:\Users\khushi shetty\Downloads\project_root\project_root\data\Raw Dataset\Temporal Data\Unit\Sales Order.csv")
# #     issues = pd.read_csv(r"C:\Users\khushi shetty\Downloads\project_root\project_root\data\Raw Dataset\Temporal Data\Unit\Factory Issue.csv")
# #     delivery = pd.read_csv(r"C:\Users\khushi shetty\Downloads\project_root\project_root\data\Raw Dataset\Temporal Data\Unit\Delivery To distributor.csv")

# #     # Build graph and features
# #     data_obj = data_module.build_graph_from_edges(edges)
# #     features_df, features_scaled, scaler_x = data_module.prepare_features(
# #         nodes, prod, sales, issues, delivery, data_obj
# #     )
# #     bottleneck_labels, targets_scaled, scaler_y = data_module.prepare_targets(
# #         nodes, prod, data_obj
# #     )

# #     # Attach tensors to graph data object
# #     data_obj.x = torch.tensor(features_scaled, dtype=torch.float)
# #     data_obj.y_reg = torch.tensor(targets_scaled, dtype=torch.float)
# #     data_obj.y_clf = torch.tensor(bottleneck_labels.values.reshape(-1, 1), dtype=torch.float)
# #     data_obj = data_module.create_train_val_masks(data_obj)

# #     input_dim = data_obj.x.shape[1]
# #     model = MultiTaskGNN(input_dim)
# #     train_multitask_model(model, data_obj)

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv, BatchNorm
# import numpy as np


# class MultiTaskGNN(torch.nn.Module):
#     """
#     Graph Attention Network for multitask regression (forecasting) and classification (bottleneck)
#     """
#     def __init__(self, input_dim, hidden_dim=64, heads=4):
#         super().__init__()
#         self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.5)  # increased dropout for uncertainty
#         self.bn1 = BatchNorm(hidden_dim * heads)
#         self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=0.5)
#         self.bn2 = BatchNorm(hidden_dim)
#         self.lin_reg = torch.nn.Linear(hidden_dim, 1)  # production prediction
#         self.lin_clf = torch.nn.Linear(hidden_dim, 1)  # bottleneck classification

#     def forward(self, x, edge_index):
#         x, _ = self.conv1(x, edge_index, return_attention_weights=True)
#         x = self.bn1(x)
#         x = F.elu(x)
#         x = F.dropout(x, 0.4, training=self.training)
#         x, _ = self.conv2(x, edge_index, return_attention_weights=True)
#         x = self.bn2(x)
#         x = F.elu(x)
#         x = F.dropout(x, 0.4, training=self.training)
#         return self.lin_reg(x), torch.sigmoid(self.lin_clf(x))


# def train_multitask_model(model, data, epochs=200, lr=0.005, patience=30):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
#     criterion_reg = torch.nn.MSELoss()
#     criterion_clf = torch.nn.BCELoss()

#     best_val_loss = float('inf')
#     counter = 0
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         reg_out, clf_out = model(data.x, data.edge_index)
#         loss_reg = criterion_reg(reg_out[data.train_mask], data.y_reg[data.train_mask])
#         loss_clf = criterion_clf(clf_out[data.train_mask], data.y_clf[data.train_mask])
#         loss = loss_reg + loss_clf
#         loss.backward()
#         optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             val_reg, val_clf = model(data.x, data.edge_index)
#             val_loss_reg = criterion_reg(val_reg[data.val_mask], data.y_reg[data.val_mask])
#             val_loss_clf = criterion_clf(val_clf[data.val_mask], data.y_clf[data.val_mask])
#             val_loss = val_loss_reg + val_loss_clf

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), 'best_multitask_gnn.pth')
#             counter = 0
#             print(f"Epoch {epoch}: New best val loss {val_loss:.6f}")
#         else:
#             counter += 1
#             print(f"Epoch {epoch}: No improvement, patience {counter}/{patience}")

#         if counter >= patience:
#             print(f"Early stopping at epoch {epoch}")
#             break


# def load_trained_model(input_dim):
#     model = MultiTaskGNN(input_dim)
#     model.load_state_dict(torch.load('best_multitask_gnn.pth'))
#     model.eval()
#     return model


# def predict_with_uncertainty(model, data, mc_samples=50):
#     model.train()  # Enable dropout for MC Dropout stochastic forward passes
#     mc_preds = []
#     mc_clf_preds = []

#     for _ in range(mc_samples):
#         pred_reg, pred_clf = model(data.x, data.edge_index)
#         mc_preds.append(pred_reg.detach().cpu().numpy())
#         mc_clf_preds.append(pred_clf.detach().cpu().numpy())

#     mc_preds = np.array(mc_preds)
#     mean_preds = mc_preds.mean(axis=0).flatten()
#     variance = mc_preds.var(axis=0).flatten()

#     mc_clf_preds = np.array(mc_clf_preds)
#     mean_bottleneck = mc_clf_preds.mean(axis=0).flatten()

#     thresholds = {'green': 0.025, 'amber': 0.035}
#     zones = ['green' if var < thresholds['green'] else 'amber' if var < thresholds['amber'] else 'red' for var in variance]

#     return mean_preds, variance, zones, mean_bottleneck

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from src import data as data_module

class MultiTaskGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, heads=4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.5)
        self.bn1 = BatchNorm(hidden_dim * heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=0.5)
        self.bn2 = BatchNorm(hidden_dim)
        self.lin_reg = torch.nn.Linear(hidden_dim, 1)
        self.lin_clf = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x, _ = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, 0.4, training=self.training)
        x, _ = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, 0.4, training=self.training)
        return self.lin_reg(x), torch.sigmoid(self.lin_clf(x))


def train_multitask_model(model, data, epochs=200, lr=0.005, patience=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion_reg = torch.nn.MSELoss()
    criterion_clf = torch.nn.BCELoss()

    best_val_loss = float('inf')
    counter = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        reg_out, clf_out = model(data.x, data.edge_index)
        loss_reg = criterion_reg(reg_out[data.train_mask], data.y_reg[data.train_mask])
        loss_clf = criterion_clf(clf_out[data.train_mask], data.y_clf[data.train_mask])
        loss = loss_reg + loss_clf
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_reg, val_clf = model(data.x, data.edge_index)

            val_reg_np = val_reg[data.val_mask].cpu().numpy()
            val_reg_true = data.y_reg[data.val_mask].cpu().numpy()

            val_clf_np = (val_clf[data.val_mask].cpu().numpy() > 0.5).astype(int)
            val_clf_true = data.y_clf[data.val_mask].cpu().numpy().astype(int)

            val_loss_reg = criterion_reg(val_reg[data.val_mask], data.y_reg[data.val_mask])
            val_loss_clf = criterion_clf(val_clf[data.val_mask], data.y_clf[data.val_mask])
            val_loss = val_loss_reg + val_loss_clf

            rmse = np.sqrt(mean_squared_error(val_reg_true, val_reg_np))
            mae = mean_absolute_error(val_reg_true, val_reg_np)
            accuracy = accuracy_score(val_clf_true, val_clf_np)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_multitask_gnn.pth')
            counter = 0
            print(f"Epoch {epoch}: New best val loss {val_loss:.6f}, RMSE={rmse:.4f}, MAE={mae:.4f}, Accuracy={accuracy:.4f}")
        else:
            counter += 1
            print(f"Epoch {epoch}: No improvement, patience {counter}/{patience}, RMSE={rmse:.4f}, MAE={mae:.4f}, Accuracy={accuracy:.4f}")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break


if __name__ == "__main__":
    # File paths
    nodes = pd.read_csv(r"C:\Users\khushi shetty\Downloads\project_root\project_root\data\Raw Dataset\Nodes\Nodes.csv")
    edges = pd.read_csv(r"C:\Users\khushi shetty\Downloads\project_root\project_root\data\Raw Dataset\Edges\Edges (Plant).csv")
    prod = pd.read_csv(r"C:\Users\khushi shetty\Downloads\project_root\project_root\data\Raw Dataset\Temporal Data\Unit\Production .csv")
    sales = pd.read_csv(r"C:\Users\khushi shetty\Downloads\project_root\project_root\data\Raw Dataset\Temporal Data\Unit\Sales Order.csv")
    issues = pd.read_csv(r"C:\Users\khushi shetty\Downloads\project_root\project_root\data\Raw Dataset\Temporal Data\Unit\Factory Issue.csv")
    delivery = pd.read_csv(r"C:\Users\khushi shetty\Downloads\project_root\project_root\data\Raw Dataset\Temporal Data\Unit\Delivery To distributor.csv")

    # Prepare data
    data_obj = data_module.build_graph_from_edges(edges)
    _, features_scaled, _ = data_module.prepare_features(nodes, prod, sales, issues, delivery, data_obj)
    bottleneck_labels, targets_scaled, _ = data_module.prepare_targets(nodes, prod, data_obj)

    data_obj.x = torch.tensor(features_scaled, dtype=torch.float)
    data_obj.y_reg = torch.tensor(targets_scaled, dtype=torch.float)
    data_obj.y_clf = torch.tensor(bottleneck_labels.values.reshape(-1,1), dtype=torch.float)
    data_obj = data_module.create_train_val_masks(data_obj)

    input_dim = data_obj.x.shape[1]
    model = MultiTaskGNN(input_dim)

    # Train model with evaluation metrics logged
    train_multitask_model(model, data_obj)
