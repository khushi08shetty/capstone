# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import networkx as nx
# from pyvis.network import Network
# import tempfile


# def file_uploader_widget():
#     st.sidebar.title("Upload your data files")
#     uploaded_nodes = st.sidebar.file_uploader("Upload Nodes CSV", type=['csv'])
#     uploaded_edges = st.sidebar.file_uploader("Upload Edges CSV", type=['csv'])
#     uploaded_prod = st.sidebar.file_uploader("Upload Production CSV", type=['csv'])
#     uploaded_sales = st.sidebar.file_uploader("Upload Sales Order CSV", type=['csv'])
#     uploaded_issues = st.sidebar.file_uploader("Upload Factory Issue CSV", type=['csv'])
#     uploaded_delivery = st.sidebar.file_uploader("Upload Delivery CSV", type=['csv'])

#     uploaded_files = {
#         'nodes': uploaded_nodes,
#         'edges': uploaded_edges,
#         'production': uploaded_prod,
#         'sales': uploaded_sales,
#         'issues': uploaded_issues,
#         'delivery': uploaded_delivery,
#     }
#     return uploaded_files


# def draw_network_viz(nodes_df, edges_df, preds, zones):
#     import tempfile
#     from pyvis.network import Network
#     import networkx as nx

#     G = nx.from_pandas_edgelist(edges_df, source='node1', target='node2')

#     assert len(nodes_df) == len(preds) == len(zones), "Lengths of nodes, preds, and zones must be equal"

#     net = Network(height="600px", width="100%", notebook=False)
#     net.toggle_physics(False)
    
#     # Add nodes by their actual labels for pyvis
#     for node, pred, zone in zip(nodes_df['Node'], preds, zones):
#         color = {'green': '#28a745', 'amber': '#ffc107', 'red': '#dc3545'}.get(zone, '#6c757d')
#         size = pred * 50 + 10
#         net.add_node(node, label=node, color=color, size=size)

#     # Add edges by node labels, matching those added as nodes
#     for source, target in edges_df[['node1', 'node2']].itertuples(index=False):
#         if source in net.get_nodes() and target in net.get_nodes():
#             net.add_edge(source, target)

#     path = tempfile.mktemp(suffix=".html")
#     net.show(path, notebook=False)
#     return path



# def show_predictions_table(df):
#     st.subheader("Bottleneck Predictions with Trust Zones")
#     st.dataframe(df[['Node', 'BottleneckProb', 'Uncertainty', 'TrustZone']])


# def plot_distributions(preds, uncertainty):
#     st.subheader("Predicted Bottleneck Probability Distribution")
#     plt.figure(figsize=(10, 4))
#     sns.histplot(preds, bins=20)
#     st.pyplot(plt.gcf())
#     plt.clf()

#     st.subheader("Prediction Uncertainty Distribution")
#     plt.figure(figsize=(10, 4))
#     sns.histplot(uncertainty, bins=20, color='orange')
#     st.pyplot(plt.gcf())
#     plt.clf()


# def human_override_ui(df):
#     st.subheader("Human-in-the-loop Overrides")
#     nodes_to_override = st.multiselect("Select nodes to override:", df['Node'].tolist())
#     new_zone = st.selectbox("Set new zone for selected nodes:", ['green', 'amber', 'red'])
#     new_prod = st.number_input("Set new production value for selected nodes:", min_value=0.0)

#     if st.button("Submit Overrides"):
#         st.success(f"Overrides applied to {len(nodes_to_override)} nodes, zone set to {new_zone}, production set to {new_prod}")
#         # Return a dictionary of node: {zone, production} pairs
#         return {node: {'zone': new_zone, 'production': new_prod} for node in nodes_to_override}
#     return {}

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pyvis.network import Network
import tempfile


def file_uploader_widget():
    st.sidebar.title("Upload your data files")
    uploaded_nodes = st.sidebar.file_uploader("Upload Nodes CSV", type=['csv'])
    uploaded_edges = st.sidebar.file_uploader("Upload Edges CSV", type=['csv'])
    uploaded_prod = st.sidebar.file_uploader("Upload Production CSV", type=['csv'])
    uploaded_sales = st.sidebar.file_uploader("Upload Sales Order CSV", type=['csv'])
    uploaded_issues = st.sidebar.file_uploader("Upload Factory Issue CSV", type=['csv'])
    uploaded_delivery = st.sidebar.file_uploader("Upload Delivery CSV", type=['csv'])

    uploaded_files = {
        'nodes': uploaded_nodes,
        'edges': uploaded_edges,
        'production': uploaded_prod,
        'sales': uploaded_sales,
        'issues': uploaded_issues,
        'delivery': uploaded_delivery,
    }
    return uploaded_files


def draw_network_viz(nodes_df, edges_df, preds, zones):
    import tempfile
    from pyvis.network import Network
    import networkx as nx

    G = nx.from_pandas_edgelist(edges_df, source='node1', target='node2')

    assert len(nodes_df) == len(preds) == len(zones), "Lengths of nodes, preds, and zones must be equal"

    net = Network(height="600px", width="100%", notebook=False)
    net.toggle_physics(False)

    for node, pred, zone in zip(nodes_df['Node'], preds, zones):
        color = {'green': '#28a745', 'amber': '#ffc107', 'red': '#dc3545'}.get(zone, '#6c757d')
        size = pred * 50 + 10
        net.add_node(node, label=node, color=color, size=size)

    for source, target in edges_df[['node1', 'node2']].itertuples(index=False):
        if source in net.get_nodes() and target in net.get_nodes():
            net.add_edge(source, target)

    path = tempfile.mktemp(suffix=".html")
    net.write_html(path)
    return path


def show_predictions_table(df):
    st.subheader("Production Predictions with Trust Zones and Bottleneck Probabilities")
    st.dataframe(df[['Node', 'PredictedProduction', 'BottleneckProb', 'ProductionUncertainty', 'TrustZone']])


def plot_distributions(production_preds, uncertainty):
    st.subheader("Predicted Production Distribution")
    plt.figure(figsize=(10, 4))
    sns.histplot(production_preds, bins=20)
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Prediction Uncertainty Distribution")
    plt.figure(figsize=(10, 4))
    sns.histplot(uncertainty, bins=20, color='orange')
    st.pyplot(plt.gcf())
    plt.clf()


def human_override_ui(df):
    st.subheader("Human-in-the-loop Overrides")

    nodes_to_override = st.multiselect("Select nodes to override:", df['Node'].tolist())
    new_zone = st.selectbox("Set new zone for selected nodes:", ['green', 'amber', 'red'])
    new_prod = st.number_input("Set new production value for selected nodes:", min_value=0.0)

    if st.button("Submit Overrides"):
        st.success(f"Overrides applied to {len(nodes_to_override)} nodes, zone set to {new_zone}, production set to {new_prod}")
        return {node: {'zone': new_zone, 'production': new_prod} for node in nodes_to_override}
    return {}