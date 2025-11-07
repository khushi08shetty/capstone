# import streamlit as st
# from src import data, model, ui
# import torch
# import pandas as pd

# def main():
#     st.title("Hybrid Human-AI Supply Chain Bottleneck Detection System")

#     uploaded_files = ui.file_uploader_widget()

#     if all(uploaded_files.values()):
#         # Load data from uploaded files
#         nodes = pd.read_csv(uploaded_files['nodes'])
#         edges = pd.read_csv(uploaded_files['edges'])
#         prod = pd.read_csv(uploaded_files['production'])
#         sales = pd.read_csv(uploaded_files['sales'])
#         issues = pd.read_csv(uploaded_files['issues'])
#         delivery = pd.read_csv(uploaded_files['delivery'])

#         data_obj = data.build_graph_from_edges(edges)
#         features_df, features_scaled, scaler_x = data.prepare_features(nodes, prod, sales, issues, delivery, data_obj)

#         bottleneck_labels, targets_scaled, scaler_y = data.prepare_targets(nodes, prod, data_obj)

#         # Attach feature tensors & targets
#         import torch
#         data_obj.x = torch.tensor(features_scaled, dtype=torch.float)
#         data_obj.y_reg = torch.tensor(targets_scaled, dtype=torch.float)
#         data_obj.y_clf = torch.tensor(bottleneck_labels.values.reshape(-1, 1), dtype=torch.float)

#         data_obj = data.create_train_val_masks(data_obj)

#         # Initialize/load model
#         input_dim = data_obj.x.shape[1]
#         gnn_model = model.load_trained_model(input_dim)

#         # Get predictions and uncertainty zones
#         bottleneck_probs, uncertainty, trust_zones = model.predict_with_uncertainty(gnn_model, data_obj)

#         min_len = min(len(nodes), len(bottleneck_probs), len(trust_zones))

#         # Trim nodes DataFrame (reset index to keep it clean)
#         nodes_trimmed = nodes.iloc[:min_len].reset_index(drop=True)

#         result_df = pd.DataFrame({
#             'Node': nodes_trimmed['Node'],
#             'BottleneckProb': bottleneck_probs[:min_len],
#             'Uncertainty': uncertainty[:min_len],
#             'TrustZone': trust_zones[:min_len],
#         })

#         # Debug prints after trimming
#         st.write(f"Aligned Nodes count: {len(nodes_trimmed)}")
#         st.write(f"Aligned Predictions count: {len(bottleneck_probs[:min_len])}")
#         st.write(f"Aligned Trust zones count: {len(trust_zones[:min_len])}")

#         st.write(f"Nodes count: {len(nodes)}")
#         st.write(f"Predictions count: {len(bottleneck_probs)}")
#         st.write(f"Trust zones count: {len(trust_zones)}")


#         ui.show_predictions_table(result_df)
#         ui.plot_distributions(bottleneck_probs, uncertainty)

#         net_html = ui.draw_network_viz(nodes_trimmed,edges, bottleneck_probs, trust_zones)
#         st.components.v1.html(open(net_html, 'r').read(), height=650)

#         overrides = ui.human_override_ui(result_df)
#         if overrides:
#             st.write("User Overrides:", overrides)
#             # Apply overrides
#             for node_id, vals in overrides.items():
#                 result_df.loc[result_df['Node'] == node_id, 'TrustZone'] = vals['zone']
#                 result_df.loc[result_df['Node'] == node_id, 'BottleneckProb'] = vals['production']
#             # Save or propagate result_df if needed
#             # result_df.to_csv("final_predictions_with_overrides.csv", index=False)
            
#             csv = result_df.to_csv(index=False)
#             st.download_button("Download CSV", csv, file_name="final_predictions_with_overrides.csv", mime="text/csv")

#     else:
#         st.info("Please upload all required CSV files for nodes, edges, and temporal data.")

# if __name__ == "__main__":
#     main()

import streamlit as st
from src import data, model, ui
import torch
import pandas as pd

def main():
    st.title("Hybrid Human-AI Supply Chain Production Prediction and Bottleneck Detection System")

    uploaded_files = ui.file_uploader_widget()

    if all(uploaded_files.values()):
        # --- Data Loading ---
        nodes = pd.read_csv(uploaded_files['nodes'])
        edges = pd.read_csv(uploaded_files['edges'])
        prod = pd.read_csv(uploaded_files['production'])
        sales = pd.read_csv(uploaded_files['sales'])
        issues = pd.read_csv(uploaded_files['issues'])
        delivery = pd.read_csv(uploaded_files['delivery'])

        # --- Feature Construction ---
        data_obj = data.build_graph_from_edges(edges)
        features_df, features_scaled, scaler_x = data.prepare_features(nodes, prod, sales, issues, delivery, data_obj)
        bottleneck_labels, targets_scaled, scaler_y = data.prepare_targets(nodes, prod, data_obj)

        data_obj.x = torch.tensor(features_scaled, dtype=torch.float)
        data_obj.y_reg = torch.tensor(targets_scaled, dtype=torch.float)
        data_obj.y_clf = torch.tensor(bottleneck_labels.values.reshape(-1, 1), dtype=torch.float)
        data_obj = data.create_train_val_masks(data_obj)

        input_dim = data_obj.x.shape[1]
        gnn_model = model.load_trained_model(input_dim)

        # --- Model Prediction with Uncertainty ---
        mean_preds, variance, zones, bottleneck_probs = model.predict_with_uncertainty(gnn_model, data_obj)
        min_len = min(len(nodes), len(mean_preds), len(zones), len(bottleneck_probs))
        nodes_trimmed = nodes.iloc[:min_len].reset_index(drop=True)

        result_df = pd.DataFrame({
            'Node': nodes_trimmed['Node'],
            'PredictedProduction': mean_preds[:min_len],
            'ProductionUncertainty': variance[:min_len],
            'TrustZone': zones[:min_len],
            'BottleneckProb': bottleneck_probs[:min_len],
        })

        if 'result_df' not in st.session_state:
            st.session_state['result_df'] = result_df.copy()
        result_df = st.session_state['result_df']

        # ---- 1. Show Predictions Table First ----
        ui.show_predictions_table(result_df)

        # ---- 2. Show Plots/Graphs Next ----
        ui.plot_distributions(result_df['PredictedProduction'], result_df['ProductionUncertainty'])

        # Network graph visualization (using production prediction and trust zone)
        net_html = ui.draw_network_viz(nodes_trimmed, edges, result_df['PredictedProduction'], result_df['TrustZone'])
        st.components.v1.html(open(net_html, 'r').read(), height=650)

        # ---- 3. Node Inspection ----
        st.write("### Node Inspection")
        selected_node = st.selectbox("Select a node to inspect:", result_df['Node'])
        if selected_node:
            node_info = result_df[result_df['Node'] == selected_node].iloc[0]
            st.write(f"**Predicted production:** {node_info['PredictedProduction']:.4f}")
            st.write(f"**Production uncertainty:** {node_info['ProductionUncertainty']:.6f}")
            st.write(f"**Trust Zone:** {node_info['TrustZone']}")
            st.write(f"**Bottleneck Probability:** {node_info['BottleneckProb']:.4f}")

            node_features = nodes[nodes['Node'] == selected_node]
            st.write("Node attributes:")
            st.write(node_features)

            connected_edges = edges[(edges['node1'] == selected_node) | (edges['node2'] == selected_node)]
            st.write("Connected edges:")
            st.write(connected_edges)

        # ---- 4. Human-AI Override/Download Table ----
        st.write("### Human-in-the-loop Overrides")
        zone_filter = st.selectbox("Filter nodes by zone to override:", options=['all', 'green', 'amber', 'red'])
        if zone_filter == 'all':
            filtered_nodes = result_df['Node'].tolist()
        else:
            filtered_nodes = result_df[result_df['TrustZone'] == zone_filter]['Node'].tolist()

        overrides = ui.human_override_ui(result_df[result_df['Node'].isin(filtered_nodes)])

        if overrides:
            st.write("User Overrides:", overrides)
            for node_id, vals in overrides.items():
                st.session_state['result_df'].loc[st.session_state['result_df']['Node'] == node_id, 'TrustZone'] = vals['zone']
                st.session_state['result_df'].loc[st.session_state['result_df']['Node'] == node_id, 'PredictedProduction'] = vals['production']
            csv = st.session_state['result_df'].to_csv(index=False)
            st.download_button("Download Final Predictions CSV", csv, file_name="final_predictions_with_overrides.csv", mime="text/csv")
        result_df = st.session_state['result_df']

    else:
        st.info("Please upload all required CSV files for nodes, edges, and temporal data.")

if __name__ == "__main__":
    main()

