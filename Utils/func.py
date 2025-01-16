def func(combined_list):

    import networkx as nx
    import matplotlib.pyplot as plt

    # Create a new graph for the patient
    G = nx.Graph()

    # Add a central node for the patient
    patient_node = 'Patient 1'
    G.add_node(patient_node, label='Patient')

    # Add nodes for symptoms and organs, and connect them to the patient node
    for symptom, duration, organ in combined_list:
        # Add symptom node and connect to patient
        G.add_node(symptom, label='Symptom')
        G.add_edge(patient_node, symptom)

        # Add organ node if it is specified and connect to the symptom
        if organ != 'nil organ':
            G.add_node(organ, label='Organ')
            G.add_edge(symptom, organ)

        # Optional: Set edge weights based on symptom durations (if applicable)
        if duration != 'nil duration':
            G[patient_node][symptom]['weight'] = duration

    # Function to plot the graph
    def plot_graph(G):
        pos = nx.spring_layout(G, k=0.5, scale=2)  # Adjust k and scale for better spacing
        plt.figure(figsize=(12, 12))  # Adjust figure size for better display
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', alpha=0.7, edge_color='gray')
    
        # Draw edge labels and weights
        weights = nx.get_edge_attributes(G, 'weight')
        if weights:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_color='green', font_size=8, label_pos=0.3)

        plt.title('Patient Symptom and Organ Network')
        plt.show()

    # Plot the graph
    plot_graph(G)




    from transformers import BertTokenizer, BertModel, BertConfig
    import torch
    import torch.nn as nn
    import re
    from torchcrf import CRF
    from transformers import AutoTokenizer, AutoModel
    from transformers import BertTokenizer, BertModel, BertConfig
    import os

    # Get the current file's directory (Utils directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct paths to the model files
    config_path = os.path.join(current_dir, "config.json")
    vocab_path = os.path.join(current_dir, "vocab.txt")
    model_path = os.path.join(current_dir, "pytorch_model.bin")

    config = BertConfig.from_pretrained(config_path)
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    biobert_model = BertModel.from_pretrained(model_path, config=config, local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    feature_matrices = []


    matrix = [None] * len(G.nodes())  # Initialize a list to store embeddings in the correct order

    edge_list = list(G.edges())
    unique_nodes = sorted(set(G.nodes()))  # Ensure all nodes are included and sorted

    node_mapping = {node: i for i, node in enumerate(unique_nodes)}  # Mapping nodes to indices

    for node in G.nodes():
        if node not in node_mapping:
            print(f"Node '{node}' not found in node_mapping.")
            continue
    
        print(node)
        text = node  # Text to be tokenized
        # inputs = tokenizer(text, return_tensors='pt').to(device)

        # with torch.no_grad():
        #     outputs = biobert_model(**inputs)
        #     embeddings = outputs.last_hidden_state

        # Averaging the embeddings
        #averaged_matrix = embeddings[0].mean(dim=0).view(1, -1)

        # Place the embedding in the correct row based on the node_mapping
    
        inputs = tokenizer(text, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = biobert_model(**inputs)
            embeddings = outputs.last_hidden_state

        # Averaging the embeddings
        averaged_matrix = embeddings[0].mean(dim=0).view(1, -1)
        matrix[node_mapping[node]] = averaged_matrix
    

    # Remove None entries (if any) and convert the list to a tensor
    matrix = torch.cat([m for m in matrix if m is not None], dim=0)  # (num_nodes, hidden_size)
    feature_matrices.append(matrix)

    print(feature_matrices)



    main_edge_list = []
    lis = []


    matrix = []
    print(G.nodes)

    edge_list = list(G.edges())
    unique_nodes = sorted(set(node for edge in edge_list for node in edge))  # Sort the unique nodes

    # Create a mapping from node labels to integers in a sorted order
    node_mapping = {node: i for i, node in enumerate(unique_nodes)}

    # Convert edge list to numeric representation
    numeric_edge_list = [[node_mapping[u], node_mapping[v]] for u, v in edge_list]

    # Create tensor and move to the appropriate device
    edge_tensor = torch.tensor(numeric_edge_list, dtype=torch.long).to(device)
    lis.append(edge_tensor)

    # Store the edge list in main_edge_list
    main_edge_list.append(edge_list)





    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SortAggregation
    from torch_geometric.data import Data
    import torch.optim as optim
    import numpy as np
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class GCN_SortPool_CNN(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, k):
            super(GCN_SortPool_CNN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            #self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.sort_pool = SortAggregation(k=k) #prioritizing nodes (dk what k is)
            self.cnn1d = nn.Conv1d(in_channels=hidden_channels, out_channels=32, kernel_size=2)
        
            # Calculate the correct input size for the fully connected layer
            cnn_output_size = 32 * (k - 1)  # Because kernel_size=2 reduces length by 1
            self.fc = nn.Linear(cnn_output_size, out_channels)
        
        def forward(self, x, edge_index):
            # GCN Layers
            x = F.relu(self.conv1(x, edge_index))
            #x = F.relu(self.conv2(x, edge_index))
        
            # SortPooling
            x = self.sort_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))  # Create a dummy batch tensor
        
            # Reshape for 1D CNN
            x = x.view(x.size(0), -1, self.sort_pool.k)  # Reshape to (batch_size, hidden_channels, k)
        
            # 1D CNN
            x = F.relu(self.cnn1d(x))
            x = x.view(x.size(0), -1)  # Flatten
        
            # Fully Connected + Sigmoid
            x = self.fc(x)
            x = torch.sigmoid(x)  # Apply sigmoid activation
            return x

    # Load the model
    model = GCN_SortPool_CNN(in_channels=768, hidden_channels=32, out_channels=1, k=14).to(device)

    import os

    # Get the current file's directory (Utils directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct path to the model file
    model_path = os.path.join(current_dir, "gcn_sortpool_cnn.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    pred_1 = [0]
    pred_1_shuffled = [0]
    # Create a list of Data objects for the test data
    graphs = [Data(x=feature_matrices[i].to(device), edge_index=lis[i].to(device)) for i in range(len(feature_matrices))]
    print(len(pred_1))
    # Perform inference
    with torch.no_grad():
        predictions = []
        targets = []
        c=0
        for i, graph in enumerate(graphs):
            c+=1
            if c>44:
                break
            output = model(graph.x, graph.edge_index.view(2, -1))
            predicted = (output > 0.5).float()  # Convert probabilities to binary predictions
            predictions.append(predicted.cpu().numpy())  # Move output to CPU and convert to numpy array
            targets.append(pred_1_shuffled[i])  # Collect the targets

    # Convert lists to numpy arrays for easy comparison
    predictions = np.array(predictions).flatten()  # Flatten to match target shape
    targets = np.array(targets).flatten()  # Flatten to match prediction shape

    # Calculate number of correct predictions
    correct_predictions = np.sum(predictions == targets)
    total_predictions = len(targets)
    accuracy = correct_predictions / total_predictions

    # Output number of correct predictions, accuracy, and predictions
    print(f"Number of correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f}")

    q = -1
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        print(f"Graph {i} - Prediction: {pred}, Target: {target}")
        q = pred

    s = ""
    if q == 0:
        s = "MRI NOT REQUIRED"
    elif q == 1:
        s = "MRI NEEDED NOW !!!"
    else:
        s = "PREDICTION_NEEDED"


    return [('', s)]