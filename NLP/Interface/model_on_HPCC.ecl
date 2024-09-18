IMPORT PYTHON3 AS PY;

rec := RECORD
    STRING SYMPTOM_TO_ORGAN;
    STRING SYMPTOM_WITH_DURATION;
    STRING RESULT;  
END;

med_data := 'A 38-year-old Middle Eastern male presented to the emergency department with sudden onset dense right hemiparesis, 2 months of headache, right facial droop and aphasia.His background history included a previous ischaemic stroke 15 months ago treated in a different institution.At that time, he presented with a headache and dysphasia and a CT brain showed a left temporo-parietal infarct.By 6 months later, he had returned to his baseline of full functional independence on treatment with clopidogrel 75 mg once daily and atorvastatin 40 mg daily.\nOn this admission, CT brain showed a left M1 occlusion and the patient was treated with intravenous alteplase and thrombectomy.\nHis CT angiogram intracranial confirmed an acute occlusion of the M1 portion of the left middle cerebral artery and revealed two separate foci of soft plaques arising from the posterior wall of the origin of the left and right ICA with accompanying carotid webs on both sides ().His MRA carotids showed a haemorrhagic “plaque” at the origin of the left ICA but no high-grade ICA stenosis or any evidence of dissection ().Axial fat-saturated T1W MRI demonstrated a crescentic hyperintense signal at the posterior aspect of the origin of the left ICA consistent with haemorrhage within the known carotid web';
STREAMED DATASET(rec) func(STRING str) := EMBED(PY)
import random
import re

l = []
l.append(str)
osl_shuffled = l

print(len(osl_shuffled))

import re

# Assuming lis contains the text data
dataset = osl_shuffled
final_list_of_sentences = []

# Compile the regex patterns once for efficiency
sentence_splitter = re.compile(r'\.\s*')
word_tokenizer = re.compile(r'\b[\w-]+\b')

# Process each text in the dataset
for text in dataset:
    # Split text into sentences and remove empty sentences
    sentences = [sentence.strip() + '.' for sentence in sentence_splitter.split(text) if sentence.strip()]
    
    # Tokenize, lowercase, and clean each sentence
    tokenized_sentences = [' '.join(word_tokenizer.findall(sentence.lower())) for sentence in sentences]
    
    # Append the cleaned sentences to the final list
    final_list_of_sentences.append(tokenized_sentences)

print(len(final_list_of_sentences))

for i in range(len(final_list_of_sentences)):
    if not final_list_of_sentences[i]:
        print('Empty')
    final_list_of_sentences[i] = [sentence for sentence in final_list_of_sentences[i] if sentence]

import torch 
import torch.nn as nn
import re
from torchcrf import CRF
from transformers import AutoTokenizer, AutoModel

# Define BiLSTM and MOD classes
class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.input_size = 768
        self.hidden_size = 256
        self.num_layers = 2
        self.dropout = 0.3
        self.bilstm = nn.LSTM(input_size=self.input_size, hidden_size=128, 
                            num_layers=self.num_layers, batch_first=True, 
                            dropout=self.dropout, bidirectional=True)

    def forward(self, embeddings):
        lstm_out, _ = self.bilstm(embeddings)
        return lstm_out

class MOD(nn.Module):
    def __init__(self, input_size, num_labels):
        super(MOD, self).__init__()
        self.bilstm = BiLSTM()
        self.linear = nn.Linear(256, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, x):
        lstm_output = self.bilstm(x)
        embeddings = self.linear(lstm_output)
        return embeddings

# Load BioBERT tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
# biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1").to(device)
from transformers import BertTokenizer, BertModel, BertConfig
config_path = r"/var/lib/HPCCSystems/mydropzone/config.json"
vocab_path = r"/var/lib/HPCCSystems/mydropzone/vocab.txt"
model_path = r"/var/lib/HPCCSystems/mydropzone/pytorch_model.bin"

config = BertConfig.from_pretrained(config_path)
tokenizer = BertTokenizer.from_pretrained(vocab_path)
biobert_model = BertModel.from_pretrained(model_path, config=config, local_files_only=True)

# Load the trained model
model = MOD(768, 6).to(device)  # Ensure model is on the same device
model.load_state_dict(torch.load(r"/var/lib/HPCCSystems/mydropzone/best_model.pth", map_location=device))
model.eval()

# Initialize lists for storing results
all_symptoms_ALL = []
symptoms_wout_duration_ALL = []
symptom_with_organ_ALL = []
new_dict_ALL = []

# Assuming final_list_of_sentences is already defined
for i in range(len(final_list_of_sentences)):    
    sentences = final_list_of_sentences[i]
    tsl_tokenized = []
    
    for sentence in sentences:
        tokens = re.findall(r'\b[\w-]+\b', sentence.lower())
        tsl_tokenized.append(tokens)
    
    vocab_test = []
    for sentence in tsl_tokenized:
        for token in sentence:
            if token not in vocab_test:
                vocab_test.append(token)
    
    word_embeddings = []
    for word in vocab_test:
        tokens = tokenizer(word, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = biobert_model(**tokens)
        embedding = outputs.last_hidden_state[0][0]
        word_embeddings.append(embedding)

    word_embeddings_test = torch.stack(word_embeddings).to(device)
    sentence_embeddings_test = []
    each_sentence = []
    
    for k in range(len(tsl_tokenized)):
        each_word_embeddings_test = []
        for token in tsl_tokenized[k]:
            word_embedding_tensor_test = word_embeddings_test[vocab_test.index(token)]  #biobert embeddingssssssssssss
            each_word_embeddings_test.append(word_embedding_tensor_test)
        sentence_embeddings_test.extend(each_word_embeddings_test)
        each_sentence.append(each_word_embeddings_test)

    pred = []
    for j, sentence in enumerate(each_sentence):
        if len(sentence) == 0:
            continue
        with torch.no_grad():
            print(i, j)
            # Stack the sentence tensors and move them to the device
            sentence_tensor = torch.stack(sentence).to(device).view(1, -1, 768)
            output = model(sentence_tensor)
            prediction = model.crf.decode(output)
            pred.append(prediction)

    # Create lists for different labels
    lis_1 = []
    lis_3 = []
    lis_4 = []
    lis_5 = []

    for i in range(len(pred)):
        l1 = {}
        l3 = {}
        l4 = {}
        l5 = {}
        prediction = pred[i]
        s = ''
        for k in range(len(prediction)):
            if prediction[k][0] != 0:
                s += tsl_tokenized[i][k] + ' '
            elif k != 0 and prediction[k][0] == 0:
                t = prediction[k-1][0]
                if t != 0:
                    if t == 1:
                        l1[s.strip()] = k
                    elif t == 2:
                        l1[s.strip()] = k
                    elif t == 3:
                        l3[s.strip()] = k
                    elif t == 4:
                        l4[s.strip()] = k
                    elif t == 5:
                        l5[s.strip()] = k
                    s = ''
        if prediction[-1][0] != 0:
            t = prediction[-1][0]
            if t == 1:
                l1[s.strip()] = len(prediction)
            elif t == 2:
                l1[s.strip()] = len(prediction)
            elif t == 3:
                l3[s.strip()] = len(prediction)
            elif t == 4:
                l4[s.strip()] = len(prediction)
            elif t == 5:
                l5[s.strip()] = len(prediction)

        lis_1.append(l1)
        lis_3.append(l3)
        lis_4.append(l4)
        lis_5.append(l5)

    symptoms_wout_duration = []
    for l1 in lis_1:
        if l1:
            for key in l1.keys():
                symptoms_wout_duration.append(key)
    
    new_dict = {}
    for i in range(len(lis_3)):
        if not lis_3[i]:
            #new_dict_ALL.append(new_dict)
            continue
        for key_3, value_3 in lis_3[i].items():
            closest_key = None
            minimum_difference = float('inf')
            for key_1, value_1 in lis_1[i].items():
                difference = abs(value_3 - value_1)
                if difference < minimum_difference:
                    minimum_difference = difference
                    closest_key = key_1
            new_dict[key_3] = closest_key

    all_symptoms_ALL.append(symptoms_wout_duration)
    symptoms_wout_duration_ALL.append(symptoms_wout_duration)
    new_dict_ALL.append(new_dict)

# print(all_symptoms_ALL)
# print(symptoms_wout_duration_ALL)
# print(symptom_with_organ_ALL)
# print(new_dict_ALL)

l = []
for i in range(len(new_dict_ALL) - 1):
    l.append(new_dict_ALL[i])
    
l.append(new_dict_ALL[len(new_dict_ALL) - 1])

symptom_with_duration_ALL = l

time_units = ['second', 'minute', 'hour', 'day', 'week', 'month', 'year', 'decade', 'seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years', 'decades']

def replace_keys(data, time_units):
    new_data = []
    for item in data:
        new_item = {}
        for key, value in item.items():
            matched_word = next((unit for unit in time_units if unit in key.lower()), None)
            if matched_word and value is not None:
                # Singularize the matched word if necessary
                if matched_word.endswith('s'):
                    matched_word = matched_word[:-1]
                new_item[matched_word] = value
        new_data.append(new_item)
    return new_data


# symptom_with_duration_ALL = [{}, {'88 to 96': None}, {}, {}, {'distribution': None}, {'2 weeks': 'cough'}, {'the same year': None}, {}]

updated_data = replace_keys(symptom_with_duration_ALL, time_units)

symptom_with_duration_ALL = updated_data
print(symptom_with_duration_ALL)

from groq import Groq
import ast
import time
import re
import httpx  # Ensure you have httpx installed

# Initialize the Groq client
client = Groq(api_key="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")

# Constants
REQUESTS_PER_MINUTE = 30
REQUESTS_PER_SECOND = REQUESTS_PER_MINUTE / 60
TOKENS_PER_MINUTE = 30000
TOKENS_PER_SECOND = TOKENS_PER_MINUTE / 60
EXPECTED_TOKENS_PER_REQUEST = 200  # Estimate based on average usage

def organ(medical_text, symptoms):
    prompt = (
        f"Use the following medical text: {medical_text} "
        f"Now make a list of dictionaries with key as symptom and value as organ (the given text should mention a specific organ, if the medical text doesn't have an organ for that symptom make the value 'unspecified'). "
        f"Answer should be ['symptom1':'organ1','symptom2':'organ2',...and so on]. "
        f"Remember to use the symptoms in the list only: {symptoms}. "
        f"The key and value should be strictly present in the above text. No generation of generic organs. "
        f"Give me the final list of dictionaries. I mean a Python list of dictionaries only, not a string. "
        f"Strictly, each value should be in the medical text given or else make it 'unspecified' and the organ should be in one word. "
        f"Respond with only the list of dictionaries. Remove any duplicate dictionaries. Do not include any other text in your response."
    )
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",  # Changed to the correct model
    )
    final_summary = response.choices[0].message.content
    return final_summary

def parse_response(response):
    # Clean the response string
    response = response.strip()
    
    # Remove the leading and trailing brackets if any
    if response.startswith('['):
        response = response[1:]
    if response.endswith(']'):
        response = response[:-1]
    
    # Replace single quotes with double quotes
    response = response.replace("'", '"')
    
    # Regex pattern to match key-value pairs
    pattern = re.compile(r'"([^"]+)":\s*"([^"]+)"')

    # Find all matches
    matches = pattern.findall(response)

    # Build dictionaries from the extracted key-value pairs
    parsed_data = [{key.strip(): value.strip()} for key, value in matches]

    return parsed_data

def write_list_to_file(data_list, file_path):
    """Writes a list to a file in append mode, with each item on a new line."""
    with open(file_path, 'a') as file:  # Open file in append mode
        for item in data_list:
            file.write(f"{item}\n")

def organ_with_retry(medical_text, symptoms, retries=3):
    """Attempts to call the organ function with retry logic."""
    for attempt in range(retries):
        try:
            return organ(medical_text, symptoms)
        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    print("Request failed after multiple attempts.")
    return []  # Return an empty list or handle as needed




# Initialize an empty list to store the results
symptom_with_organ_ALL = []
c = 0

for i in range(len(all_symptoms_ALL)):
    try:
        # Generate response for each dataset entry
        funResponse = organ_with_retry(dataset[i], all_symptoms_ALL[i])
        
        # Convert the response string to a Python list of dictionaries
        symptom_with_organ = parse_response(funResponse)
        print(symptom_with_organ)
        print(c)
        
        # Write the results to a file
        
        c += 1
        
        # Rate limiting: Delay to maintain requests per minute limit
        time.sleep(1 / REQUESTS_PER_SECOND)  # Delay to respect requests per second
        time.sleep(EXPECTED_TOKENS_PER_REQUEST / TOKENS_PER_SECOND)  # Delay to respect token limits

        # Append the parsed data to the result list
        symptom_with_organ_ALL.append(symptom_with_organ)

    except (httpx.ReadTimeout, httpx.ConnectError) as e:
        # Handle Groq API related errors
        print(f"Error calling Groq API for index {i}: {e}")
        continue  # Skip to the next index
    
    except (SyntaxError, ValueError) as e:
        # Handle parsing errors
        print(f"Error parsing response for index {i}: {e}")
        continue  # Skip to the next index

# Initialize an empty list to store the key-value pairs
key_value_pairs = []

# Iterate through the list of dictionaries
for graph in symptom_with_organ_ALL:
    for item in graph:
        if isinstance(item, dict):  # Ensure the item is a dictionary
            for k, v in item.items():
                if v != 'unspecified':
                    key_value_pairs.append(f"{k}: {v}")


        

import networkx as nx

# Initialize a list to store graphs
graphs = []

# Process each patient's symptom data to create a graph
for i, patient_data in enumerate(symptom_with_organ_ALL):
    # Create a new graph
    G = nx.Graph()
    
    # Add a central node for the patient
    patient_node = f'Patient {i+1}'
    G.add_node(patient_node, label='Patient')
    
    # Add nodes for symptoms and connect them to the patient node
    for item in patient_data:
        for symptom, organ in item.items():
            if organ != 'unspecified':
                G.add_node(symptom, label='Symptom')
                G.add_edge(patient_node, symptom)
                # Add nodes for organs and connect them to the symptoms
                if organ != 'unspecified':
                    G.add_node(organ, label='Organ')
                    G.add_edge(symptom, organ)
            else:
                G.add_node(symptom, label='Symptom')
                G.add_edge(patient_node, symptom)

    # Optional: Set edge weights based on symptom durations
    if i < len(symptom_with_duration_ALL):
        for symptom, duration in symptom_with_duration_ALL[i].items():
            if G.has_edge(patient_node, symptom):
                G[patient_node][symptom]['weight'] = time_units.index(duration)

    # Append the graph for the patient to the list
    graphs.append(G)


from transformers import BertTokenizer, BertModel, BertConfig
import torch
import numpy as np

# Load the tokenizer and model
# config_path = r"/var/lib/HPCCSystems/mydropzone/config.json"
# vocab_path = r"/var/lib/HPCCSystems/mydropzone/vocab.txt"
# model_path = r"/var/lib/HPCCSystems/mydropzone/pytorch_model.bin"

# config = BertConfig.from_pretrained(config_path)
# tokenizer = BertTokenizer.from_pretrained(vocab_path)
# model = BertModel.from_pretrained(model_path, config=config, local_files_only=True)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

feature_matrices = []

for G in graphs:
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
        if node not in vocab_test:
            inputs = tokenizer(text, return_tensors='pt').to(device)

            with torch.no_grad():
                outputs = biobert_model(**inputs)
                embeddings = outputs.last_hidden_state

            # Averaging the embeddings
            averaged_matrix = embeddings[0].mean(dim=0).view(1, -1)
            matrix[node_mapping[node]] = averaged_matrix
        else:
            matrix[node_mapping[node]] = word_embeddings_test[vocab_test.index(node)].view(1, -1)

    # Remove None entries (if any) and convert the list to a tensor
    matrix = torch.cat([m for m in matrix if m is not None], dim=0)  # (num_nodes, hidden_size)
    feature_matrices.append(matrix)

# Convert feature_matrices to numpy array if needed
#feature_matrices_np = [matrix.cpu().numpy() for matrix in feature_matrices]

# Print the numpy arrays if needed

# Assuming `device` has been defined as before (CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_edge_list = []
lis = []

for G in graphs:
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

# Print the tensors
#for edge_tensor in lis:
    #print(edge_tensor)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SortAggregation
from torch_geometric.data import Data
import torch.optim as optim

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

model.load_state_dict(torch.load('/var/lib/HPCCSystems/mydropzone/gcn_sortpool_cnn.pth',map_location=torch.device('cpu')))
model.eval()
pred_1 = [1]
pred_1_shuffled = [1]
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

str = ""
if q == 0:
    str = "MRI NOT REQUIRED"
elif q == 1:
    str = "MRI NEEDED NOW !!!"
else:
    str = "PREDICTION_NEEDED"

lis_1 = []
lis_2 = []
lis_3 = []

symptom_with_organ_ALL = symptom_with_organ_ALL[0]

for i in range(len(symptom_with_organ_ALL)):
    dic = symptom_with_organ_ALL[i]   #each element is literally a {key:value}
    s = ''
    for j in list(dic.keys()):
        s += j + '->' +dic[j]
    lis_1.append(s)
print(symptom_with_duration_ALL)
symptom_with_duration_ALL = symptom_with_duration_ALL[0]

for i in list(symptom_with_duration_ALL.keys()):
    s = ''
    s += i + '->' + symptom_with_duration_ALL[i]
    lis_2.append(s)

lis_3.append(str)

l = []
k = max(len(lis_1),len(lis_2),len(lis_3))

for i in range(k):
    s1 = ''
    if (i < len(lis_1)):
        s1 = lis_1[i]
    s2 = ''
    if (i < len(lis_2)):
        s2 = lis_2[i]
    s3 = ''
    if (i < len(lis_3)):
        s3 = lis_3[i]
    
    t = tuple([s1, s2, s3])
    l.append(t)

return l
    
ENDEMBED;

// output(func(med_data));

EXPORT gcn_crf() := FUNCTION

STRING med_data := 'none' : STORED('med_data');

rec2 := RECORD
STRING DO_YOU_NEED_AN_MRI;
END;

RETURN func(med_data);
END;
