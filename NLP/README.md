<h1><b>NLP ON ELECTRONIC HEALTH RECORDS</b></h1>


This specific task involves the extraction of symptoms from a particular sentence that will work as an input for our first stage of identification - this stage extracts features to decide whether an MRI is to be performed or not. This involves training of discharge summaries of patients that are obtained from hospitals as well as PubMed dataset to obtain the following:
1. Symptoms of the patients
2. Gender
3. Affected body part
4. Duration of affected symptom

These important features, which will be the very first experienced by a patient will help the model understand the need for an MRI or not. Our research involved looking at the following models, implementing them from scratch and training it on th dataset:
1. Simple Transfer Learning on BioBert
2. COMPREHEND model for NER
3. BiLSTM - CRF model

The following are the results of the models we have tried on performing NER on our dataset for feature extraction:
| MODEL | TOTAL ACCURACY | CAPTURING REQUIRED FEATURES ACCURACY  |
|----------|----------|-------|
| COMPREHEND | 98.04 % | 73.4 % |
| BiLSTM - CRF model | 98.6 % | 74.39 % |

<br>


<h1>Decision-Making for MRI Based on Extracted Features</h1>

<h3>Description</h3>
Use the extracted medical entities to determine whether a patient requires an MRI.
The features identified include gender, symptoms, duration, and affected organs.
<h3>Data Source</h3>
The features extracted from the BiLSTM-CRF model were used to create a dataset for further analysis.
<h3>Labeling Process</h3>
The data was used to create a graph for each patient, representing the relationships between the extracted medical entities.
This whole process can be viewed in Graph_model.ipynb
<h3>Model Architecture</h3>
A Graph Attention Networks (GAT) model was employed to analyze the graphs and make MRI-related decisions.
<h3>Results</h3>
The GAT model was trained using data from patients who underwent an MRI and those who did not, effectively learning to predict the necessity of an MRI based on the extracted features.

<h2>An example of Patient graphs obtained</h2>

![image](https://github.com/user-attachments/assets/0a03163a-2e58-4b2b-bfe8-cb1dcdb425ec)

The next stage of the project is to pass these graphs created through a graph machine learning model that is doing a classification task: whether a person has to proceed for an M.R.I or not. The models were further trained on 12,000 patients data and were deployed onto the cluster.
For that purpose, the following models were used:


| MODEL | TOTAL ACCURACY| 
|----------|----------|
| SORT POOL (k most nodes) | 76.35 % |
| GLOBAL MEAN POOL | 72.68 % |



















