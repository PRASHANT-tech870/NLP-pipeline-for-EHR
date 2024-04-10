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
