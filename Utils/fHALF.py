def fHALF(str):

    import random
    import re

    l = []
    l.append(str)
    osl_shuffled = l

    


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

    

    for i in range(len(final_list_of_sentences)):
        if not final_list_of_sentences[i]:
            print('Empty')
        final_list_of_sentences[i] = [sentence for sentence in final_list_of_sentences[i] if sentence]

    final_list_of_sentences

    import torch
    import torch.nn as nn
    import re
    from torchcrf import CRF
    from transformers import AutoTokenizer, AutoModel
    from transformers import BertTokenizer, BertModel, BertConfig
    from .download_utils import download_model_if_needed

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
    import os

    # Get the current file's directory (Utils directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct paths to the model files
    config_path = os.path.join(current_dir, "config.json")
    vocab_path = os.path.join(current_dir, "vocab.txt")
    model_path = os.path.join(current_dir, "pytorch_model.bin")

    # Add the Google Drive URL for your model
    GDRIVE_URL = "https://drive.google.com/uc?id=1q43z-Eo_ZE41tpJOhOIbd_WPS0ufDz6j"  # Replace with your actual Google Drive URL
    
    # Download the model if it doesn't exist
    model_path = download_model_if_needed(model_path, GDRIVE_URL)
    from torch.ao.quantization import quantize_dynamic
    config = BertConfig.from_pretrained(config_path)
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    biobert_model = BertModel.from_pretrained(model_path, config=config, local_files_only=True)
    # biobert_model = quantize_dynamic(
    #     biobert_model, {torch.nn.Linear}, dtype=torch.qint8
    # )
    # Load the trained model
    model = MOD(768, 6).to(device)  # Ensure model is on the same device
    import os
    

    # Get the current file's directory (Utils directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct path to the model file
    model_path = os.path.join(current_dir, "quantized_model_2.pth")
    model = quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    ##
    
    model.eval()

    # Initialize lists for storing results
    all_symptoms_ALL = []
    symptoms_wout_duration_ALL = []
    symptom_with_organ_ALL = []
    new_dict_ALL = []

    organs_list = []
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
                word_embedding_tensor_test = word_embeddings_test[vocab_test.index(token)]
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

                    #store the organs in a set??
                    organs_list.append(list(l5.keys()))
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



    all_symptoms_ALL = [list(set(lst)) for lst in all_symptoms_ALL]

    for i in symptoms_wout_duration_ALL:
        if 'headache' in i:
            print(True)


    hashset = set()
    for i in organs_list:
        if i:
            for j in i:
                hashset.add(j)



    import string
    import re

    # Create a set of common words to remove
    common_words = set(['a', 'and' 'an', 'the', 'is', 'are', 'was', 'were', 'of', 'in', 'to', 'for', 'and', 'or', 'but', 'with', 'on', 'at', 'as', 'by', 'for', 'with', 'as', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'as', 'until', 'while', 'as', 'until', 'while'])

    # Function to clean a single word
    def clean_word(word):
        # Remove punctuation and convert to lowercase
        word = word.lower().translate(str.maketrans('', '', string.punctuation))
        # Remove digits
        word = re.sub(r'\d+', '', word)
        return word.strip()

    # Clean the hashset
    cleaned_hashset = set()
    for item in hashset:
        # Split the item into words
        words = item.split()
        # Clean each word and keep only non-common words
        cleaned_words = [clean_word(word) for word in words if clean_word(word) and clean_word(word) not in common_words]
        # Join the cleaned words back together
        cleaned_item = ' '.join(cleaned_words)
        if cleaned_item:  # Only add non-empty items
            cleaned_hashset.add(cleaned_item)



    hashset = cleaned_hashset

    #CHECK WHETHER A SYMPTOM IS PRESENT IN THE GIVEN SENTENCE
    #IF YES: LOOK FOR ORGANS IN THE SENTENCE AND IN THE (i + 1) AND (i - 1) SENTENCES
    #JUST EXTRACT THE PAIRS FOR N0W
    symp_to_organ_tentative = []
    for i in range(len(final_list_of_sentences)):   #EACH CASE
        for t in range(len(final_list_of_sentences[i])):    #SENTENCE IN EACH CASE

            #iterate through the symptoms_wout_duration_ALL list and check for it in i:
            for j in range(len(symptoms_wout_duration_ALL)):
                symptoms_in_case = symptoms_wout_duration_ALL[j]
                for k in range(len(symptoms_in_case)):
                    #CHECK FOR THE SYMPTOM IN i
                    symptom = symptoms_in_case[k]
                    if symptom in final_list_of_sentences[i][t]:
                        #SEARCH FOR AN ORGAN in (t, t + 1 and t - 1) AND JUST EXTRACT THE MAPPINGS
                        for q in hashset:
                            if q in final_list_of_sentences[i][t]:
                                #MAKE THE SYMPTOM - ORGAN PAIRS:
                                temp = []
                                temp.append(symptom)
                                temp.append(q)
                                symp_to_organ_tentative.append(temp)

                    if t- 1 >= 0 and symptom in final_list_of_sentences[i][t - 1] :
                        #SEARCH FOR AN ORGAN in (t, t + 1 and t - 1) AND JUST EXTRACT THE MAPPINGS
                        for q in hashset:
                            if q in final_list_of_sentences[i][t - 1]:
                                #MAKE THE SYMPTOM - ORGAN PAIRS:
                                temp = []
                                temp.append(symptom)
                                temp.append(q)

                    if t + 1 < len(final_list_of_sentences[i]) and symptom in final_list_of_sentences[i][t + 1] :
                        #SEARCH FOR AN ORGAN in (t, t + 1 and t - 1) AND JUST EXTRACT THE MAPPINGS
                        for q in hashset:
                            if q in final_list_of_sentences[i][t + 1]:
                                #MAKE THE SYMPTOM - ORGAN PAIRS:
                                temp = []
                                temp.append(symptom)
                                temp.append(q)


    s = []
    for i in symp_to_organ_tentative:
        if i not in s:
            s.append(i)
    print(s)
    symp_to_organ_tentative = s

    l = []
    for i in range(len(new_dict_ALL) - 1):
        if new_dict_ALL[i] != new_dict_ALL[i + 1]:
            l.append(new_dict_ALL[i])
    
    l.append(new_dict_ALL[len(new_dict_ALL) - 1])

    symptom_with_duration_ALL = l
    symptom_with_duration_ALL

    time_units = ['seconds', 'minutes', 'hours', 'days', 'months', 'years', 'decades']

    def replace_keys(data, time_units):
        new_data = []
        for item in data:
            new_item = {}
            for key, value in item.items():
                matched_word = next((unit for unit in time_units if unit in key), None)
                if matched_word and value is not None:
                    new_item[matched_word] = value
            new_data.append(new_item)
        return new_data

    updated_data = replace_keys(symptom_with_duration_ALL, time_units)
    print(updated_data)

    symptom_with_duration_ALL = updated_data
    symptom_with_duration_ALL

    symptom_duration_map = {}
    for duration_dict in symptom_with_duration_ALL:
        for duration, symptom in duration_dict.items():
            symptom_duration_map[symptom] = duration

    # Combine the lists to create the final output
    combined_list = []
    for symptom in symptoms_wout_duration_ALL[0]:
        # Get the duration for the current symptom, or 'nil duration' if not found
        duration = 'nil duration'
        for key in symptom_duration_map:
            if key in symptom:
                duration = symptom_duration_map[key]
                break

        # Find all organs associated with the symptom, or 'nil organ' if no association is found
        found_organ = False
        for symp, organ in symp_to_organ_tentative:
            if symptom == symp:
                combined_list.append([symptom, duration, organ])
                found_organ = True

        # If no organ was found for the symptom, add 'nil organ'
        if not found_organ:
            combined_list.append([symptom, duration, 'nil organ'])


    duration_mapping = {
        'seconds': 2,
        'minutes': 3,
        'hours': 4,
        'days': 5,
        'months': 6,
        'years': 7,
        'decades': 8,
        'nil duration': 1
    }

    # Update the second element in each sublist
    for lis in combined_list:
        lis[1] = duration_mapping[lis[1]]

    # Convert each sublist to a tuple
    for i in range(len(combined_list)):
        combined_list[i] = tuple(combined_list[i])


    return combined_list
