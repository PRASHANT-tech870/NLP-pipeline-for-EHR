IMPORT PYTHON3 AS PY;
SET OF INTEGER  list := [1,2,4,5];

rec := RECORD
    STRING SYMPTOMS;
    STRING DURATION_OF_PERSISTANCE;
    STRING GENDER;  
    STRING ORGAN;
END;

STRING med := 'A 63-year-old man was having cough and admitted to our Respiratory Disease Unit at the University Hospital?ncona, for a 6-month exertional dyspnea and bilateral pleural effusion prevalent on the ride side, detected on chest computed tomography (CT).\nHe was former smoker without occupational exposure to asbestos. His medical history was remarkable for asymptomatic brain aneurysm, blood hypertension, multiple lumbar disc herniation. On admission to our unit, physical examination, oxygen saturation on room air, heart rate and blood pressure were normal, whilst breathing sound was suppressed at the third right lower lung fields.\nThe patient first underwent a repeated CT scan that allowed us to rule out a pulmonary embolism and confirmed moderate right pleural effusion with parietal and visceral pleural thickening, in the absence of significant parenchymal abnormalities (). Thoracic ultrasound (TUS) revealed hyperechogenic pleural fluid with atelectasis of basal segments of the right lower lobe (); at thoracentesis, fluid appeared cloudy and yellow coloured, and a physico-chemical exam was consistent with exudate and microbiological tests, including an acid-alcohol-fast bacilli (AAFB) search, were negative ().\nA subsequent medical thoracoscopy (MT) revealed the presence of yellow pleural fluid (overall 1800 mL removed) and parietal pleura hyperemia with fibrotic plaques (). Ten pleural biopsies were obtained by forceps on parietal pleura and histopathological examination documented a large lymphoplasmacytic infiltration, fibrosis, reactive mesothelial cells and vascular proliferation, in absence of neoplastic lesions or granulomas; the final diagnosis was suggestive for non-specific pleuritis (NSP).';
STREAMED DATASET(rec) f(STRING medical_string) := EMBED(PY)
import re
tsl = []
tsl.append(medical_string)

l = []
for i in range(len(tsl)):
    t = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', tsl[i])
    s = ""
    for j in range(len(t)):
        s += t[j]
        if 'MRI' in t[j]:
            break
    l.append(s)
    tsl[i] = s



tsl_tokenized = []
for sentence in tsl:
    # Tokenize each sentence
    tokens = re.findall(r'\b[\w-]+\b', sentence.lower())
    tsl_tokenized.append(tokens)
vocab_test = []
for i in range(len(tsl_tokenized)):
    for j in range(len(tsl_tokenized[i])):
        if (tsl_tokenized[i][j] not in vocab_test):
            vocab_test.append(tsl_tokenized[i][j])

#BIOBERT EMBEDDINGS GENERATED HERE FOR THE SENTENCE TO BE TESTED:
#TESTING CODE -> THE TRAINING CODE IS IN THE NEXT CELL

from transformers import AutoTokenizer, AutoModel
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BioBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1").to(device)

# List of wordsfrom
word_list_test = vocab_test # Replace with your list of words

word_embeddings = []

# Iterate through each word
for word in word_list_test:
    # Tokenize word

    tokens = tokenizer(word, return_tensors="pt").to(device)
    
    # Pass tokens through BioBERT model
    with torch.no_grad():
        outputs = model(**tokens)
    
    # Get embeddings for the first token (CLS token)
    embedding = outputs.last_hidden_state[0][0]  # Assuming you want to use the CLS token embedding
    
    # Append embedding to list
    word_embeddings.append(embedding)

# Convert list of embeddings to tensor
word_embeddings_test = torch.stack(word_embeddings).to(device)

sentence_embeddings_test = []
# LET US FIND SENTENCE EMBEDDINGS
for i in range(len(tsl_tokenized)):
    each_word_embeddings_test = []
    for j in range(len(tsl_tokenized[i])):
        # Create tensor and then move it to device
        word_embedding_tensor_test = torch.tensor(word_embeddings_test[word_list_test.index(tsl_tokenized[i][j])])
        each_word_embeddings_test.append(word_embedding_tensor_test.to(device))
    sentence_embeddings_test.append(torch.stack(each_word_embeddings_test))    


sentence_embeddings_test = torch.stack(sentence_embeddings_test).to(device)


import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.input_size = 768
        self.hidden_size = 256
        self.num_layers = 2
        self.dropout = 0.3
        self.bilstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        
    def forward(self, embeddings):
        lstm_out, _ = self.bilstm(embeddings)
        return lstm_out

t = BiLSTM()


class MOD(nn.Module):
    def __init__(self, input_size, num_labels):
        super(MOD, self).__init__()
        self.bilstm = BiLSTM()
        self.linear = nn.Linear(256, num_labels)  
        self.crf = CRF(num_labels)

    def forward(self, x):
        lstm_output = self.bilstm(x)
        embeddings = self.linear(lstm_output)  
        return (embeddings)

model = MOD(768, 6) 
model.load_state_dict(torch.load('/var/lib/HPCCSystems/mydropzone/best_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

input_tensor = (sentence_embeddings_test)  

with torch.no_grad():
    output = model(input_tensor)
    pred = model.crf.decode(output)
print(' THE LIST: ', (tsl_tokenized))
print('SIZE OF THE LIST:n', len(tsl_tokenized[0]))
print(len(pred))
print(sentence_embeddings_test.shape)   
'''
for k in range(len(pred)):
	if (k != 0 and pred[k-1][0] == 5 and pred[k][0] == 1):
		print('1: ',tsl_tokenized[0][k-1], tsl_tokenized[0][k])
		k += 1
	elif (pred[k][0] == 2):
		print('2: ',tsl_tokenized[0][k])
	elif (k != 0 and pred[k-1][0] == 5 and pred[k][0] == 5):
		print('5: ',tsl_tokenized[0][k-1], tsl_tokenized[0][k])
		k += 1
	elif (pred[k][0] == 1):
		print('1: ',tsl_tokenized[0][k])
	elif(k != len(pred) - 1 and pred[k][0] == 5 and pred[k+1][0] == 5):
		c	
	elif(pred[k][0] == 5):
		print('5: ',tsl_tokenized[0][k])
	if (pred[k][0] == 4):
		print('4: ',tsl_tokenized[0][k])
'''
s = ''
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
#TUPLE SHOULD BE OF THE FORM: (symptom, time of persistance, gender, organs affected)	
for k in range(len(pred)):
	if (pred[k][0] == 5):
		s += tsl_tokenized[0][k] + ' '
	elif (k != 0 and pred[k][0] == 0):
		t = pred[k-1][0]
		if t != 0:
			if t == 1:
				l1.append(s)
			if t == 2:
				l1.append(s)
			if t == 3:
				l3.append(s)
			if t == 4:
				l4.append(s)
			if t == 5:
				l5.append(s)
			print(t,':', s)
			s = ''
		s = ''
	if (pred[k][0] == 1):
		s += tsl_tokenized[0][k] + ' '
	elif (pred[k][0] == 4):
		s += tsl_tokenized[0][k] + ' '
if (pred[len(pred)-1][0] != 0):
	print(pred[len(pred)-1][0], ':', tsl_tokenized[0][len(pred)-1])	

z = max(len(l1),len(l3), len(l4), len(l5))
lis = []
for o in range(z):
	if o <= len(l1)-1:
		z1 = l1[o]
	else:
		z1 = '-'
	if o <= len(l3)-1:
		z3 = l3[o]
	else:
		z3 = '-'
	if o <= len(l4)-1:
		z4 = l4[o]
	else:
		z4 = '-'
	if o <= len(l5)-1:
		z5 = l5[o]
	else:
		z5 = '-'
	lis.append((z1, z3, z4, z5))

return lis

ENDEMBED;



EXPORT lets_try() := FUNCTION

  STRING med_data := 'none' : STORED('med_data');
  
  op := f(med_data);
  rec2 := RECORD
    STRING SYMPTOMS :=  op.SYMPTOMS;
    STRING DURATION_OF_PERSISTANCE := op.DURATION_OF_PERSISTANCE;
    STRING GENDER := op.GENDER;  
    STRING ORGAN := op.ORGAN;
  END;
  RETURN OUTPUT(op, rec2);
END;