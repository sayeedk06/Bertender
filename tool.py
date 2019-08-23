import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging
#visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


text = "প্রায় ১৬৭ বছরের পুরোনো এই বিদ্যালয়টি বিশ্ববিদ্যালয়ের অংশ হলে এর ঐতিহ্য বিপন্ন হবে বলে তাদের আশঙ্কা।"
marked_text = "[CLS] " + text + " [SEP]"
print (marked_text)

re.sub('।', '', marked_text)

tokenized_text = tokenizer.tokenize(marked_text)
# print (tokenized_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

segments_ids = [1] * len(tokenized_text)
# print (segments_ids)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()



# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)


print ("Number of layers:", len(encoded_layers))
layer_i = 0

print ("Number of batches:", len(encoded_layers[layer_i]))
batch_i = 0

print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))







# Convert the hidden state embeddings into single token vectors

# Holds the list of 12 layer embeddings for each token
# Will have the shape: [# tokens, # layers, # features]
token_embeddings = []

# For each token in the sentence...
for token_i in range(len(tokenized_text)):

  # Holds 12 layers of hidden states for each token
  hidden_layers = []

  # For each of the 12 layers...
  for layer_i in range(len(encoded_layers)):

    # Lookup the vector for `token_i` in `layer_i`
    vec = encoded_layers[layer_i][batch_i][token_i]

    hidden_layers.append(vec)

  token_embeddings.append(hidden_layers)

# Sanity check the dimensions:
print ("Number of tokens in sequence:", len(token_embeddings))
print ("Number of layers per token:", len(token_embeddings[0]))
