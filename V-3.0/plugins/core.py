#For BERT pretrained model
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging
#For preprocessing and tracking
import re
import numpy as np

#For visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib.animation import FuncAnimation

#For clustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
#euclideandistance
from sklearn.neighbors import DistanceMetric

# for argparse
text = ''
perplexity = 0
start_layer = 0
end_layer = 0
# for argparse
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

sent_dic = dict()

def add_sp_token(text):

    """Adds special [CLS] and [SEP] tokens after reading each line from the corpus
    and then appends it to the line_list to be returned"""

    lines = open('data/%s.txt'% text, encoding='utf-8').read().split('।')
    line_list = []
    for line in lines:
        marked_text = " [CLS] " + line +" [SEP] "
        line_list.append(marked_text)

    return line_list
def BERT_initializer(line1):


    tokenized_text = tokenizer.tokenize(line1)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])



                                            # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
                                            # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)


    token_embedding = token_embeddings(encoded_layers)



    return token_embedding, tokenized_text, segments_ids

def token_embeddings(encoded_layers):

    # Concatenate the tensors for all 12 layers.
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(encoded_layers, dim=0)
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)

    return token_embeddings


def BERT_layers(token_embeddings,start_layer, end_layer):
    # global token_vecs_sum
    token_vecs_sum = []


        # Stores the token vectors, with shape [22 x 768]

        # For each token in the sentence...
    for token in token_embeddings:

        sum_vec = torch.sum(token[start_layer:end_layer], dim=0)
        # Use `sum_vec` to represent `token`.

        token_vecs_sum.append(sum_vec)
    # print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

    return token_vecs_sum
def tsne(tokenized_text,tensor_to_numpy,text, segments_ids):
    "Creates a TSNE model"
    tokens = []
    labels = []
    token_values = []
    for i,x in enumerate(tokenized_text):

        tokens.append(tensor_to_numpy[i])
        labels.append(x)

    tsne_model = TSNE(perplexity = perplexity, n_components=2, init='pca', n_iter=5000, random_state=23, verbose = 2)
    new_values = tsne_model.fit_transform(tokens)
    temp_values = (new_values.tolist())


    token_values.append(temp_values)


    # keeping track of tokens according to sentence
    track = 0
    newlist = new_values.tolist()
    # global segments_ids
    for i in segments_ids:
        if(i == 1):
            sent_dic.setdefault(text, []).append(newlist[track])
            track = track + 1

    return labels, token_values

def line_feed(text):
    all_label = []
    all_values = []
    for text in add_sp_token(text):
        # print(text)
        token_embeddings, tokenized_text, segments_ids = BERT_initializer(text)

        token_vecs_sum = BERT_layers(token_embeddings, start_layer, end_layer)
        tensor_to_numpy = [t.numpy() for t in token_vecs_sum]

        labels, token_values = tsne(tokenized_text,tensor_to_numpy,text, segments_ids)

        all_label.append(labels)
        all_values.append(token_values)

    return all_label, all_values

class Plugin:
    def __init__(self,*args):
        print('\t\n****Summation plugin activated****\n')
        self.text = ""
        for i in args:
            self.text = i['text']
            perplexity = i['perplexity']
            start_layer = i['start_layer']
            end_layer = i['end_layer']

    def initial(self):
        label, values = line_feed(self.text)

        return label, values
