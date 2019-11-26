#GUI
from PyQt5.QtWidgets import QMainWindow, QApplication,QPushButton
import sys
import random
#matplotlibxpyqt
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
# import pyqtgraph as pg

from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging
import re
import numpy as np
#visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE
import seaborn as sns
#commandline
import argparse
#clustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


# Load BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-multilingual-cased')

labels = []
sent_dic = dict()
all_values = []
aff = []

def removing_cls_sep(text, tokens):
    a = np.array(tokens)
    b = a.flatten()
    for i in b:
        i = i[1:-1]
        print(i)
        aff.append(i)
        print("next")

    count = 0
    index = []
    for t in text:
        if t == "[CLS]":
            index.append(count)
        if t == "[SEP]":
            index.append(count)
        count += 1

    for i in sorted(index, reverse = True):
        del text[i]



def adding_special_tokens(text1):

    lines = open('data/%s.txt'% text1, encoding='utf-8').read().split('।')
    tar = []
    for line in lines:

        marked_text = " [CLS] " + line +" [SEP] "
        tar.append(marked_text)
#         print(tar)
    return tar

def index_list(tokenized_text, indexlist):
    for ind , item in enumerate(tokenized_text):

        if '#' in item:

            indexlist.append(ind)
    # print(indexlist)
    return indexlist

    #this function takes the hashed words, removes the hashes , and appends it to the previous word, enabling us to
    #see the word it got subword-ed to
def clean_list(words_list, tokens_list):
    new_list = []
    new_token_list = []
    for index, (word, token) in enumerate(zip(words_list, tokens_list)):
        print(word)
        if word.find('#') == -1:
            new_list.append(word)
            new_token_list.append(token)
        else:
            curr_len = len(new_list)

            if curr_len > 0:
                prev_word_idx = curr_len - 1
                print(new_list[prev_word_idx])
                new_list[prev_word_idx] = new_list[prev_word_idx] + word.replace('#', '')
                print(new_list[prev_word_idx])
                new_token_list[prev_word_idx] = int((new_token_list[prev_word_idx]+token)/2)

    return new_list, new_token_list


def token_embeddings_function(re_tokenized_text, encoded_layers, token_embeddings, layer_i, batch_i, token_i):

    for token_i in range(len(re_tokenized_text)):


        # Holds 12 layers of hidden states for each token
        hidden_layers = []

            # For each of the 12 layers...
        for layer_i in range(len(encoded_layers)):

                # Lookup the vector for `token_i` in `layer_i`
            vec = encoded_layers[layer_i][batch_i][token_i]
            hidden_layers.append(vec)

        token_embeddings.append(hidden_layers)
    return token_embeddings


def method(line1):

    tokenized_text = tokenizer.tokenize(line1)
    print("this is tokenized_text")
    print (tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    print("this is indexed_tokens")
    print(indexed_tokens)

    re_tokenized_text, re_tokenized_tokens = clean_list(tokenized_text, indexed_tokens)
    print("Finding the original words back for representation: ")
    print(re_tokenized_text)


    #tokenizing just the start words of any subwords, and then zipping it with the complete word
    for tup in zip(re_tokenized_text, re_tokenized_tokens):
        print (tup)


    segments_ids = [1] * len(re_tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([re_tokenized_tokens])
    segments_tensors = torch.tensor([segments_ids])

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
    token_embeddings = token_embeddings_function(re_tokenized_text, encoded_layers, token_embeddings, layer_i, batch_i, token_i)
    # For each token in the sentence...


    # Sanity check the dimensions:
    print ("Number of tokens in sequence:", len(token_embeddings))
    print ("Number of layers per token:", len(token_embeddings[0]))


    return token_embeddings, re_tokenized_text, segments_ids


def layers(token_embeddings,start_layer, end_layer):
    # global token_vecs_sum
    token_vecs_sum = []


        # Stores the token vectors, with shape [22 x 768]

        # For each token in the sentence...
    for token in token_embeddings:

        sum_vec = torch.sum(torch.stack(token)[int(start_layer):int(end_layer)], 0)
        # Use `sum_vec` to represent `token`.

        token_vecs_sum.append(sum_vec)
    print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

    return token_vecs_sum


def tsne(re_tokenized_text,arr,text, segments_ids):
    "Creates a TSNE model"
    # labels = []
    tokens = []

    for i,x in enumerate(re_tokenized_text):

        tokens.append(arr[i])
        print('tsne here:' + x)
        labels.append(x)
        print(labels)

    tsne_model = TSNE(perplexity = args.perplexity, n_components=2, init='pca', n_iter=5000, random_state=23, verbose = 2)
    new_values = tsne_model.fit_transform(tokens)
    temp_values = (new_values.tolist())
    print("TEMP")
    print(temp_values)

    print("RETOKENIZED")
    print(re_tokenized_text)

    # temp_values = another_clean(re_tokenized_text, temp_values)

    all_values.append(temp_values)

    print("New values")
    print(all_values)




    # keeping track of tokens according to sentence
    track = 0
    newlist = new_values.tolist()
    # global segments_ids
    for i in segments_ids:
        if(i == 1):
            sent_dic.setdefault(text, []).append(newlist[track])
            track = track + 1
        # keeping tracks ends here
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'PyQt5 matplotlib example - pythonspot.com'
        self.width = 640
        self.height = 400
        self.initUI()
        self. PlotCanvas()
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # m = PlotCanvas(self, width=5, height=4)
        # m.move(0,0)

        button = QPushButton('PyQt5 button', self)
        button.setToolTip('This s an example button')
        button.move(500,0)
        button.resize(140,100)
        self.PlotCanvas(FigureCanvas)
        self.show()
    def PlotCanvas(self,FigureCanvas):
        global labels, sent_dic, all_values
        for text in adding_special_tokens(args.text1):
            # method(text)
            print(text)
            token_embeddings, re_tokenized_text, segments_ids = method(text)
# print(re_tokenized_text)

            # layers(token_embeddings,args.start_layer, args.end_layer)
            token_vecs_sum = layers(token_embeddings, args.start_layer, args.end_layer)
#     for i,x in enumerate(re_tokenized_text):
#         print (i,x)

            arr = [t.numpy() for t in token_vecs_sum]

            # re_tokenized_text = another_clean(re_tokenized_text)
            tsne(re_tokenized_text,arr,text, segments_ids)

            print("RETOKENIZED")
            print(re_tokenized_text)

            # for i in re_tokenized_text:
            #     all_text.append(i)



            print("RETOKENIZED")
            print(re_tokenized_text)
            print(labels)

        # print("All text")
        # print(all_text)

        print("All Values: ")
        print(all_values)








        """plotting starts here"""

        prop = fm.FontProperties(fname='kalpurush.ttf')
        x = []
        y = []

        count = 0
        print("k")
        removing_cls_sep(labels, all_values)
        for t in aff:
            j = t
            for k in j:
                x.append(k[0])
                y.append(k[1])
        #         print(k[0])

                count = count + 1
        """DBSCAN commented"""
        # train = DBSCAN(eps=15, min_samples=2)
        flat_list = [item for sublist in aff for item in sublist]
        # train.fit(flat_list)
        np_flat_list = np.array(flat_list)
        # y_pred = train.fit_predict(np_flat_list)
        """DBSCAN commented"""
        # f = plt.figure(figsize=(16, 16))

        """DBSCAN commented"""
        f, axes = plt.subplots(nrows = 2, ncols=1)
        # axes[0].scatter(np_flat_list[:, 0], np_flat_list[:, 1],c=y_pred, cmap='Paired')
        # plt.title("DBSCAN")
        #
        # n_clusters_ = len(set(train.labels_)) - (1 if -1 in labels else 0)
        # print("Estimated number of clusters: %d" % n_clusters_)
        """DBSCAN commented"""
        """kMeans clustering begins here"""
        y_pred = KMeans(n_clusters=8, random_state=0).fit_predict(np_flat_list)
        axes[0].scatter(np_flat_list[:, 0], np_flat_list[:, 1],c=y_pred, cmap='Paired')
        n_clusters_ = len(set(y_pred)) - (1 if -1 in labels else 0)
        print("Estimated number of clusters: %d" % n_clusters_)


        """kMeans clustering ends here"""
        # f = plt.figure(figsize=(16, 16))
        sns.set(palette='bright')
        for i in range(len(labels)):
            # p = plt.scatter(np_flat_list[:, 0], np_flat_list[:, 1],c=y_pred)
            # p = plt.scatter(x[i],y[i])
            p = axes[1].scatter(np_flat_list[:, 0], np_flat_list[:, 1],c=y_pred, cmap='Paired')
            plt.annotate(labels[i],
                            xy=(x[i], y[i]),
                            xytext=(0, 0),
                            textcoords='offset points',
                            ha='right',
                           fontsize=19, fontproperties=prop)



        plt.show()



if __name__== '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("text1",type=str,help="first text to plot")
    # parser.add_argument("text2",type=str,help="Second text to plot")
    parser.add_argument("--start_layer",type=int,default=8,help="starting layer number to plot")
    parser.add_argument("--end_layer",type=int,default=12,help="ending layer number to plot")
    parser.add_argument("--perplexity",type=int,default=3,help="number of nearest neighbour")

    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window =MainWindow()
    window.show()
    app.exec()
    # root.mainloop()
