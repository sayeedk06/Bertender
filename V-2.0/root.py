from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging
import re
#visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE
import seaborn as sns
#commandline
import argparse



tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-multilingual-cased')



labels = []

sent_dic = dict()

all_values = []
def adding_special_tokens(text1):

    lines = open('data/%s.txt'% text1, encoding='utf-8').read().split('।')
    tar = []
    for line in lines:
        line re.sub('[()]', '', line) #brackets removed
        marked_text = " [CLS] " + line +" [SEP] "
        tar.append(marked_text)
#         print(tar)
    return tar

def index_list(tokenized_text, indexlist):
    for ind , item in enumerate(tokenized_text):

        if '#' in item:

            indexlist.append(ind)

    return indexlist
    #this function takes the hashed words, removes the hashes , and appends it to the previous word, enabling us to
    #see the word it got subword-ed to
def clean_list(words_list):
    new_list = []

    for idx, word in enumerate(words_list):
        # no # sign
        if word.find('#') == -1:
            new_list.append(word)
        # there is a # sign
        else:
            curr_len = len(new_list)

            if curr_len > 0:
                prev_word_idx = curr_len - 1
                new_list[prev_word_idx] = new_list[prev_word_idx] + word.replace('#', '')

    return new_list

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

    indexlist = []

    indexlist = index_list(tokenized_text, indexlist)


    #re_tokenized_text is only the start words, words that do not have hash in them.
    re_tokenized_text = [l for l in tokenized_text if '#' not in l]
    # print("first RETOKENIZED TEXT")
    # print(re_tokenized_text)



    print("When the words get tokenized: ")
    print(tokenized_text)
    re_tokenized_text = clean_list(tokenized_text)
    print("Finding the original words back for representation: ")
    print(re_tokenized_text)


    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    re_indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    print(re_indexed_tokens)
    print(len(re_indexed_tokens))
    print("Tokenizing the original form, includes the hash words")
    for tup in zip(tokenized_text, indexed_tokens):
        print (tup)


    for index in sorted(indexlist, reverse = True):
        del re_indexed_tokens[index]
    print(re_indexed_tokens)

    #tokenizing just the start words of any subwords, and then zipping it with the complete word
    for tup in zip(re_tokenized_text, re_indexed_tokens):
        print (tup)



    segments_ids = [1] * len(re_tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([re_indexed_tokens])
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
    # for token_i in range(len(re_tokenized_text)):



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

    tokens = []


    for i,x in enumerate(re_tokenized_text):


        # if '#' not in x and '[UNK]' not in x:

        tokens.append(arr[i])
        print('tsne here:' + x)
        labels.append(x)
        print(labels)


    tsne_model = TSNE(perplexity = args.perplexity, n_components=2, init='pca', n_iter=5000, random_state=23, verbose = 2)
    new_values = tsne_model.fit_transform(tokens)
    all_values.append(new_values.tolist())


    # keeping track of tokens according to sentence
    track = 0
    newlist = new_values.tolist()
    # global segments_ids
    for i in segments_ids:
        if(i == 1):
            sent_dic.setdefault(text, []).append(newlist[track])
            track = track + 1
        # keeping tracks ends here

class Root(Tk):
    """docstring for."""

    def __init__(self):
        super(Root, self).__init__()
        self.title("Bangla Visualization with Bert")
        self.minsize(600,400)
        self. matplotCanvas()



    def matplotCanvas(self):
        # global arr
        global labels, sent_dic, all_values
        for text in adding_special_tokens(args.text1):
            # method(text)
            print(text)
            token_embeddings, re_tokenized_text, segments_ids = method(text)

            token_vecs_sum = layers(token_embeddings, args.start_layer, args.end_layer)


            arr = [t.numpy() for t in token_vecs_sum]
            tsne(re_tokenized_text,arr,text, segments_ids)


            print("RETOKENIZED")
            print(re_tokenized_text)
            print(labels)

# mouse click event starts here

        def on_click(event):
            print('you pressed', event.button, event.xdata, event.ydata)
            self.check = 5
            axes = plt.gca()
            left, right = axes.get_xlim()
            if(left< 0 and right<=0):
                if(abs(left)-abs(right)) < 500 and (abs(left)-abs(right)) > 200:
                    self.check = 4
                elif(abs(left)-abs(right)) < 200 and (abs(left)-abs(right)) > 100:
                    self.check = 2
                elif(abs(left)-abs(right)) < 100:
                    self.check = 1
            elif(left >= 0 and right >0):
                if(abs(left)-abs(right)) < 500 and (abs(left)-abs(right)) > 200:
                    self.check = 4
                elif(abs(left)-abs(right)) < 200 and (abs(left)-abs(right)) > 100:
                    self.check = 2
                elif(abs(left)-abs(right)) < 100:
                    self.check = 1
            elif(left < 0 and right >=0):
                # if(300 < (abs(left)- right) < 400):
                #     self.check = 4
                if(200 < (abs(left)-right) < 300):
                    self.check = 3
                elif(100 < (abs(left)-right) < 200):
                    self.check = 2
                elif((abs(left)-right) < 100):
                    self.check = 0.5
            xpos, ypos = event.xdata, event.ydata
            for key in sent_dic:
                for values in sent_dic[key]:
                    if abs(xpos - values[0]) < self.check and abs(ypos - values[1]) < self.check:
                        print(key)
                        self.text_show = plt.text(event.xdata, event.ydata, key, fontsize=5, fontproperties=prop)
                        canvas.draw()
                        # key_press_handler(event, canvas, toolbar)

        def off_click(event):
            self.text_show.remove()
            canvas.draw()
# mouse click event ends here



        """plotting starts here"""

        prop = fm.FontProperties(fname='kalpurush.ttf')
        x = []
        y = []


        print("k")
        for t in all_values:
            j = t
            for k in j:
                x.append(k[0])
                y.append(k[1])




        f = plt.figure(figsize=(16, 16))
        sns.set(palette='bright')
        for i in range(len(labels)):
            p = plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                            xy=(x[i], y[i]),
                            xytext=(0, 0),
                            textcoords='offset points',
                            ha='right',
                           fontsize=19, fontproperties=prop)


        # self.check = 5
        # axes = plt.gca()
        # left, right = axes.get_xlim()
        # if(left< 0 and right<=0):
        #     if(abs(left)-abs(right)) < 500 and (abs(left)-abs(right)) > 200:
        #         self.check = 4
        #     elif(abs(left)-abs(right)) < 200 and (abs(left)-abs(right)) > 100:
        #         self.check = 2
        #     elif(abs(left)-abs(right)) < 100:
        #         self.check = 1
        # elif(left >= 0 and right >0):
        #     if(abs(left)-abs(right)) < 500 and (abs(left)-abs(right)) > 200:
        #         self.check = 4
        #     elif(abs(left)-abs(right)) < 200 and (abs(left)-abs(right)) > 100:
        #         self.check = 2
        #     elif(abs(left)-abs(right)) < 100:
        #         self.check = 1
        # elif(left < 0 and right > 0):
        #     if(left + right) < 500 and (left+right) > 200:
        #         self.check = 4
        #     elif(left+right) < 200 and (left+right) > 100:
        #         self.check = 2
        #     elif(left+right) < 100:
        #         self.check = 1
        # print(left)
        # print(right)
        canvas = FigureCanvasTkAgg(f, self)
        f.canvas.mpl_connect('button_press_event', on_click)
        f.canvas.mpl_connect('button_release_event', off_click)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        toolbar.pack()
        canvas.get_tk_widget().pack(side = TOP, fill = BOTH, expand = True)



        """plotting ends here"""

if __name__== '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("text1",type=str,help="first text to plot")
    # parser.add_argument("text2",type=str,help="Second text to plot")
    parser.add_argument("--start_layer",type=int,default=8,help="starting layer number to plot")
    parser.add_argument("--end_layer",type=int,default=12,help="ending layer number to plot")
    parser.add_argument("--perplexity",type=int,default=3,help="number of nearest neighbour")

    args = parser.parse_args()

    root = Root()
    root.mainloop()
