from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
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
from matplotlib.animation import FuncAnimation
#commandline
import argparse
#clustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
#euclideandistance
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('euclidean')


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-multilingual-cased')



# start_layer = 0
# end_layer = 0
labels = []
# token_embeddings = []
# re_tokenized_text = []
# segments_ids = []
sent_dic = dict()
# token_vecs_sum = []
# tar = []
# indexlist = []
all_values = []
aff = []

#Dictionary of the euclidean distances starts here
def dictionaryofeuclideandistanceandtheircoordinates(label,x,y,word):
    indexforelist = []
    countfore = 0

    for i in label:
        if i == word:
            indexforelist.append(countfore)
        countfore +=1

    mappingxco_ordinates = []
    mappingyco_ordinates = []

    for i in indexforelist:
        mappingxco_ordinates.append(x[i])
        mappingyco_ordinates.append(y[i])

    lengthofspecifiedwordfindings = len(mappingxco_ordinates)

    xypairedlist = []

    for i in range (0,lengthofspecifiedwordfindings):
        k = []
        k.append(mappingxco_ordinates[i])
        k.append(mappingyco_ordinates[i])
        xypairedlist.append(k)

    a = dist.pairwise(xypairedlist)

    count = 0
    distance_list = []

    for i in a:
        for j in i:
            count += 1
            distance_list.append(j)

    listco = [] #a list in the order the distances are shown

    for i in xypairedlist:
        for j in xypairedlist:
            listco.append(i)
            listco.append(j)

    dictionaryofdistances = dict()

    for index, item in enumerate(distance_list):
        target_start_index = 2 * index
        target_end_index = 2 * index + 1

        dictionaryofdistances[item] = listco[target_start_index:(target_end_index + 1)]



    return dictionaryofdistances, distance_list
#Dictionary of the euclidean distances ends here


#Plotting the words outside a certain boundary starts here

def plottingthe_words_outside_a_boundary(dictionary, distance_list, boundary):


    thetwocoordinates = []

    for i in dictionary:
        if i > boundary:

            thetwocoordinates.append(dictionary[i])

    therestx = []
    theresty = []

    for i in thetwocoordinates:
        for j in i:
            therestx.append(j[0])
            theresty.append(j[1])

    # fig, axes1 = plt.subplots()

    # axes1.scatter(therestx, theresty, cmap='Paired')
    # axes1.set_title('Points of the word outside the set boundary')

    return therestx, theresty

#Plotting the words outside a certain boundary ends here


#Finding word instances starts here
def plottingdesiredword(label,x,y,word):
    print("Initial:\n")
    print(x)
    print(y)
    indexforelist = []
    countfore = 0
    for i in label:
        if i == word:
            indexforelist.append(countfore)
        countfore +=1
    print(indexforelist)
    mappingxco_ordinates = []
    mappingyco_ordinates = []

    for i in indexforelist:
        mappingxco_ordinates.append(x[i])
        mappingyco_ordinates.append(y[i])

    print("function")
    print(mappingxco_ordinates)
    print(mappingyco_ordinates)
    return mappingxco_ordinates , mappingyco_ordinates
#Finding word instances ends here

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


    # global text1,token_embeddings, re_tokenized_text, segments_ids
    # text1 = read_text(line1)
    # marked_text = "[CLS] " + text1 + " [SEP] " + text2 + " [SEP] " #adding bert special tokens
    # print(text1)


    tokenized_text = tokenizer.tokenize(line1)
    print("this is tokenized_text")
    print (tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    print("this is indexed_tokens")
    print(indexed_tokens)

    # indexlist = []

    # indexlist = index_list(tokenized_text, indexlist)


    # #re_tokenized_text is only the start words, words that do not have hash in them.
    # re_tokenized_text = [l for l in tokenized_text if '#' not in l]
    # print("first RETOKENIZED TEXT")
    # print(re_tokenized_text)



    # print("When the words get tokenized: ")
    # print(tokenized_text)
    re_tokenized_text, re_tokenized_tokens = clean_list(tokenized_text, indexed_tokens)
    print("Finding the original words back for representation: ")
    print(re_tokenized_text)


    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # re_indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # print(re_indexed_tokens)
    # print(len(re_indexed_tokens))
    # print("Tokenizing the original form, includes the hash words")
    # for tup in zip(tokenized_text, indexed_tokens):
    #     print (tup)


    # for index in sorted(indexlist, reverse = True):
    #     del re_indexed_tokens[index]
    # print(re_indexed_tokens)

    #tokenizing just the start words of any subwords, and then zipping it with the complete word
    for tup in zip(re_tokenized_text, re_tokenized_tokens):
        print (tup)

    # def addSentenceId(token_text):
    #     count = 0
    #     for i in token_text:
    #         count = count + 1
    #         if i == '[SEP]':
    #             break
    #
    #     segments_ids1 = [0] * count
    #     segments_ids2 = [1] * (len(token_text) - count)
    #     segments_ids = segments_ids1 + segments_ids2
    #     print(segments_ids)
    #     # segments_ids = [1] * len(tokenized_text)
    #     # print (segments_ids)
    #     return segments_ids

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
    # for token_i in range(len(re_tokenized_text)):

    # Holds 12 layers of hidden states for each token
        # hidden_layers = []

        # For each of the 12 layers...
        # for layer_i in range(len(encoded_layers)):

            # Lookup the vector for `token_i` in `layer_i`
        #     vec = encoded_layers[layer_i][batch_i][token_i]
        #
        #     hidden_layers.append(vec)
        #
        # token_embeddings.append(hidden_layers)

    # Sanity check the dimensions:
    print ("Number of tokens in sequence:", len(token_embeddings))
    print ("Number of layers per token:", len(token_embeddings[0]))



    # print("Choose the layers: ")
    # start_layer = input("What should be the starting layer: ")
    # end_layer = input("What should be the end layer: ")

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
#     global labels, all_values, sent_dic

    for i,x in enumerate(re_tokenized_text):


        # if '#' not in x and '[UNK]' not in x:

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




# mouse click event starts here
        # global sent_dic = dict()
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
                    if abs(xpos - values[0]) < 5 and abs(ypos - values[1]) < 5:
                        print(key)
                        self.text_show = plt.text(event.xdata, event.ydata, key, fontsize=5, fontproperties=prop)
                        canvas.draw()
                        # key_press_handler(event, canvas, toolbar)

        def off_click(event):
            self.text_show.remove()
            canvas.draw()
# mouse click event ends here

        "Menu bar functionality starts here"
        def new_window(tuplex,tupley):

            def getboundary(event):
                # print(slider.get())
                boundary = slider.get()
                therestx, theresty = plottingthe_words_outside_a_boundary(dictionary, distance_list, boundary)
                print("x co ordinates outside the boundary")
                print(therestx)
                print("y co ordinates outside the boundary")
                print(theresty)
                # axes1.scatter(therestx,theresty, cmap='Paired')
                axes1.clear()
                axes1.scatter(therestx,theresty, cmap='Paired')
                word_canvas.draw()


            mapx,mapy = plottingdesiredword(labels, tuplex, tupley, text_input.get())
            dictionary, distance_list = dictionaryofeuclideandistanceandtheircoordinates(labels, tuplex, tupley, text_input.get())
            maximumdistance = max(distance_list)

            window = Toplevel(self)
            window.minsize(width=800, height=500)
            fig, axes1 = plt.subplots()


            axes1.scatter(mapx, mapy, cmap='Paired')

            word_canvas = FigureCanvasTkAgg(fig, window)
            plot_widget = word_canvas.get_tk_widget()
            plot_widget.pack(side = TOP, fill = BOTH, expand = True)


            # var = DoubleVar()
            help = Label(window, text="Slide to set boundary",font=("Helvetica", 16))
            help.pack()
            slider = Scale(window,from_=0, to=maximumdistance, orient=HORIZONTAL,command=getboundary)
            slider.pack(fill = BOTH)
            # print(var.get())
            word_canvas.draw()

        "Menu bar functionality ends here"


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

        """
        Euclidean distance measurement starts here
        """
        # tuplex = tuple(x)
        # tupley = tuple(y)


        """
        Euclidian distance measurement ends here
        """

        canvas = FigureCanvasTkAgg(f, self)
        f.canvas.mpl_connect('button_press_event', on_click)
        f.canvas.mpl_connect('button_release_event', off_click)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        toolbar.pack()

        canvas.get_tk_widget().pack(side = BOTTOM, fill = BOTH, expand = True)

        text_input = Entry(self)
        text_input.pack(side = LEFT)
        input_button=Button(self, height=1, width=10, text="Find", command=lambda: new_window(x,y))
        input_button.pack(side = LEFT)
        # canvas.get_tk_widget().pack(side= BOTTOM, fill= BOTH, expand= True)

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