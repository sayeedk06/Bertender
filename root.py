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

class Root(Tk):
    """docstring for."""

    def __init__(self):
        super(Root, self).__init__()
        self.title("Bangla Visualization with Bert")
        self.minsize(600,400)
        self. matplotCanvas()


    def matplotCanvas(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        print("\nThe given sentences are displayed below:\n")
        text1 = u"সে ক্ষেত্রে তারা রাজনৈতিক মামলায় গ্রেপ্তার-হয়রানি বন্ধ করা, নেতারা যাতে প্রকাশ্যে আসতে পারেন।"

        text2 = u"প্রচার চালাতে পারেন ইত্যাদি বিষয়ে নির্বাচন কমিশনের কাছে নিশ্চয়তা চাওয়ার বিষয়ে বিএনপির কেন্দ্রীয় নেতাদের মধ্যে আলোচনা হচ্ছে।"

        marked_text = "[CLS] " + text1 + " [SEP] " + text2 + " [SEP] " #adding bert special tokens
        print (marked_text)

        marked_text = re.sub('।', '', marked_text)

        tokenized_text = tokenizer.tokenize(marked_text)
        # print (tokenized_text)
        # keeping track of hashed tokens
        indexlist = []
        for ind , item in enumerate(tokenized_text):
            if '#' in item:
                indexlist.append(ind)
        print(indexlist)
        # ends here

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

        re_tokenized_text = clean_list(tokenized_text)
        print(re_tokenized_text)
        # ends here
        print("Indexed tokens are: ")
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        re_indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        print(re_indexed_tokens)
        print(len(re_indexed_tokens))

        print("Tokenizing the original form, includes the hash words")
        for tup in zip(tokenized_text, indexed_tokens):
            print (tup)

        for index in sorted(indexlist, reverse = True):
            del re_indexed_tokens[index]
        # print(re_indexed_tokens)
        print("Tokens after removing hashed words")
        for tup in zip(re_tokenized_text, re_indexed_tokens):
            print (tup)



        def addSentenceId(token_text):
            count = 0
            for i in token_text:
                count = count + 1
                if i == '[SEP]':
                    break

            segments_ids1 = [0] * count
            segments_ids2 = [1] * (len(token_text) - count)
            segments_ids = segments_ids1 + segments_ids2
            print(segments_ids)
            # segments_ids = [1] * len(tokenized_text)
            # print (segments_ids)
            return segments_ids

        segments_ids = addSentenceId(re_tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([re_indexed_tokens])
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
        for token_i in range(len(re_tokenized_text)):

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


        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum = []

        # For each token in the sentence...
        for token in token_embeddings:
            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(torch.stack(token)[-4:], 0)

            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)

        print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))


        arr = [t.numpy() for t in token_vecs_sum]

# mouse click event starts here
        sent_dic = dict()
        def on_click(event):
            print('you pressed', event.button, event.xdata, event.ydata)
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
        "Creates and TSNE model and plots it"
        labels = []
        tokens = []

        for i,x in enumerate(re_tokenized_text):
            # if '#' not in x and '[UNK]' not in x:

            tokens.append(arr[i])
            labels.append(x)

        tsne_model = TSNE(perplexity=3, n_components=2, init='pca', n_iter=5000, random_state=23, verbose = 2)
        new_values = tsne_model.fit_transform(tokens)
        # keeping track of tokens according to sentence

        track = 0
        newlist = new_values.tolist()
        for i in segments_ids:
            if(i == 0):
                sent_dic.setdefault(text1, []).append(newlist[track])
                track = track + 1
            else:
                sent_dic.setdefault(text2, []).append(newlist[track])
                track = track + 1
        # keeping tracks ends here

        """plotting starts here"""
        prop = fm.FontProperties(fname='kalpurush.ttf')
        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        f = plt.figure(figsize=(16, 16))
        sns.set(palette='bright')
        for i in range(len(x)):
            p = plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                    xy=(x[i], y[i]),
                    xytext=(0, 0),
                    textcoords='offset points',
                    ha='right',
                   fontsize=19, fontproperties=prop)

        canvas = FigureCanvasTkAgg(f, self)
        f.canvas.mpl_connect('button_press_event', on_click)
        f.canvas.mpl_connect('button_release_event', off_click)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        toolbar.pack()
        canvas.get_tk_widget().pack(side = TOP, fill = BOTH, expand = True)

        # canvas.get_tk_widget().pack(side= BOTTOM, fill= BOTH, expand= True)

        """plotting ends here"""



root = Root()
root.mainloop()
