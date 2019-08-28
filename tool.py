import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging
import re
#visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE



tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


text1 = "সে ক্ষেত্রে তারা রাজনৈতিক মামলায় গ্রেপ্তার-হয়রানি বন্ধ করা, নেতারা যাতে প্রকাশ্যে আসতে পারেন।"
text2 = "প্রচার চালাতে পারেন ইত্যাদি বিষয়ে নির্বাচন কমিশনের কাছে নিশ্চয়তা চাওয়ার বিষয়ে বিএনপির কেন্দ্রীয় নেতাদের মধ্যে আলোচনা হচ্ছে।"
marked_text = "[CLS] " + text1 + " [SEP] " + text2 + " [SEP] " #adding bert special tokens
print (marked_text)

re.sub('।', '', marked_text)

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


skip_hash = clean_list(tokenized_text)
print(skip_hash)
# ends here

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
for tup in zip(re_tokenized_text, re_indexed_tokens):
    print (tup)
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

sentence1 = []
sentence2 = []

def press(event):
    print('you pressed', event.button, event.xdata, event.ydata)
    if event.xdata in sentence1 and event.ydata in sentence1:
        print("sentence 1")
    elif event.xdata in sentence2 and event.ydata in sentence2:
        print("sentence 2")




"""Creates a TSNE model and plots it"""
labels = []
tokens = []

for i,x in enumerate(tokenized_text):
    if '#' not in x and '[UNK]' not in x:

        tokens.append(arr[i])
        labels.append(x)

tsne_model = TSNE(perplexity=3, n_components=2, init='pca', n_iter=5000, random_state=23, verbose = 2)
new_values = tsne_model.fit_transform(tokens)
"""Creation of tsne model ends here"""
# keeping track of tokens according to sentence

count = 0
for i in segments_ids:
    if(i == 0):
        sentence1.append(new_values[i])
    else:
        sentence2.append(new_values[i])
# keeping tracks ends here

"""plotting starts here"""
prop = fm.FontProperties(fname='kalpurush.ttf')
x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])

fig = plt.figure(figsize=(16, 16))
for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(labels[i],
            xy=(x[i], y[i]),
            xytext=(0, 0),
            textcoords='offset points',
            ha='right',
           fontsize=19, fontproperties=prop)
fig.canvas.mpl_connect('button_press_event', press)

plt.show()
"""plotting ends here"""
