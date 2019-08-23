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
print (tokenized_text)
