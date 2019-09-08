# BanglaVisualization-withBert
A visualization tool that will take in one or a pair of Bangla sentences and plot the words based on the words embeddings created by BERT.

# To run the program
'''bash
python runmeforallwork.py first second
''' 

# To change input sentences
'''bash
python runmeforallwork.py text1 ৪০ শতাংশ জমির উপর মাদ্রাসাটি প্রতিষ্ঠিত। text2 বাংলা বাংলাদেশের মাতৃভাষা।
'''

# To choose layers
'''bash
python runmeforallwork.py first second --start_layer 3 --end_layer 7
''' 

# To choose the perplexity of the T-Sne plot
'''bash
python runmeforallwork.py first second --perplexity 20
'''
