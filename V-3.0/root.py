from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
import numpy as np
#commandline
import argparse
#visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
#clusters
from sklearn.cluster import KMeans
#plugin architecture
import importlib


PLUGIN_NAME = "plugins.core"
plugin_module = importlib.import_module(PLUGIN_NAME, '.')



class Root(Tk):

    def __init__(self):
        super(Root,self).__init__()
        self.title("BERTENDER")
        self.minsize(800,600)

        print(plugin_module)
        plugin = plugin_module.Plugin(args_dict)

        label, values = plugin.initial()
        # print(values)

        """plotting starts here"""

        prop = fm.FontProperties(fname='kalpurush.ttf')
        x = []
        y = []

        count = 0
        for token in values:
            # j = t
            for temp in token:
                x.append(temp[0])
                y.append(temp[1])
                # print(k[0])

                count = count + 1

        flat_list = [item for sublist in values for item in sublist]
        np_flat_list = np.array(flat_list)

        f, axes = plt.subplots(nrows = 2, ncols=1)
        y_pred = KMeans(n_clusters=8, random_state=0).fit_predict(np_flat_list)
        axes[0].scatter(np_flat_list[:, 0], np_flat_list[:, 1],c=y_pred, cmap='Paired')
        n_clusters_ = len(set(y_pred)) - (1 if -1 in label else 0)
        for i in range(len(label)):

            p = axes[1].scatter(np_flat_list[:, 0], np_flat_list[:, 1],c=y_pred, cmap='Paired')
            plt.annotate(label[i],
                            xy=(x[i], y[i]),
                            xytext=(0, 0),
                            textcoords='offset points',
                            ha='right',
                           fontsize=19, fontproperties=prop)

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        toolbar.pack()

        canvas.get_tk_widget().pack(side = BOTTOM, fill = BOTH, expand = True)

        text_input = Entry(self)
        text_input.pack(side = LEFT)
        input_button=Button(self, height=1, width=10, text="Find", command=lambda: new_window(x,y,text_input))
        input_button.pack(side = LEFT)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("text",type=str,help="corpus name")
    parser.add_argument("--start_layer",type=int,default=8,help="starting layer number to plot")
    parser.add_argument("--end_layer",type=int,default=12,help="ending layer number to plot")
    parser.add_argument("--perplexity",type=int,default=3,help="number of nearest neighbour")

    args = parser.parse_args()
    args_dict = vars(args)
    # print(args)
    root = Root()
    root.mainloop()
