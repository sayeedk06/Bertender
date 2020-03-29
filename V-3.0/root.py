from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
import numpy as np
#commandline
import argparse
#visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

#plugin architecture
import importlib


PLUGIN_NAME = "plugins.core"
CLUSTER_PLUGIN = "plugins.core2_cluster"
plugin_module = importlib.import_module(PLUGIN_NAME, '.')
plugin_module_2 = importlib.import_module(CLUSTER_PLUGIN, '.')



class Root(Tk):

    def __init__(self):
        super(Root,self).__init__()
        self.title("BERTENDER")
        self.minsize(800,600)
        self.main_exec()


    def main_exec(self):
        plugin = plugin_module.Plugin(args_dict)
        plugin2 = plugin_module_2.Plugin()
        print(plugin_module)

        label, values, sent_dic = plugin.initial()

        def on_click(event):
            # print('you pressed', event.button, event.xdata, event.ydata)
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
                        # print(key)
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
        for token in values:
            for temp in token:
                x.append(temp[0])
                y.append(temp[1])


        flat_list = [item for sublist in values for item in sublist]
        np_flat_list = np.array(flat_list)

        f, axes = plt.subplots(nrows = 2, ncols=1)

        y_pred = plugin2.initial(np_flat_list,label)

        axes[0].scatter(np_flat_list[:, 0], np_flat_list[:, 1],c=y_pred, cmap='Paired')

        for i in range(len(label)):

            p = axes[1].scatter(np_flat_list[:, 0], np_flat_list[:, 1],c=y_pred, cmap='Paired')
            plt.annotate(label[i],
                            xy=(x[i], y[i]),
                            xytext=(0, 0),
                            textcoords='offset points',
                            ha='right',
                           fontsize=19, fontproperties=prop)


        """UI work in tkinter and integration of tkinter
         with matplotlib"""
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
