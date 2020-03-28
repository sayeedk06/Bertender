from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler

#commandline
import argparse

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

        # w = Label(self, text=plugin.matplotCanvas(self))
        # w.pack()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("text",type=str,help="corpus name")
    # parser.add_argument("text2",type=str,help="Second text to plot")
    parser.add_argument("--start_layer",type=int,default=8,help="starting layer number to plot")
    parser.add_argument("--end_layer",type=int,default=12,help="ending layer number to plot")
    parser.add_argument("--perplexity",type=int,default=3,help="number of nearest neighbour")

    args = parser.parse_args()
    args_dict = vars(args)
    # print(args)
    root = Root()
    root.mainloop()
