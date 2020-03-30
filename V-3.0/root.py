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
#euclideandistance
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('euclidean')

PLUGIN_NAME = "plugins.core_3"
CLUSTER_PLUGIN = "plugins.core2_cluster"
plugin_module = importlib.import_module(PLUGIN_NAME, '.')
plugin_module_2 = importlib.import_module(CLUSTER_PLUGIN, '.')


#Finding the remaining sentences starts here

def source_of_words(the_remaining_coordinates, sentence_dictionary):
    sentences = []
    for key, value in sentence_dictionary.items():
        for i in value:
            for j in the_remaining_coordinates:
                if i == j:
                    sentences.append(key)
    return sentences

#Finding the remaining sentences ends here

#Finding word instances starts here
def word_plot(label,x,y,word):
    # print("Initial:\n")
    # print(x)
    # print(y)
    indexforelist = []
    countfore = 0
    for i in label:
        if i == word:
            indexforelist.append(countfore)
        countfore +=1
    # print(indexforelist)
    mappingxco_ordinates = []
    mappingyco_ordinates = []

    for i in indexforelist:
        mappingxco_ordinates.append(x[i])
        mappingyco_ordinates.append(y[i])

    # print("function")
    # print(mappingxco_ordinates)
    # print(mappingyco_ordinates)
    return mappingxco_ordinates , mappingyco_ordinates
#Finding word instances ends here



#Dictionary of the euclidean distances starts here
def distance_x_y(label,x,y,word):
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

    length_required = len(mappingxco_ordinates)

    xypairedlist = []

    for i in range (0,length_required):
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

    distance_dictionary = dict()

    for index, item in enumerate(distance_list):
        target_start_index = 2 * index
        target_end_index = 2 * index + 1

        distance_dictionary[item] = listco[target_start_index:(target_end_index + 1)]



    return distance_dictionary, distance_list
#Dictionary of the euclidean distances ends here

#Plotting the words outside a certain boundary starts here
def boundary_word_plot(dictionary, distance_list, boundary):

    the_two_boundaries = []

    for i in dictionary:
        if i > boundary:

            the_two_boundaries.append(dictionary[i])

    therestx = []
    theresty = []

    for i in the_two_boundaries:
        for j in i:
            therestx.append(j[0])
            theresty.append(j[1])

    thecoordinates = []
    for i in the_two_boundaries:
        for j in i:
            thecoordinates.append(j)

    return therestx, theresty, thecoordinates
#Plotting the words outside a certain boundary ends here

#SECONDARY WINDOW STARTS here
def second_window(tuplex,tupley,labels,sent_dic,text_input):
    prop = fm.FontProperties(fname='kalpurush.ttf')
    def getboundary(event):
        # print(slider.get())
        boundary = slider.get()
        therestx, theresty, thecoordinates = boundary_word_plot(dictionary, distance_list, boundary)
        sentences = source_of_words(thecoordinates, sent_dic)
        axes1.clear()
        axes1.scatter(therestx,theresty, cmap='Paired')
        word_canvas.draw()
    def window_click(event):
        # print('you pressed', event.button, event.xdata, event.ydata)
        xpos, ypos = event.xdata, event.ydata
        for key in sent_dic:
            for values in sent_dic[key]:
                if abs(xpos - values[0]) < 5 and abs(ypos - values[1]) < 5:
                    text_show = plt.text(event.xdata, event.ydata, key, fontsize=5, fontproperties=prop)
                    word_canvas.draw()

    mapx,mapy = word_plot(labels, tuplex, tupley, text_input.get())
    dictionary, distance_list = distance_x_y(labels, tuplex, tupley, text_input.get())
    maximumdistance = max(distance_list)

    window = Toplevel()
    window.minsize(width=1080, height=900)
    fig, axes1 = plt.subplots()


    axes1.scatter(mapx, mapy, cmap='Paired')

    word_canvas = FigureCanvasTkAgg(fig, window)
    # word_canvas.bind("<Button-1>", window_click)
    word_canvas.mpl_connect('button_press_event', window_click)
    plot_widget = word_canvas.get_tk_widget()
    plot_widget.pack(side = TOP, fill = BOTH, expand = True)
    toolbar = NavigationToolbar2Tk(word_canvas, window)
    toolbar.update()
    toolbar.pack()

    help = Label(window, text="Slide to set boundary",font=("Helvetica", 16))
    help.pack()
    slider = Scale(window,from_=0, to=maximumdistance, orient=HORIZONTAL,command=getboundary)
    slider.pack(fill = BOTH)
    # print(var.get())
    word_canvas.draw()
#SENCONDARY WINDOW ENDS HERE








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
        input_button=Button(self, height=1, width=10, text="Find", command=lambda: second_window(x,y,label,sent_dic,text_input))
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
