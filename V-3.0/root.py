from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
import numpy as np
#commandline
import argparse
#visualization
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource,LabelSet,HoverTool
import bokeh
#font
import matplotlib.font_manager as fm
# import pyglet
# pyglet.font.add_file('AponaLohit.ttf')
#plugin architecture
import importlib
#euclideandistance
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('euclidean')

PLUGIN_NAME = "plugins.core_3"
CLUSTER_PLUGIN = "plugins.core2_cluster"
SECOND_WIND = "plugins.core_4"
plugin_module = importlib.import_module(PLUGIN_NAME, '.')
plugin_module_2 = importlib.import_module(CLUSTER_PLUGIN, '.')
plugin_module_3 = importlib.import_module(SECOND_WIND, '.')

#SECONDARY WINDOW STARTS here
def second_window(tuplex,tupley,labels,sent_dic,text_input):
    plugin3 = plugin_module_3.Plugin()
    font_name = fm.FontProperties(fname='kalpurush.ttf')
    def getboundary(event):
        # print(slider.get())
        boundary = slider.get()
        # therestx, theresty, thecoordinates = boundary_word_plot(dictionary, distance_list, boundary)
        therestx, theresty, thecoordinates = plugin3.boundary_word_plot(dictionary, distance_list, boundary)
        # sentences = source_of_words(thecoordinates, sent_dic)
        sentences = plugin3.source_of_words(thecoordinates, sent_dic)
        axes1.clear()
        axes1.scatter(therestx,theresty, cmap='Paired')
        word_canvas.draw()
    def window_click(event):
        # print('you pressed', event.button, event.xdata, event.ydata)
        xpos, ypos = event.xdata, event.ydata
        for key in sent_dic:
            for values in sent_dic[key]:
                if abs(xpos - values[0]) < 5 and abs(ypos - values[1]) < 5:
                    text_show = plt.text(event.xdata, event.ydata, key, fontsize=5, fontproperties=font_name)
                    word_canvas.draw()

    # mapx,mapy = word_plot(labels, tuplex, tupley, text_input.get())
    mapx, mapy = plugin3.word_plot(labels, tuplex, tupley, text_input.get())
    # dictionary, distance_list = distance_x_y(labels, tuplex, tupley, text_input.get())
    dictionary, distance_list = plugin3.distance_x_y(labels, tuplex, tupley, text_input.get())
    maximumdistance = max(distance_list)

    window = Toplevel()
    window.minsize(width=1080, height=900)
    fig, axes1 = plt.subplots()


    axes1.scatter(mapx, mapy, cmap='Paired', label=text_input.get())
    axes1.legend(prop=font_name,title="Given word: ", borderpad=0.5 )
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

        # print(plugin_module)

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

        # f, axes = plt.subplots(nrows = 2, ncols=1)

        y_pred, centers = plugin2.initial(np_flat_list,label)

        # axes[0].scatter(np_flat_list[:, 0], np_flat_list[:, 1],c=y_pred, cmap='Paired')
        # axes[0].scatter(centers[:, 0], centers[:, 1],c='black', alpha=0.5)
        # for i in range(len(label)):
        #
        #     p = axes[1].scatter(np_flat_list[:, 0], np_flat_list[:, 1],c=y_pred, cmap='Paired')
        #     axes[1].scatter(np_flat_list[:, 0], np_flat_list[:, 1],c=y_pred, cmap='Paired')
        #     plt.annotate(label[i],
        #                     xy=(x[i], y[i]),
        #                     xytext=(0, 0),
        #                     textcoords='offset points',
        #                     ha='right',
        #                    fontsize=19, fontproperties=prop)
        #
        #
        # """UI work in tkinter and integration of tkinter
        #  with matplotlib"""
        # canvas = FigureCanvasTkAgg(f, self)
        # f.canvas.mpl_connect('button_press_event', on_click)
        # f.canvas.mpl_connect('button_release_event', off_click)
        # canvas.draw()
        # toolbar = NavigationToolbar2Tk(canvas, self)
        # toolbar.update()
        # toolbar.pack()
        #
        # canvas.get_tk_widget().pack(side = BOTTOM, fill = BOTH, expand = True)
        #
        # text_input = Entry(self)
        # text_input.pack(side = LEFT)
        # input_button=Button(self, height=1, width=10, text="Find", command=lambda: second_window(x,y,label,sent_dic,text_input))
        # input_button.pack(side = LEFT)
        count = 0
        color_palette = {}
        for i in bokeh.palettes.viridis(8):
            color_palette[count] = i
            count +=1
        colors = []
        for i in y_pred:
            if i in color_palette.keys():
                colors.append(color_palette[i])


        source = ColumnDataSource(data={
                'x': np_flat_list[:, 0],
                'y': np_flat_list[:, 1],
                'words': label,
                'cluster_color': colors,
                'sentence' : sent_dic
        })
        output_file("output.html")

        # create a new plot with a title and axis labels
        labels = LabelSet(
            x='x',
            y='y',
            text='words',
            level='glyph',
            x_offset=5,
            y_offset=5,
            source=source,
            render_mode='canvas')

        p = figure(title="BERTENDER", x_axis_label='x', y_axis_label='y',plot_width=1400, plot_height=920)
        hover=HoverTool(tooltips=[
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("Source text:", "@sentence")
        ])
        p.circle('x', 'y', source=source, fill_color='cluster_color', size=10,alpha=0.8)
        p.circle(centers[:, 0], centers[:, 1], fill_color='black', size=10, alpha=0.2)
        p.add_layout(labels)
        p.add_tools(hover)
        show(p)


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
