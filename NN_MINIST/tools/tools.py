import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class plotool(object):
    mpl.rcParams['xtick.labelsize'] = 24
    mpl.rcParams['ytick.labelsize'] = 24

    def data_graph(self,x,y,name='data',save=False, color_shape='r.'):
        plt.figure(name)
        plt.plot(x, y, color_shape, lw=3)
        if save is True:
            plt.savefig('./result.png')
        plt.show()

    def func2X_plot(self,func_name,func,save=False, color_shape='r.'):
        fig1 = plt.figure(func_name)

        x = np.linspace(-5, 5, 100)
        y = func(x)
        plt.plot(x, y, color_shape, lw=3)
        if save is True:
            plt.savefig('./result.png')
        plt.show()

if __name__ == 'main':
    x = np.linspace(0, 5, 100)
    y = 2*np.sin(x) + 0.3*x**2
    y_data = y + np.random.normal(scale=0.3, size=100)
    plot = plotool()
    plot.data_graph(x,y_data)
