import matplotlib.pyplot as plt
import numpy as np

class ImagePlot(object):
    def __init__(self, dims, vmin=-1.0, vmax=1.0, name=None):
        self.dims = dims
        self.name = name
        self.vmin = vmin
        self.vmax = vmax

    def init_plot(self, fig, ax, tmax):
        self.fig = fig
        self.ax = ax
        self.tmax = tmax
        if self.name:
            self.ax.set_title(self.name)
        #self.background = fig.canvas.copy_from_bbox(ax.bbox)
        self.img_data = np.zeros((self.dims, tmax)).astype(np.float32)
        self.img_data[0,0] = self.vmax
        self.img_data[0,1] = self.vmin
        #self.img_data = np.random.randn(40,40)
        self.imgplt = self.ax.imshow(self.img_data, interpolation='nearest',animated=True)
        self.imgplt.set_cmap('gray')
        self.img_data = np.zeros((self.dims, tmax)).astype(np.float32)

    def update(self, data):
        self.img_data[:,1:self.tmax] = self.img_data[:,0:self.tmax-1]
        self.img_data[:,0] = data
        self.imgplt.set_data(self.img_data)
        #self.ax.draw_artist(self.imgplt)

class GraphPlot(object):
    def __init__(self, dim, y_min, y_max, name=None):
        self.y_max = y_max
        self.y_min = y_min
        self.dim = dim
        self.name = name

    def init_plot(self, fig, ax, tmax):
        self.fig = fig
        self.ax = ax
        self.tmax = tmax
        ax.set_aspect('equal')
        ax.set_xlim(0, tmax)
        ax.set_ylim(self.y_min, self.y_max)
        ax.hold(True)
        if self.name:
            self.ax.set_title(self.name)
        #self.background = fig.canvas.copy_from_bbox(ax.bbox)
        self.x_data = np.tile(np.arange(0, self.tmax), [self.dim, 1]).T
        self.y_data = np.zeros((self.tmax, self.dim))
        #self.points = self.ax.plot(self.x_data, self.y_data, '-')[0]
        self.points = [None]*self.dim
        for d in range(self.dim):
            self.points[d] = self.ax.plot(self.x_data[:,d], self.y_data[:,d], linestyle='-')[0]

    def update(self, data):
        #self.y_data.append(data)
        #self.y_data = self.y_data[-self.tmax:]
        self.y_data[1:self.tmax] = self.y_data[0:self.tmax-1]
        self.y_data[0] = data
        for d in range(self.dim):
            self.points[d].set_data(self.x_data[:,d], self.y_data[:,d])
        #self.points.set_data(self.x_data, self.y_data)

        #self.fig.canvas.restore_region(self.background)
        #self.ax.draw_artist(self.points)
        #self.fig.canvas.blit(self.ax.bbox)
        #self.fig.canvas.draw()

class NNVis(object):
    def __init__(self, plotobjs, t_max):
        self.plotobjs = plotobjs
        self.t_max = t_max
        self.nplots = len(self.plotobjs)
        self.init_plots()

    def init_plots(self):
        self.axs = [None]*self.nplots
        self.fig, axes = plt.subplots(self.nplots, 1)
        if self.nplots == 1:
            axes = [axes]
        for i in range(self.nplots):
            ax = axes[i]
            self.plotobjs[i].init_plot(self.fig, ax, self.t_max)
        plt.show(False)
        plt.draw()

    def update(self, data):
        assert(len(data) == self.nplots)
        for i in range(self.nplots):
            self.plotobjs[i].update(data[i])
        self.fig.canvas.draw()

    def draw(self):
        pass

def main():
    gph1 = GraphPlot(2, -10, 10, name='plot1')
    gph2 = ImagePlot(50, name='plot2')
    nvis = NNVis([gph1, gph2], 100)
    #nvis.init_plots()
    for i in range(900):
        nvis.update([np.random.randn(2), np.random.randn(50)])
        #nvis.update(0, [np.random.randn(2)*2])
    import pdb; pdb.set_trace()

def imgtest():
    img = np.random.randn(40,40)
    image = plt.imshow(img,interpolation='nearest',animated=True,label="blah") 
    plt.show()

    for k in range(1,100): 
        print k
        img = np.random.randn(40,40) 
        image.set_data(img) 
        plt.draw() 


if __name__ == "__main__":
    #imgtest()
    main()
