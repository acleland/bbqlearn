import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from bbqlearn import *


def add_box(ax, box, color):
    rect = patches.Rectangle((box.x, box.y), box.width, box.height, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)



class Gui:

    def __init__(self):
        self.q = Qlearn()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.q.state.image)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def mark_image(self):
        self.ax.clear()
        self.ax.imshow(self.q.state.image)
        gt = self.q.state.gt
        box = self.q.state.box

        add_box(self.ax, gt, 'g')
        add_box(self.ax, box, 'r')
        plt.draw()


    def onclick(self, event):
        self.q.step()
        self.mark_image()


if __name__=='__main__':
    gui = Gui()