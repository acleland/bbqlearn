
from tkinter import *
from tkinter.ttk import Frame
from PIL import Image, ImageTk, ImageDraw
from bbqlearn import *

h = 2162
w = 2029

FRAME_W = 1000
FRAME_H = 600
SIZE = FRAME_W, FRAME_H

class Application(Frame):
    def __init__(self, master):
        super(Application, self).__init__(master)
        self.grid()
        self.bttn_clicks = 0
        
        self.label = Label(self, text ='Super Awesome Image Viewer')
        self.label.grid()

        self.button = Button(self, text = 'Push my button!')
        self.button['command'] = self.update_count
        self.button.grid()
        
        self.count_lbl = Label(self, text = str(self.bttn_clicks))
        self.count_lbl.grid()

        self.showpic = Button(self, text = 'Show me a picture!', command = self.show_me)
        self.showpic.grid()

        self.original = Image.open('../Data/Train/pdw1.jpg')
        _, self.skew, self.gt = parse_label(read_label('../Data/Train/pdw1a.labl'))
        print('gt:', self.gt)
        self.draw_box(self.original, self.gt, 'red', linewidth=5)


    def update_count(self):
        self.bttn_clicks += 1
        self.count_lbl['text'] = str(self.bttn_clicks)

    def draw_box(self, img, box, color, linewidth):
        draw = ImageDraw.Draw(img)
        rect = box.toVector2()
        for line in range(linewidth):
            draw.rectangle(rect , fill=None, outline=color)
            rect = [rect[0]+1, rect[1]+1, rect[2]-1, rect[3]-1]  # Draw progressively smaller rects

    def show_me(self):
        self.original.show()

 


root = Tk()
root.title('Super Bad-Ass GUI')
#root.geometry('1000x700')
app = Application(root)

root.mainloop()



