
from tkinter import *
from tkinter.ttk import Frame
from PIL import Image, ImageTk
from bbqlearn import *

h = 2162
w = 2029

FRAME_W = 1000
FRAME_H = 500

class Application(Frame):
    def __init__(self, master):
        super(Application, self).__init__(master)
        self.grid()
        self.bttn_clicks = 0
        
        self.label = Label(self, text ='Hell yeah, I\'m bad-ass')
        self.label.grid(row=0, column=0)

        self.button = Button(self, text = 'Push my button!')
        self.button['command'] = self.update_count
        self.button.grid(row=1, column=1)
        
        self.count_lbl = Label(self, text = str(self.bttn_clicks))
        self.count_lbl.grid(row=2, column=1)

        self.img = Image.open('../Data/Train/pdw1.jpg')
        self.img = self.img.resize((FRAME_W,FRAME_H))
        self.imgP = ImageTk.PhotoImage(self.img)
        self.imglbl = Label(self, text = 'Oooh, pretty!', image =self.imgP)
        self.imglbl.grid(row=1,column=0, rowspan=10)


    def update_count(self):
        self.bttn_clicks += 1
        self.count_lbl['text'] = str(self.bttn_clicks)


root = Tk()
root.title('Super Bad-Ass GUI')
root.geometry('1200x1000')
app = Application(root)

root.mainloop()



