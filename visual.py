
from PIL import Image, ImageTk
from tkinter import *
from tkinter.ttk import *
#from bbqlearn import *

h = 2162
w = 2029

class Application(Frame):
    def __init__(self, master):
        super(Application, self).__init__(master)
        self.grid()
        self.create_widgets()
    def create_widgets(self):
        self.label = Label(self, text ='Hell yeah, I\'m bad-ass')
        self.label.grid()
        self.button = Button(self, text = 'Push my button!')
        self.button.grid()


root = Tk()
root.title('Super Bad-Ass GUI')
root.geometry('210x220')
app = Application(root)


root.mainloop()



