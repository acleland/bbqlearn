import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
from scipy.special import expit
import sys

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.models import Model
from keras.preprocessing import image
from tools import *


# Some constants

TRAIN_PATH = "../Data/Train/"
OUTPUT_PATH = "../Output/"

actions = ['left','right','up','down','bigger','smaller','fatter','taller','stop']
num_actions = len(actions)
history_length = 2
num_features = 4096
state_length = num_features + num_actions * history_length

SHIFT = 5  # number of pixels box shifts by in each left, right, up, down action
ZOOM_FRAC = 0.1  # Fraction box zooms by in bigger, smaller, taller, fatter actions

PRINTING = False



# VGG Stuff
# --------------------------------------------------------------------------------

def vgg_prepare(img):
    x = img.resize((224,224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

class VggHandler:
    def __init__(self):
        self.model = VGG16(weights = 'imagenet', include_top=True)
        self.fc6_model = Model(inputs = self.model.input, outputs=self.model.get_layer(index=20).output)
    def get_fc6(self, img, printing = False):
        x = vgg_prepare(img)
        if printing:
            print(x.shape, x)
        return self.fc6_model.predict(x)

    def get_pred(self, img, printing=False):
        x = vgg_prepare(img)
        preds = self.model.predict(x)
        if printing:
            print(decode_predictions(preds, top=3))
        return preds

# --------------------------------------------------------------------------------

class State:
    def __init__(self, image, box, history_length=history_length):
        
        self.image = image
        self.box = box.copy()

        self.actions = [self.left, self.right, self.up, self.down, self.bigger, self.smaller, self.fatter, self.taller, self.stop]
        self.num_actions = len(self.actions)
        
        self.history_length = history_length
        self.history = deque(maxlen=history_length)
        for _ in range(history_length):
            self.history.appendleft(np.zeros(self.num_actions))
        self.readable_history = deque(maxlen=history_length)

        self.shift = SHIFT
        self.zoom_frac = ZOOM_FRAC
        self.done = False

    def __str__(self):
        return self.image_name + '\n' + str(self.box)

    def get_features(self, vgg):
        crop = get_crop(self.image, self.box)
        return vgg.get_fc6(crop)[0]

    def get_vector(self, vgg):
        features = self.get_features(vgg)
        return np.concatenate((features, np.concatenate(self.history)))

    def get_skew(self):
        return self.image.crop(self.box.toVector2())

    # Actions: (Don't call these explicitly, otherwise, state history  and features 
    #           won't update correctly. Instead use state.take_action(action_number).)
    def left(self):
        self.box = self.box.adjust_x(-self.shift)
    def right(self):
        self.box = self.box.adjust_x(self.shift)
    def up(self):
        self.box = self.box.adjust_y(-self.shift) 
    def down(self):
        self.box = self.box.adjust_y(self.shift) 
    def bigger(self):
        self.box = self.box.zoom(1.0+self.zoom_frac)
    def smaller(self):
        self.box = self.box.zoom(1.0-self.zoom_frac)
    def fatter(self):
        self.box = self.box.adjust_width(1+self.zoom_frac)
    def taller(self):
        self.box = self.box.adjust_height(1+self.zoom_frac)
    def stop(self):
        self.done = True

    def take_action(self, action_number):
        self.actions[action_number]()
        self.box = self.box.round()
        self.history.appendleft(onehot(action_number, self.num_actions))
        self.readable_history.appendleft(self.actions[action_number].__name__)

# --------------------------------------------------------------------------------

class CNN_Env:
    def __init__(self):
        self.actions = actions
        self.num_actions = num_actions
        self.num_features = num_features
        self.state_length = state_length
        self.vgg = VggHandler()
        self.state = None        

    def load(self, example):
        imgf, bb, self.gt = parse_label(read_label(TRAIN_PATH + example + '.labl'))
        img = load_image(TRAIN_PATH + imgf + '.jpg')
        self.state = State(img, bb, history_length)

    def get_state(self):
        return self.state.get_vector(self.vgg)

    def take_action(self, a):
        iou = self.state.box.iou(self.gt)
        self.state.take_action(a)
        new_iou = self.state.box.iou(self.gt)
        r = np.sign(new_iou-iou)
        return self.state.get_vector(self.vgg), r

    def show(self):
        pass
