import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import re 
from PIL import Image
from collections import namedtuple
#from keras.preprocessing import image
from scipy.special import expit

TRAIN_PATH = "../Data/Train/"
OUTPUT_PATH = "../Output/"

# Utility Functions
# --------------------------------------------------------------------------------

def sqdist(p1, p2):
    diff = p2 - p1
    return np.dot(diff, diff)
def dist(p1, p2):
    return np.sqrt(sqdist(p1,p2))
def sse(coords, ground_truth_coords):
    sumv = 0
    for i in range(len(coords)):
        sumv += sqdist(coords[i], ground_truth_coords[i])
    return sumv
def sigmoid(z):
    return expit(z)
def onehot(i, n):
    vec = np.zeros(n)
    vec[i] = 1
    return vec

def random_argmax(vector):
    v = np.asarray(vector)
    return np.random.choice(np.flatnonzero(v == v.max()))


ImageData = namedtuple('ImageData', 'width, height dogs people')

# --------------------------------------------------------------------------------

class Box:
    def __init__(self, x, y, width, height):
        # self.vector = np.array([x,y,x+width-1, y+height-1])
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    @staticmethod
    def fromCenter(center_x, center_y, width, height):
        return Box(int(np.round(center_x - width/2)), int(np.round(center_y-height/2)), width, height)
    @staticmethod
    def fromVector(box):
        return Box(box[0], box[1], box[2], box[3])

    def toVector(self):
        # returns straightforward vector representation of the rectangle
        return [self.x, self.y, self.width, self.height]
    def toVector2(self):
        # returns vector of topleft and bottom-right coordinates of rectangle
        return [self.x, self.y, self.x+self.width, self.y+self.height]

    def copy(self):
        return Box(self.x, self.y, self.width, self.height)

    def area(self):
        return self.width*self.height
    def get_intersection(self, other):
        max_topleft_x = max(self.x, other.x)
        max_topleft_y = max(self.y, other.y)
        min_bottomright_x = min(self.x + self.width, other.x + other.width)
        min_bottomright_y = min(self.y + self.height, other.y + other.height)
        width = min_bottomright_x - max_topleft_x
        height = min_bottomright_y - max_topleft_y
        if (width < 0 or height < 0): 
            return None 
        return Box(max_topleft_x, max_topleft_y, width, height)
    def iou(self, other):
        intersection = self.get_intersection(other)
        if (intersection is not None):
            return intersection.area()/(self.area() + other.area() - intersection.area())
        return 0.0
    def center(self):
        return self.x+self.width/2, self.y+self.height/2
    def adjust_width(self, factor):
        center_x, center_y = self.center()
        return Box.fromCenter(center_x,center_y, self.width*factor, self.height)
    def adjust_height(self, factor):
        center_x, center_y = self.center()
        return Box.fromCenter(center_x,center_y, self.width, self.height*factor)
    def adjust_x(self, delta):
        return Box(self.x + delta, self.y, self.width, self.height)
    def adjust_y(self, delta):
        return Box(self.x, self.y + delta, self.width, self.height)

    def zoom(self,factor):
        center_x, center_y = self.center()
        return Box.fromCenter(center_x,center_y, self.width*factor, self.height*factor)
    def round(self):
        return Box.fromVector(np.round(self.toVector()).astype(int))


    def __str__(self):
        return str((self.x, self.y, self.width, self.height))

# --------------------------------------------------------------------------------
# File scripting and image functions

def get_crop(img, box):
    return img.crop(box.toVector2())

def resize(img, size):
    return img.resize(size)

def get_cropped_resized(pil_image, cropbox):
    return pil_image.crop(tuple(cropbox.toVector2())).resize((224,224))

def get_cropped_resized_from_path(imagepath, cropbox):
    img = Image.open(imagepath)
    crop = img.crop(tuple(cropbox.toVector2()))
    return crop.resize((224,224))


def read_label(filename):
    with open(filename) as f:
        label = f.read()
    return label

def parse_label(label):
    label = label.split('|')
    image_filename = label[0]
    bb = Box.fromVector([int(x) for x in label[1:5]])
    gt = Box.fromVector([int(x) for x in label[5:]])
    return image_filename, bb, gt

def load_image(filepath):
    #return image.load_img(filepath)
    return Image.open(filepath)
def get_train_labels(path):
    fnames = []
    for labelFile in glob.glob(path + '*.labl'):
        labelName = re.search(path + '(.+)\.labl', labelFile).group(1)
        fnames.append(labelName)
    return fnames

def get_train_validation(path):
    labels = get_train_labels(path)
    train = []
    validation = []
    for label in labels:
        labelnum = int(re.search('(pdw)([0-9]+)([a-z])',label).group(2))
        if labelnum <= 390:
            train.append(label)
        else:
            validation.append(label)
    return train, validation

def get_labels(minv, maxv):
    labels = []
    for i in range(minv,maxv+1):
        for j in range(10):
            labels.append('pdw' + str(i) + chr(97 + j))
    return labels
        



