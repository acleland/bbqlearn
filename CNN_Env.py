from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.models import Model
from keras.preprocessing import image
from image_tools import *

class CNN_Env:
    def __init__(self):
        