import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import re 
import time
import pickle
import random
from collections import deque
from scipy.special import expit
import sys
from enum import Enum

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.models import Model
from keras.preprocessing import image
from PIL import Image

TRAIN_PATH = "../Data/Train/"
OUTPUT_PATH = "../Output/"
SAVE_FILE = "testfile.p"

# Parameters
NUM_EPOCHS = 10
TRAIN_SET_SIZE = 2
ACTIONS_PER_EPISODE = 5
DISCOUNT_FACTOR = 0.5
LEARNING_RATE = 0.2
EPSILON = 0.5

Actions = Enum('Actions', 'LEFT RIGHT UP DOWN BIGGER SMALLER FATTER TALLER STOP')

NUM_ACTIONS = len(Actions)
HISTORY_LENGTH = 2
NUM_FEATURES = 4096
STATE_VECTOR_LENGTH = NUM_FEATURES + NUM_ACTIONS * HISTORY_LENGTH
SHIFT = 5  # number of pixels box shifts by in each left, right, up, down action
ZOOM_FRAC = 0.1  # Fraction box zooms by in bigger, smaller, taller, fatter actions

PRINTING = False


# Utility Functions

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


def epsilon_choose(num_choices, index_of_best_choice, epsilon):
    """Returns index_of_best_choice with probability 1-epsilon), 
    otherwise returns an index in range 0 to num_choices-1 with a uniform random probability"""
    if np.random.rand() < epsilon:
        return np.random.randint(0, num_choices) 
    return index_of_best_choice

def epsilon_choose_test(epsilon):
    bins = np.zeros(10,int)
    maxiter = 1000
    for _ in range(maxiter):
        n = epsilon_choose(10, 3, epsilon)
        bins[n] += 1
    hist = bins / maxiter
    print(bins)
    print(hist)

# --------------------------------------------------------------------------------
# File scripting

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
    return image.load_img(filepath)

def get_train_labels():
    fnames = []
    for labelFile in glob.glob(TRAIN_PATH + '*.labl'):
        labelName = re.search(TRAIN_PATH + '(.+)\.labl', labelFile).group(1)
        fnames.append(labelName)
    return fnames

def get_train_validation():
    labels = get_train_labels()
    train = []
    validation = []
    for label in labels:
        labelnum = int(re.search('(pdw)([0-9]+)([a-z])',label).group(2))
        if labelnum <= 390:
            train.append(label)
        else:
            validation.append(label)
    return train, validation

# --------------------------------------------------------------------------------

def make_square_with_centered_margins(img):
    # Find longer, shorter sides, return img if sides are equal
    if (img.shape[0] < img.shape[1]):
        short_side = 0
        long_side = 1
        # Need to add diff=longer-shorter to shorter side
        diff = img.shape[long_side]-img.shape[short_side]
        # Add floor(diff/2) to "bottom", (diff - floor(diff/2)) to "top"
        append_margin = np.floor_divide(diff,2)
        prepend_margin = diff - append_margin
        append_zeros = np.zeros((append_margin, img.shape[long_side], img.shape[2]))
        prepend_zeros = np.zeros((prepend_margin, img.shape[long_side], img.shape[2]))
        new_image = np.concatenate((img,append_zeros), axis=0)
        new_image = np.concatenate((prepend_zeros, new_image), axis=0)
        return new_image
    elif (img.shape[1] < img.shape[0]):
        short_side = 1
        long_side = 0
        # Need to add diff=longer-shorter to shorter side
        diff = img.shape[long_side]-img.shape[short_side]
        # Add floor(diff/2) to "bottom", (diff - floor(diff/2)) to "top"
        append_margin = np.floor_divide(diff,2)
        prepend_margin = diff - append_margin
        append_zeros = np.zeros((img.shape[long_side], append_margin, img.shape[2]))
        prepend_zeros = np.zeros((img.shape[long_side], prepend_margin, img.shape[2]))
        new_image = np.concatenate((img,append_zeros), axis=1)
        new_image = np.concatenate((prepend_zeros, new_image), axis=1)
        return new_image
    else:
        return img  # the image is already square

# --------------------------------------------------------------------------------

class Perceptron:
    def __init__(self, num_actions, input_vector_length, learning_rate, bias=True):
        # random weights in range -.05..+.05
        self.weights = np.random.rand(num_actions, input_vector_length)*.1 - .05
        #self.weights = np.random.randn(num_actions, input_vector_length)/np.sqrt(input_vector_length)
        self.learning_rate = learning_rate
    def __str__(self):
        return str(self.weights)
    def getQ(self, input_vector, action_index):
        return sigmoid(np.dot(self.weights[action_index], input_vector))
    def getQvector(self, input_vector):
        return sigmoid(np.dot(self.weights, np.transpose(input_vector)))
    def update_weights(self, target, output, input_vector, action_index):
        delta_w = self.learning_rate*(target - output) * input_vector
        self.weights[action_index] += delta_w
    def learn(self, target, input_vector, action_index):
        output = self.getQ(input_vector, action_index)
        self.update_weights(target, output, input_vector, action_index)

def testPerceptron():
    x = np.array([3.0,4.0])
    shifted_x = np.array([0.0,0.0])
    perceptron = Perceptron(len(actions), 2, .5)
    print("ground_truth: ", x)
    print("shifted_x: ", shifted_x)
    print("initial weights: \n", perceptron.weights)
    for i in range(4):
        action_index = i  # np.random.randint(len(actions))
        action = actions[action_index]
        print("action: ", action.__name__)
        new_state = action(shifted_x)
        print("new state: ", new_state)

def testPerceptron2():
    img = skimage.io.imread("pdw1.jpg")
    img = img / 255.0
    ground_truth = Box(1605, 346, 22, 126)
    shifted = Box(1630, 446, 22, 126)
    state = State(img, shifted)
    print("bounding box:", state.box)
    features = state.get_features()
    print("features:", features)
    print("features shape:", features.shape)
    perceptron = Perceptron(len(state.actions), len(features), 0.5)
    qvec = perceptron.getQvector(features)
    state.left()
    features2 = state.get_features()
    qvec2 = perceptron.getQvector(features2)
    print("Q(s)", qvec)
    print("Q(s')", qvec2)




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

def testVgg():
    img_f, bb, gt = parse_label(read_label(TRAIN_PATH + 'pdw1a.labl'))
    pdw1 = Image.open(TRAIN_PATH + img_f + '.jpg')
    crop = pdw1.crop(gt.toVector2())
    vgg = VggHandler()
    features = vgg.get_fc6(crop)
    print(features)

# --------------------------------------------------------------------------------


class State:
    def __init__(self, image, box, history_length=HISTORY_LENGTH):
        
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

    def get_features(self):
        crop = get_crop(self.image, self.box)
        return vgg.get_fc6(crop)[0]

    def get_vector(self):
        features = self.get_features()
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

# Q-learning algorithm

class Qlearn:
    def __init__(self, NumEpochs=NUM_EPOCHS, TrainSetSize=TRAIN_SET_SIZE, printing=PRINTING):
        # Save parameters as attributes for reference purposes
        self.num_epochs = NumEpochs
        self.train_set_size = TrainSetSize
        self.actions_per_episode = ACTIONS_PER_EPISODE
        self.discount_factor = DISCOUNT_FACTOR
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON
        self.num_actions = NUM_ACTIONS
        self.history_length = HISTORY_LENGTH
        self.state_vector_length = NUM_FEATURES + self.history_length * self.num_actions
        self.printing = printing

        # Initialize the VGG16 Model
        self.vgg = VggHandler()

        # Initialize perceptron and training data        
        self.perceptron = Perceptron(num_actions=self.num_actions, input_vector_length=self.state_vector_length, learning_rate=self.learning_rate) 
        self.train_list, _ = get_train_validation()
        self.train_list = self.train_list[:self.train_set_size] 
        random.shuffle(self.train_list)
        self.done = False

        # Reward Data initialization
        self.rewards = []
        self.avg_reward_by_episode = []
        self.avg_reward_by_epoch = []

        # Variable for keeping track of time spent extracting features
        # self.action_time = 0.0

        # # This stuff is for initializing the q.step() function. (for visualization)
        # self.epoch_number = 1
        # self.episode  = 1
        # self.actions_taken = 0
        # self.total_actions = 0
        
        # random.shuffle(self.train_list)
        # self.train_queue = deque(self.train_list)
        # example = self.train_queue.pop()
        # self.actions_taken = 0
        # self.state = State(example, self.history_length)

        
    def __str__(self):
        s = '\nQlearn Status: \nPerceptron:\n' + str(self.perceptron)
        s += '\nTrain List: {:d} items'.format(len(self.train_list)) 
        s +=  '\n' + str(self.train_list)
        s += '\nDone?: ' + str(self.done)
        s += '\nNum Epochs: {:d}'.format(self.num_epochs)
        s += '\nActions Per Episode: {:d}'.format(self.actions_per_episode)
        s += '\nDiscount Factor: {:.2f}'.format(self.discount_factor)
        s += '\nLearning Rate: {:.2f}'.format(self.learning_rate)
        s += '\nEpsilon: {:.2f}'.format(self.epsilon)
        s += '\nNum Actions: {:d}'.format(self.num_actions)
        s += '\nHistory Length: {:d}'.format(self.history_length)
        s += '\nState Vector Length: {:d}'.format(self.state_vector_length)
        return s

    def print(self):
        print(str(self))

    def next_action(self, state, gt, end_of_episode):
        # Select action according to epsilon greedy
        s = state.get_vector()
        Qs = self.perceptron.getQvector(s)
        best_action = random_argmax(Qs)
        action_index = epsilon_choose(self.state.num_actions, best_action, epsilon=self.epsilon)
        action_name = state.actions[action_index].__name__

        # Compute Q(s,a)
        Qsa = self.perceptron.getQ(s, action_index)
        iou = state.box.iou(gt)

        # Take action to get new state s'
        state.take_action(action_index)
        s_prime = state.get_vector()
        new_iou = state.box.iou(gt)
        delta_iou = new_iou - iou
        r = np.sign(delta_iou)  # immediate reward
        

        # Compute Q(s',a') for all a' in actions
        Qs_prime = self.perceptron.getQvector(s_prime)

        # Determine output, apply perceptron learning rule
        y = r
        if not end_of_episode:
            y += self.discount_factor*np.max(Qs_prime)
        
        self.perceptron.update_weights(y, Qsa, s_prime, action_index)
        return action_name, delta_iou, y

    def run(self, save_name):
        self.start_time = time.time()
        for epoch_number in range(self.num_epochs):
            rewards_this_epoch = []
            print("Epoch: ", epoch_number+1, "of", self.num_epochs)
            random.shuffle(self.train_list)
            for episode_num in range(len(self.train_list)):
                rewards_this_episode = []
                example = self.train_list[episode_num]
                print('Episode', episode_num + 1, 'of', len(self.train_list), example)
                imgf, bb, gt = parse_label(read_label(TRAIN_PATH + example + '.labl'))
                img = load_image(TRAIN_PATH + imgf + '.jpg')
                self.state = State(img, bb, self.history_length)
                for action_number in range(self.actions_per_episode):
                    print('Action ', action_number+1)
                    t = time.time()
                    action, delta_iou, reward = self.next_action(self.state, gt, action_number+1 == self.actions_per_episode)
                    
                    print(action, 'delta iou:', delta_iou, 'reward:', reward)
                    self.rewards.append(reward)
                    rewards_this_episode.append(reward)
                    rewards_this_epoch.append(reward)
                self.avg_reward_by_episode.append(np.mean(rewards_this_episode))
            self.avg_reward_by_epoch.append(np.mean(rewards_this_epoch))

        self.done = True
        self.runtime = time.time() - self.start_time
        self.time_per_action = self.runtime / self.num_epochs / len(self.train_list) / self.actions_per_episode
        print('Training Complete. Saving to', save_name)
        print('Run time:', self.runtime)
        print('Average time per action', self.runtime/self.num_epochs / len(self.train_list) / self.actions_per_episode)
        # print('Time per action:', self.time_per_action)
        # print('Total extraction time:', self.action_time)
        # print('Average extraction time:', self.action_time / self.num_epochs / len(self.train_list) / self.actions_per_episode)
        

        plt.figure(1)
        x = np.arange(1,self.num_epochs+1)
        y = np.asarray(self.avg_reward_by_epoch)
        plt.plot(x,y)
        plt.xlabel('Epoch')
        plt.ylabel('Average Reward')
        plt.savefig(save_name + '_reward_by_epoch.pdf', bbox_inches='tight')

        plt.figure(2)
        x = np.arange(1,len(self.rewards)+1)
        y = np.asarray(self.rewards)
        plt.plot(x,y, '-')
        plt.xlabel('Action')
        plt.ylabel('Reward')
        plt.savefig(save_name + '_reward_by_action.pdf', bbox_inches='tight')

        plt.figure(2)
        x = np.arange(1,len(self.rewards)+1)
        y = np.asarray(self.rewards)
        plt.plot(x,y, '-')
        plt.xlabel('Action')
        plt.ylabel('Reward')
        plt.savefig(save_name + '_reward_by_action.pdf', bbox_inches='tight')

        plt.figure(3)
        x = np.arange(1,len(self.avg_reward_by_episode)+1)
        y = np.asarray(self.avg_reward_by_episode)
        plt.plot(x,y, '-')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(save_name + '_reward_by_episode.pdf', bbox_inches='tight')
        
        pickle.dump(self.perceptron, open(save_name + '_perceptron.p', 'wb'))
        plt.show()




# --------------------------------------------------------------------------------

# Load Initial Training Features and set up VGG Handler
training_features = pickle.load(open('all_features.p', 'rb'))
vgg = VggHandler()

# --------------------------------------------------------------------------------
if __name__ == '__main__':
    Qlearn().run(SAVE_FILE)

    
