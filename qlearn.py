import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.special import expit
import time
import pickle
from collections import deque
from tools import *
from HOG_Env import *
import os

# from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
# from keras.models import Model
# from keras.preprocessing import image
# from PIL import Image

# Some constants


HUMANS = "../Data/Humans/"
DOGS = "../Data/Skews3/"

IMAGE_PATH = "../Data/Cropped/"
SAVE_PATH = "../Output/dumbtest/"

NUM_EPOCHS = 1
ACTIONS_PER_EPISODE = 2
VISUAL = True

LEARNING_RATE = 0.2
EPSILON = 0.5
DISCOUNT_FACTOR = 0.1

def EPS_CONST(epoch):
    return EPSILON

def halfsy(epoch):
    # starts at .5 and halves every 200 epochs
    return np.power(2, -1 -(epoch-1)/200)

# Utility Functions

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

class Perceptron:
    def __init__(self, num_actions, input_vector_length):
        # random weights in range -.05..+.05
        self.weights = np.random.rand(num_actions, input_vector_length + 1)*.1 - .05
        #self.weights = np.random.randn(num_actions, input_vector_length)/np.sqrt(input_vector_length)
    def __str__(self):
        return str(self.weights)
    def getQ(self, input_vector, action_index):
        x = np.concatenate(([1],input_vector))  # insert bias term
        return sigmoid(np.dot(self.weights[action_index], x))
    def getQvector(self, input_vector):
        x = np.concatenate(([1],input_vector))  # insert bias term
        return sigmoid(np.dot(self.weights, np.transpose(x)))
    def update_weights(self, target, output, input_vector, action_index, learning_rate=LEARNING_RATE):
        x = np.concatenate(([1],input_vector))
        delta_w = learning_rate*(target - output) * x
        self.weights[action_index] += delta_w
    def save(self, save_path):
        np.save(save_path, self.weights)
    @staticmethod
    def from_weights(w):
        r, c = w.shape
        p = Perceptron(r,c-1)
        p.weights = w
        return p
    @staticmethod
    def load(load_path):
        w = np.load(load_path)
        return Perceptron.from_weights(w)

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



# --------------------------------------------------------------------------------

def print_stats(vector):
    print('shape', vector.shape)
    print('min', np.min(vector))
    print('max', np.max(vector))
    print('mean', np.mean(vector))



def print_round(name, v,decimal=3):
    print(name, np.around(v,decimal)) 


def show_weights(m, n):
    vmin = min(-.1, min(np.min(m),np.min(n)))
    vmax = max(.1, max(np.max(m),np.max(n)))
    fig, axes = plt.subplots(nrows=2)
    plot = axes[0].matshow(m, cmap='rainbow', aspect='auto', vmin=vmin, vmax=vmax)
    plot2 = axes[1].matshow(n, cmap='rainbow',aspect='auto', vmin=vmin, vmax=vmax)
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])
    fig.suptitle('Weights before and after')
    fig.colorbar(plot, cax=cbar_ax)
    plt.show()
    plt.clf()



def get_gt_skews(imagefile, label_path):
    skews = []
    for i in range(10):
        label_f = imagefile + chr(97+i)
        _, skew, gt = parse_label(read_label(label_path + label_f + '.labl'))
        skews.append(skew)
    return gt, skews

def save_array(array, name, folder):
    path = folder + '/' + name + '.npy'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def show_state(img, gt, bb0, bb1):
    plot_img_boxes(img, [gt, bb0, bb1], ['y','r','w'], slimming=False)

# --------------------------------------------------------------------------------

def box_in_bounds(img_shape, box):
    left, top, right, bottom = box.toVector2()
    width = img_shape[1]
    height = img_shape[0]
    if left < 0 or right > width:
        return False
    elif top < 0 or bottom > height:
        return False
    return True


def load_images(image_path, image_list):
    images = {}
    for image_label in image_list:
        images[image_label] = load_image(image_path + image_label + '.jpg')
    return images


def create_skew(box, image_shape, min_iou):
    skewed = []
    data = box.toVector()
    FRAC = .25
    sigma_x = FRAC * box.width
    sigma_y = FRAC * box.height
    sigma = (sigma_x, sigma_y, sigma_x, sigma_y)
    for i in range(4):
            skew = int(np.random.normal(0, sigma[i]))
            skewed.append(max(data[i] + skew, 0))
    skewed = Box.fromVector(skewed).round()
    if box_in_bounds(image_shape, skewed) and skewed.iou(box) > min_iou:
        return skewed
    else: 
        return create_skew(box, image_shape, min_iou)


# --------------------------------------------------------------------------------


# Q-learning algorithm
class Qlearn:
    def __init__(self, env=None):
        #self.env = env
        self.perc = Perceptron(NUM_ACTIONS, STATE_LENGTH)

    def run(self, save_path, train_list, subject='dogs', 
                                        num_epochs= NUM_EPOCHS, 
                                        actions_per_episode=ACTIONS_PER_EPISODE,
                                        learning_rate=LEARNING_RATE,
                                        discount_factor=DISCOUNT_FACTOR,
                                        epsilon_func = EPS_CONST, 
                                        visual=VISUAL):
        
        if subject.upper() == 'HUMANS':
            label_data = pickle.load(open('../Data/Cropped/human_labels.p','rb'))
            IMAGE_PATH = '../Data/Cropped/Humans/'
        elif subject.upper() == 'DOGS':
            label_data = pickle.load(open('../Data/Cropped/dog_labels.p','rb'))
            IMAGE_PATH = '../Data/Cropped/Dogs/'
        
        # Create example list for training
        exlist = []
        for image_label in train_list:
            gt, skews = label_data[image_label]
            for skew in skews:
                exlist.append((image_label, gt, skew))

        # Load images into memory
        images = {}
        for image_label in train_list:
            images[image_label] = load_image(IMAGE_PATH + image_label + '.jpg')

        np.random.shuffle(exlist)
        
        perc = self.perc
        
        # init reward data
        avg_reward_by_epoch = []
        
        # init epsilon data
        epsilon_by_epoch = []
        weights_by_epoch = [np.copy(perc.weights)]
        
        # Improvement rate and avg change in iou data
        imp_rates = []
        avg_change_ious = []

        # Action count by epoch
        actions_by_epoch = []

        for epoch in range(1, num_epochs+1):
            rewards_this_epoch = []
            epsilon = epsilon_func(epoch)
            epsilon_by_epoch.append(epsilon)
            episode_count = 0
            imp_count = 0
            iou_changes = []
            action_count = [0 for _ in range(NUM_ACTIONS)]
            
            print('Epoch', epoch, 'of', num_epochs)

            for example in exlist: 
                if visual:
                    print(example)
                
                image_label, gt, skew = example  
                img = images[image_label]

                state = State(img, skew, history_length = HISTORY_LENGTH)
                s_prime = state.get_vector()
                init_iou = skew.iou(gt)

                for i in range(1, actions_per_episode+1):
                    if visual:
                        print('Action', i)

                    # Current state s
                    s = s_prime

                    # Get Q(s) vector
                    Qs = perc.getQvector(s)

                    # Select action according to epsilon greedy
                    best_action = random_argmax(Qs)
                    a = epsilon_choose(NUM_ACTIONS, best_action, epsilon)
                    action_count[a] += 1

                    # Compute Q(s,a)
                    Qsa = Qs[a]

                    # Take action a to obtain s'
                    iou = gt.iou(state.box)
                    state.take_action(a)
                    s_prime = state.get_vector()
                    new_iou = gt.iou(state.box)
                    r = np.sign(new_iou-iou)
                    rewards_this_epoch.append(r)

                    # Compute Q(s')
                    Qs_prime = perc.getQvector(s_prime)
                    Qs_prime_max = np.max(Qs_prime)
                    
                    
                    # Compute target y
                    if i == actions_per_episode:
                        y = sigmoid(r)
                    else:
                        y = sigmoid(r + discount_factor * Qs_prime_max)
                    
                    # Update weights
                    if visual:
                        old_weights = np.copy(perc.weights)
                    perc.update_weights(y, Qsa, s, a, learning_rate)

                    if visual:
                        # Show relevant information
        
                        print_round('Q(s)', Qs)
                        print('a:', a, actions[a])
                        print('Q(s,a)=', Qsa)
                               
                        #env.show()
                        show_state(img, gt, skew, state.box)
                        plt.show()
                        plt.close()

                        print("\ns', r computed from action a")
                        print('old iou', init_iou)
                        print('new iou', new_iou)
                        print("reward", r)
                        print_round("Q(s')", Qs_prime)
                        print("max_a' Q(s'):", Qs_prime_max)

                        print("\ny = sigmoid( r + " + str(discount_factor) + "*max_a' Q(s') )")
                        print("y=", y)
                        print("Error: y - Q(s,a)", y - Qsa)
                        
                        print('weights updated according to dw =', str(learning_rate), '* (y - Q(s,a))*s')
                        print()

                episode_count += 1
                new_iou = state.box.iou(gt)
                iou_change = new_iou - init_iou
                iou_changes.append(iou_change)
                if iou_change > 0:
                    imp_count += 1


            # Record various epoch data 
            avg_change_ious.append(np.mean(iou_changes))
            imp_rates.append(imp_count/episode_count)
            avg_reward_by_epoch.append(np.mean(rewards_this_epoch))
            weights_by_epoch.append(np.copy(perc.weights))
            actions_by_epoch.append(action_count)


            # Save data every X epochs and at end of training
            if (epoch-1) % 20 == 0 or epoch == num_epochs:
                save_array(perc.weights, 'perceptron', save_path)
                save_array(avg_reward_by_epoch, 'avg_reward_by_epoch', save_path)
                save_array(weights_by_epoch, 'weights_by_epoch', save_path)
                save_array(epsilon_by_epoch, 'epsilon_by_epoch', save_path)
                save_array(imp_rates, 'imp_rates_by_epoch', save_path)
                save_array(avg_change_ious, 'avg_change_ious_by_epoch', save_path)
                save_array(actions_by_epoch, 'actions_by_epoch', save_path)

                tlistname = save_path + '/train_list.p'
                os.makedirs(os.path.dirname(tlistname), exist_ok=True)
                pickle.dump(train_list, open(tlistname, 'wb'))

        print('Training complete. Saving to ' + save_path)



# --------------------------------------------------------------------------------
if __name__ == '__main__':
    tlist = get_imgfiles(1,2)
    Qlearn().run(SAVE_PATH, tlist, subject='dogs', visual=True)

    
