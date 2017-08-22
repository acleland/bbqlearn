import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.special import expit
import time
import pickle
from collections import deque
from tools import *
import os

# from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
# from keras.models import Model
# from keras.preprocessing import image
# from PIL import Image

# Some constants

NUM_EPOCHS = 3
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


def bar_plot(v, xlabel=None, ylabel=None):
    x = np.arange(len(v))
    plt.bar(x, v)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()

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

# Q-learning algorithm

class Qlearn:
    def __init__(self, env):
        self.env = env
        self.perc = Perceptron(env.num_actions, env.state_length)

    def run(self, save_path, train_list, num_epochs= NUM_EPOCHS, 
                                        actions_per_episode=ACTIONS_PER_EPISODE,
                                        learning_rate=LEARNING_RATE,
                                        discount_factor=DISCOUNT_FACTOR,
                                        epsilon_func = EPS_CONST, 
                                        visual=VISUAL):
        np.random.shuffle(train_list)
        env = self.env
        perc = self.perc
        # init reward data
        avg_reward_by_epoch = []
        # init epsilon data
        epsilon_by_epoch = []
        weights_by_epoch = [np.copy(perc.weights)]
        for epoch in range(1, num_epochs+1):
            rewards_this_epoch = []
            epsilon = epsilon_func(epoch)
            epsilon_by_epoch.append(epsilon)
            print('Epoch', epoch, 'of', num_epochs)
            for episode in range(1, len(train_list)+1):
                if visual:
                    print('Episode', episode, 'of', len(train_list))
                train_ex = train_list[episode-1]
                if visual:
                    print(train_ex)
                env.load(train_ex)
                s_prime = None
                for i in range(1, actions_per_episode+1):
                    if visual:
                        print('Action', i)
                    # Get the state from the environment
                    if s_prime is not None:
                        s = s_prime
                    else:
                        s = env.get_state()
                    # Get Q(s) vector
                    Qs = perc.getQvector(s)
                    # Select action according to epsilon greedy
                    best_action = random_argmax(Qs)
                    a = epsilon_choose(env.num_actions, best_action, epsilon)
                    # Compute Q(s,a)
                    Qsa = Qs[a]
                    # Take action a to obtain s'
                    s_prime, r = env.take_action(a)
                    rewards_this_epoch.append(r)
                    # Compute Q(s')
                    Qs_prime = perc.getQvector(s_prime)
                    Qs_prime_max = np.max(Qs_prime)
                    y = sigmoid(r + discount_factor * Qs_prime_max)
                    # Update weights
                    if visual:
                        old_weights = np.copy(perc.weights)
                    perc.update_weights(y, Qsa, s, a, learning_rate)
                    # Show relevant information
                    
                    
                    
        
                    if visual:
                        print_round('Q(s)', Qs)
                        print('a:', a, env.actions[a])
                        print('Q(s,a)=', Qsa)
                    
                    if visual:
                        env.show()
                        print("\ns', r computed from action a")
                        print("r", r)
                        print_round("Q(s')", Qs_prime)
                        print("max_a' Q(s'):", Qs_prime_max)

                        print("\ny = sigmoid( r + " + str(discount_factor) + "*max_a' Q(s') )")
                        print("y=", y)
                        print("Error: y - Q(s,a)", y - Qsa)
                        
                        print('weights updated according to y - Q(s,a)')
                        print('new weights stats')
                        print_stats(perc.weights)
                        if visual:
                            show_weights(old_weights, perc.weights)
                        print()
            avg_reward_by_epoch.append(np.mean(rewards_this_epoch))
            weights_by_epoch.append(np.copy(perc.weights))
        print('Training complete. Saving to ' + save_path)
        # Save data and charts from run
        percname = save_path + '/perceptron.npy'
        rname = save_path + '/avg_reward_by_epoch.npy'
        wname = save_path + '/weights_by_epoch.npy'
        rfigname = save_path + '/reward_by_epoch.pdf'
        tlistname = save_path + '/train_list.p'
        eps_name = save_path + '/epsilon_by_epoch.npy'
        os.makedirs(os.path.dirname(percname), exist_ok=True)
        os.makedirs(os.path.dirname(rname), exist_ok=True)
        os.makedirs(os.path.dirname(wname), exist_ok=True)
        os.makedirs(os.path.dirname(wname), exist_ok=True)
        os.makedirs(os.path.dirname(eps_name), exist_ok=True)
        perc.save(percname)
        np.save(rname, avg_reward_by_epoch)
        np.save(wname, weights_by_epoch)
        np.save(eps_name, epsilon_by_epoch)
        pickle.dump(train_list, open(tlistname, 'wb'))
        if visual:
            os.makedirs(os.path.dirname(rfigname), exist_ok=True)
            plt.figure(1)
            x = np.arange(1,num_epochs+1)
            y = np.asarray(avg_reward_by_epoch)
            plt.plot(x,y)
            plt.xlabel('Epoch')
            plt.ylabel('Average Reward')
            plt.savefig(rfigname, bbox_inches='tight')
            plt.show()


# --------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
    #Qlearn().run(SAVE_FILE)

    
