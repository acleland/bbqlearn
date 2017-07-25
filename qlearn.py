import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.special import expit
import time
import pickle
from collections import deque


# from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
# from keras.models import Model
# from keras.preprocessing import image
# from PIL import Image

# Some constants

LEARNING_RATE = 0.2
EPSILON = 0.5
DISCOUNT_FACTOR = 0.5

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
    def load(load_path):
        w = np.load(load_path)
        r, c = w.shape
        p = Perceptron(r,c-1)
        p.weights = w
        return p

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


# Q-learning algorithm

class Qlearn:
    def __init__(self, env):
        self.env = env
        self.perc = Perceptron(env.num_actions, env.state_length)

    def run(self, save_path, train_list, num_epochs, actions_per_episode, visual=True):
        np.random.shuffle(train_list)
        env = self.env
        perc = self.perc
        for epoch in range(1, num_epochs+1):
            for episode in range(1, len(train_list)+1):
                train_ex = train_list[episode-1]
                env.load(train_ex)
                for i in range(1, actions_per_episode+1):
                    # Get the state from the environment
                    s = env.get_state()
                    # Get Q(s) vector
                    Qs = perc.getQvector(s)
                    # Select action according to epsilon greedy
                    best_action = random_argmax(Qs)
                    a = epsilon_choose(env.num_actions, best_action, epsilon=EPSILON)
                    # Compute Q(s,a)
                    Qsa = Qs[a]
                    # Take action a to obtain s'
                    s_prime, r = env.take_action(a)
                    # Compute Q(s')
                    Qs_prime = perc.getQvector(s_prime)
                    Qs_prime_max = np.max(Qs_prime)
                    y = sigmoid(r + DISCOUNT_FACTOR * Qs_prime_max)
                    # Update weights
                    perc.update_weights(y, Qsa, s, a)
                    # Show relevant information
                    if visual:
                        env.show()
                    print('s', s)
                    print_stats(s)
                    print()
                    print('a:', a, env.actions[a])
                    print('Q(s,a)=', Qsa)
                
                    print("s'", s_prime)
                    print("r", r)
                    
                    print("max_a' Q(s'):", Qs_prime_max)
                    print("y = sigmoid( r + " + str(DISCOUNT_FACTOR) + "*max_a' Q(s') )")
                    print("y=", y)
                    print("Error: y - Q(s,a)", y - Qsa)
                    
                    print('weights updated according to y - Q(s,a)')
                    print('new weights stats')
                    print_stats(perc.weights)

     

# class Qlearn:
#     def __init__(self, train_list, ):
#         # Save parameters as attributes for reference purposes
#         self.num_epochs = NumEpochs
#         self.train_set_size = TrainSetSize
#         self.actions_per_episode = ACTIONS_PER_EPISODE
#         self.discount_factor = DISCOUNT_FACTOR
#         self.learning_rate = LEARNING_RATE
#         self.epsilon = EPSILON
#         self.num_actions = NUM_ACTIONS
#         self.history_length = HISTORY_LENGTH
#         self.state_vector_length = NUM_FEATURES + self.history_length * self.num_actions
#         self.printing = printing

#         # Initialize the VGG16 Model
#         self.vgg = VggHandler()

#         # Initialize perceptron and training data        
#         self.perceptron = Perceptron(num_actions=self.num_actions, input_vector_length=self.state_vector_length, learning_rate=self.learning_rate) 
#         self.train_list, _ = get_train_validation()
#         self.train_list = self.train_list[:self.train_set_size] 
#         random.shuffle(self.train_list)
#         self.done = False

#         # Reward Data initialization
#         self.rewards = []
#         self.avg_reward_by_episode = []
#         self.avg_reward_by_epoch = []

#         # Variable for keeping track of time spent extracting features
#         # self.action_time = 0.0

#         # # This stuff is for initializing the q.step() function. (for visualization)
#         # self.epoch_number = 1
#         # self.episode  = 1
#         # self.actions_taken = 0
#         # self.total_actions = 0
        
#         # random.shuffle(self.train_list)
#         # self.train_queue = deque(self.train_list)
#         # example = self.train_queue.pop()
#         # self.actions_taken = 0
#         # self.state = State(example, self.history_length)

        
#     def __str__(self):
#         s = '\nQlearn Status: \nPerceptron:\n' + str(self.perceptron)
#         s += '\nTrain List: {:d} items'.format(len(self.train_list)) 
#         s +=  '\n' + str(self.train_list)
#         s += '\nDone?: ' + str(self.done)
#         s += '\nNum Epochs: {:d}'.format(self.num_epochs)
#         s += '\nActions Per Episode: {:d}'.format(self.actions_per_episode)
#         s += '\nDiscount Factor: {:.2f}'.format(self.discount_factor)
#         s += '\nLearning Rate: {:.2f}'.format(self.learning_rate)
#         s += '\nEpsilon: {:.2f}'.format(self.epsilon)
#         s += '\nNum Actions: {:d}'.format(self.num_actions)
#         s += '\nHistory Length: {:d}'.format(self.history_length)
#         s += '\nState Vector Length: {:d}'.format(self.state_vector_length)
#         return s

#     def print(self):
#         print(str(self))

#     def next_action(self, state, gt, end_of_episode):
#         # Select action according to epsilon greedy
#         s = state.get_vector()
#         Qs = self.perceptron.getQvector(s)
#         best_action = random_argmax(Qs)
#         action_index = epsilon_choose(self.state.num_actions, best_action, epsilon=self.epsilon)
#         action_name = state.actions[action_index].__name__

#         # Compute Q(s,a)
#         Qsa = self.perceptron.getQ(s, action_index)
#         iou = state.box.iou(gt)

#         # Take action to get new state s'
#         state.take_action(action_index)
#         s_prime = state.get_vector()
#         new_iou = state.box.iou(gt)
#         delta_iou = new_iou - iou
#         r = np.sign(delta_iou)  # immediate reward
        

#         # Compute Q(s',a') for all a' in actions
#         Qs_prime = self.perceptron.getQvector(s_prime)

#         # Determine output, apply perceptron learning rule
#         y = r
#         if not end_of_episode:
#             y += self.discount_factor*np.max(Qs_prime)
        
#         self.perceptron.update_weights(y, Qsa, s_prime, action_index)
#         return action_name, delta_iou, y

#     def run(self, save_name):
#         self.start_time = time.time()
#         for epoch_number in range(self.num_epochs):
#             rewards_this_epoch = []
#             print("Epoch: ", epoch_number+1, "of", self.num_epochs)
#             random.shuffle(self.train_list)
#             for episode_num in range(len(self.train_list)):
#                 rewards_this_episode = []
#                 example = self.train_list[episode_num]
#                 print('Episode', episode_num + 1, 'of', len(self.train_list), example)
#                 imgf, bb, gt = parse_label(read_label(TRAIN_PATH + example + '.labl'))
#                 img = load_image(TRAIN_PATH + imgf + '.jpg')
#                 self.state = State(img, bb, self.history_length)
#                 for action_number in range(self.actions_per_episode):
#                     print('Action ', action_number+1)
#                     t = time.time()
#                     action, delta_iou, reward = self.next_action(self.state, gt, action_number+1 == self.actions_per_episode)
                    
#                     print(action, 'delta iou:', delta_iou, 'reward:', reward)
#                     self.rewards.append(reward)
#                     rewards_this_episode.append(reward)
#                     rewards_this_epoch.append(reward)
#                 self.avg_reward_by_episode.append(np.mean(rewards_this_episode))
#             self.avg_reward_by_epoch.append(np.mean(rewards_this_epoch))

#         self.done = True
#         self.runtime = time.time() - self.start_time
#         self.time_per_action = self.runtime / self.num_epochs / len(self.train_list) / self.actions_per_episode
#         print('Training Complete. Saving to', save_name)
#         print('Run time:', self.runtime)
#         print('Average time per action', self.runtime/self.num_epochs / len(self.train_list) / self.actions_per_episode)
#         # print('Time per action:', self.time_per_action)
#         # print('Total extraction time:', self.action_time)
#         # print('Average extraction time:', self.action_time / self.num_epochs / len(self.train_list) / self.actions_per_episode)
        

#         plt.figure(1)
#         x = np.arange(1,self.num_epochs+1)
#         y = np.asarray(self.avg_reward_by_epoch)
#         plt.plot(x,y)
#         plt.xlabel('Epoch')
#         plt.ylabel('Average Reward')
#         plt.savefig(save_name + '_reward_by_epoch.pdf', bbox_inches='tight')

#         plt.figure(2)
#         x = np.arange(1,len(self.rewards)+1)
#         y = np.asarray(self.rewards)
#         plt.plot(x,y, '-')
#         plt.xlabel('Action')
#         plt.ylabel('Reward')
#         plt.savefig(save_name + '_reward_by_action.pdf', bbox_inches='tight')

#         plt.figure(2)
#         x = np.arange(1,len(self.rewards)+1)
#         y = np.asarray(self.rewards)
#         plt.plot(x,y, '-')
#         plt.xlabel('Action')
#         plt.ylabel('Reward')
#         plt.savefig(save_name + '_reward_by_action.pdf', bbox_inches='tight')

#         plt.figure(3)
#         x = np.arange(1,len(self.avg_reward_by_episode)+1)
#         y = np.asarray(self.avg_reward_by_episode)
#         plt.plot(x,y, '-')
#         plt.xlabel('Episode')
#         plt.ylabel('Reward')
#         plt.savefig(save_name + '_reward_by_episode.pdf', bbox_inches='tight')
        
#         pickle.dump(self.perceptron, open(save_name + '_perceptron.p', 'wb'))
#         plt.show()


# --------------------------------------------------------------------------------
if __name__ == '__main__':
    Qlearn().run(SAVE_FILE)

    
