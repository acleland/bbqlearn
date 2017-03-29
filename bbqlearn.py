import numpy as np 
import matplotlib.pyplot as plt
import pickle

# Parameters
NumEpochs = 1000
NumTrainingExamples = 100
NumShiftsPerCoord = 5
ShiftRange = .5
NumActionsPerEpisode = 10
step_size = 0.01
discount_factor = 0.5
learning_rate = 0.2
printing = False



# Actions
def left(p):
    return p + np.array([-step_size, 0])
def right(p):
    return p + np.array([step_size, 0])
def up(p):
    return p + np.array([0, step_size])
def down(p):
    return p + np.array([0, -step_size])
def stop(p):
    return p
actions = [left, right, up, down]

# Distance Functions
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
    return 1/(1+np.exp(-z))


# Code for creating shifted points
def get_data(ground_truth_coords, num_shifts_per_coord=5, delta_range=5):
    """returns data as np-array of ground-truth-coord, shifted-coord pairs"""
    # data[i,0] = ground truth coord of point i. data[i, 1] = shifted coord of point i.
    data = []
    for coord in ground_truth_coords:
        for _ in range(num_shifts_per_coord):
            data.append([coord, coord + np.random.rand(2)*2*delta_range - delta_range])
    return np.asarray(data)

# --------------------------------------------------------------------------------

class Q_Perceptron:
    def __init__(self, num_actions, input_vector_length, learning_rate):
        self.weights = np.random.rand(num_actions, input_vector_length)
        self.learning_rate = learning_rate
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
    perceptron = Q_Perceptron(len(actions), 2, .5)
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

class Box:
    def __init__(self, x, y, width, height):
        self.vector = np.array([x,y,x+width-1, y+height-1])
        self.width = width
        self.height = height
    def area(self):
        return self.width*self.height
    def center(self):
        return self.vector[:2] + (self.vector[2:]+1)/2
    def iou(self, other):
        max_topleft_x = max(self.vector[0], other.vector[0])
        max_topleft_y = max(self.vector[1], other.vector[1])
        min_bottomright_x = min(self.vector[2], other.vector[2])
        min_bottomright_y = min(self.vector[3], other.vector[3])
        intersection_area = max(0, (min_bottomright_x - max_topleft_x + 1)*(min_bottomright_y - max_topleft_y + 1))
        return intersection_area/(self.area() + other.area() - intersection_area)
    def __str__(self):
        return str(self.vector)

def testBox():
    box1 = Box(0,0,6,5)
    box2 = Box(3,2,4,6)
    box3 = Box(4,7,1,2)
    box4 = Box(1,1,1,1)
    box5 = Box(1,-1,1,1)
    box6 = Box(-2,0,1,1)
    box7 = Box(7,0,1,1)
    box8 = Box(-1,-1,1,1)
    print("box1:", box1)
    print("box2: ", box2.vector)
    print("box3: ", box3.vector)
    print("box4: ", box4.vector)
    print("box5: ", box5.vector)
    print("box6: ", box6.vector)
    print("box7: ", box7.vector)
    print("box8: ", box8.vector)
    print("IOU(1,2) = 0.2: ", box1.iou(box2))
    print("IOU(1,3) = 0.0: ", box1.iou(box3))
    print("IOU(2,3) = 0.04: ", box2.iou(box3))
    print("IOU(1,4) = 0.33: ", box1.iou(box4))
    print("IOU(1,5) = 0.0: ", box5.iou(box1))
    print("IOU(1,6) = 0.0: ", box1.iou(box6))
    print("IOU(1,7) = 0.0: ", box1.iou(box7))  
    print("IOU(1,8) = 0.0: ", box1.iou(box8)) 

# --------------------------------------------------------------------------------

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

# Q-learning algorithm
def qlearn(training_data, NumEpochs, printing=True):
    perceptron = Q_Perceptron(num_actions=4, input_vector_length=2, learning_rate=0.2) 
    states = np.copy(training_data[:,1])

    ground_truths = training_data[:,0]
    sses = [sse(states, ground_truths)]
    actions_taken = 0

    for epoch_number in range(NumEpochs):
        print("Epoch: ", epoch_number+1, "of", NumEpochs)
        for example_num in range(len(training_data)):
            example = training_data[example_num]
            s = states[example_num]
            s_prime = s
            ground_truth = example[0]
            for action_number in range(NumActionsPerEpisode):
                s = states[example_num]
                d = dist(ground_truth, s)
                actions_taken += 1

                # Select action at random
                Qs = perceptron.getQvector(s)
                best_action = np.argmax(Qs)
                action_index = epsilon_choose(len(actions), best_action, epsilon=(1-epoch_number/NumEpochs)) 
                a = actions[action_index] 

                # Compute Q(s,a)
                Qsa = perceptron.getQ(s, action_index)

                # Take action to get new state s'
                s_prime = a(s)
                d_prime = dist(ground_truth, s_prime)
                r = np.sign(d - d_prime)  # immediate reward

                # Compute Q(s',a') for all a' in actions
                Qs_prime = perceptron.getQvector(s_prime)

                # Determine output, apply perceptron learning rule
                y = r
                if action_number + 1 < NumActionsPerEpisode:
                    y += discount_factor*np.max(Qs_prime)
                
                perceptron.update_weights(y, Qsa, s, action_index)

                # Update state
                states[example_num] = s_prime

                # Compute sum square errors
                sses.append(sse(states, ground_truths))

                
                if printing:
                    #print("Epoch: ", epoch_number+1, "of", NumEpochs)
                    print("Action number: ", (action_number + 1) , "of", NumActionsPerEpisode)
                    print("Actions taken: ", actions_taken)
                    print("ground truth: ", ground_truth)
                    print("s: ",  s)
                    print("distance: ", d)
                    print("action: ", a.__name__)
                    print("Q(s,a) = ", Qsa)
                    print("s' = ", s_prime)
                    print("new distance: ", d_prime)
                    print("immediate reward: ", r)
                    print("total reward y: ", y)
                    print("Q(s', a') for each a' = ")
                    for i in range(len(actions)):
                        print("\t", actions[i].__name__, "\t", Qs_prime[i])
                    print()
                

    return states, perceptron.weights, sses

# --------------------------------------------------------------------------------
def plot_sse_by_actions(fig_num, sses):
    plt.figure(1)
    plt.plot(np.arange(len(sses)), sses, '-b')
    plt.savefig('sse')
    

def plot_before_after(fig_num, ground_truths, original_shifts, final_coords):
    plt.figure(2)
    plt.plot(ground_truths[:,0], ground_truths[:,1], 'b.', 
        original_shifts[:,0], original_shifts[:,1], 'r.',
        final_coords[:,0], final_coords[:,1], 'g.')
    plt.savefig('before_after')
    

def plot_Q(fignum, weights, action_index, var_index):
    x = np.arange(0,1,.01)
    Q = weights[action_index, var_index]*x
    plt.figure(fignum)
    plt.plot(x, Q, 'b-')


# --------------------------------------------------------------------------------

def run_and_plot():
    # Initialize data
    training_data = pickle.load(open("training_data.p", "rb"))
    ground_truth_coords = pickle.load(open("ground_truth_coords.p","rb"))

    states, weights, sses = qlearn(training_data, NumEpochs, printing)

    plot_sse_by_actions(1, sses)
    plot_before_after(2, ground_truth_coords, training_data[:,1], states)
    plot_Q(3, weights, 0, 0)
    plt.show()

# --------------------------------------------------------------------------------

#run_and_plot()

#testPerceptron()
#epsilon_choose_test(.7)
testBox()