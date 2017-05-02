import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils
import skimage
import skimage.io
import glob
import re 
import time

TRAIN_PATH = "../Data/Train/"
OUTPUT_PATH = "../Output/"

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

# --------------------------------------------------------------------------------

class Perceptron:
    def __init__(self, num_actions, input_vector_length, learning_rate):
        self.weights = np.random.rand(num_actions, input_vector_length)*2 - 1
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
        #self.vector = np.array([x,y,x+width-1, y+height-1])
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    @staticmethod
    def fromCenter(center_x, center_y, width, height):
        return Box(int(np.round(center_x - width/2)), int(np.round(center_y-height/2)), width, height)

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
    def zoom(self,factor):
        center_x, center_y = self.center()
        return Box.fromCenter(center_x,center_y, self.width*factor, self.height*factor)


    def __str__(self):
        return str((self.x, self.y, self.width, self.height))

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
    print("box2: ", box2)
    print("box3: ", box3)
    print("box4: ", box4)
    print("box5: ", box5)
    print("box6: ", box6)
    print("box7: ", box7)
    print("box8: ", box8)
    print("IOU(1,2) = 0.2: ", box1.iou(box2))
    print("IOU(1,3) = 0.0: ", box1.iou(box3))
    print("IOU(2,3) = 0.04: ", box2.iou(box3))
    print("IOU(1,4) = 0.033: ", box1.iou(box4))
    print("IOU(1,5) = 0.0: ", box5.iou(box1))
    print("IOU(1,6) = 0.0: ", box1.iou(box6))
    print("IOU(1,7) = 0.0: ", box1.iou(box7))  
    print("IOU(1,8) = 0.0: ", box1.iou(box8)) 

# --------------------------------------------------------------------------------

def get_crop(image, x, y, width, height):
    return image[x:x+width, y:y+height,:]

def get_crop_box(image, box):
    return get_crop(image, box.x, box.y, box.width, box.height)

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

def test_my_square():
    filenames = get_filenames()
    img, bb, gt = parse_training_label(filenames[0])


# --------------------------------------------------------------------------------

class VggHandler:
    def __init__(self):
        self.vgg = vgg16.Vgg16()
        self.image_holder = tf.placeholder("float", [1, 224, 224, 3])
        self.vgg.build(self.image_holder)

    def get_fc6(self, image):
        vgg = self.vgg
        image = skimage.transform.resize(image, (224, 224), mode='constant')
        image = image.reshape((1,224,224,3))
        with tf.Session() as sess:
            feed_dict = {self.image_holder: image}
            fc6 = sess.run(vgg.fc6, feed_dict=feed_dict)
        return fc6.reshape(fc6.shape[1])

class State:
    def __init__(self, vgg_handler, image, box, dx=10, zoom_frac=0.1):
        self.box = box
        self.image = image
        self.vgg = vgg_handler
        self.actions = [self.left, self.right, self.up, self.down, self.bigger, self.smaller, self.fatter, self.taller, self.stop]
        self.dx = dx
        self.zoom_frac = zoom_frac
        self.done = False
        # 4096 features + history features

    @staticmethod
    def fromCoords(vgg_handler, image, x, y, w, h, dx=10, zoom_frac=0.1):
        return State(vgg_handler, image, Box(x,y,w,h), dx, zoom_frac)

    @staticmethod
    def fromFile(vgg_handler, image_filename, box, dx=10, zoom_frac=0.1):
        image = skimage.io.imread('image_filename')/255.0
        return State(vgg_handler, image, box, dx, zoom_frac)

    def get_features(self):
        crop = get_crop(self.image, self.box.x, self.box.y, self.box.width, self.box.height)
        return self.vgg.get_fc6(crop)

    # Actions:
    def left(self):
        self.box.x -= self.dx
    def right(self):
        self.box.x += self.dx
    def up(self):
        self.box.y -= self.dy 
    def down(self):
        self.box.y += self.dy
    def bigger(self):
        self.box = self.box.zoom(1.0+zoom_frac)
    def smaller(self):
        self.box = self.box.zoom(1.0-zoom_frac)
    def fatter(self):
        self.box = self.box.adjust_width(1+zoom_frac)
    def taller(self):
        self.box = self.box.adjust_height(1+zoom_frac)
    def stop(self):
        self.done = True

    def take_action(action_number):
        self.actions[action_number]

def testState():
    img = skimage.io.imread("pdw1.jpg")
    img = img / 255.0
    vgg_handler = VggHandler()
    state = State.fromCoords(vgg_handler, img, 1605, 346, 222, 126)
    print("bounding box:",state.box)
    features = state.get_features()
    print("features:",features)
    print("features shape:", features.shape)



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

def get_filenames():
    fnames = []
    for labelFile in glob.glob(TRAIN_PATH + '*.labl'):
        labelName = re.search('Train/(.+)\.labl', labelFile).group(1)
        fnames.append(labelName)
    return fnames

def parse_training_label(filename):
    with open(TRAIN_PATH + filename + '.labl', 'r') as f:
        label = f.read()
    label = [int(i) for i in label.split('|')]
    bounding_box = Box(label[1], label[0], label[3], label[2])
    ground_truth = Box(label[5], label[4], label[7], label[6])
    image_filename = re.search('(pdw[0-9])(.+)', filename).group(1)
    image_filename += '.jpg'
    return image_filename, bounding_box, ground_truth

def get_from_file(filename):
    with open(TRAIN_PATH + filename + '.labl', 'r') as f:
        label = f.read()
    label = [int(i) for i in label.split('|')]
    image_filename = re.search('(pdw[0-9])(.+)', filename).group(1)
    image_filename += '.jpg'
    return image_filename, label


def get_from_fnames(fnames):
    image_files = []
    labels = []
    label_filenames = fnames
    for label_file in label_filenames:
        image_filename, label = get_from_file(label_file)
        image_filename = TRAIN_PATH + image_filename
        image_files.append(image_filename)
        labels.append(label)
    return image_files, labels

def read_from_queue(input_queue):
    label = input_queue[1]
    #image_reader = tf.WholeFileReader()
    #_, image_file = image_reader.read(input_queue[0])
    image_file = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_file, channels=3)
    image = tf.image.resize_images(image, [224,224])
    return image, label

def get_train_data():
    label_filenames = get_filenames()
    return ([parse_training_label(filename) for filename in label_filenames])

def save_training_features(filename):
    train_data = get_train_data()
    vgg_handler = VggHandler()
    #print(train_data)
    features = []
    count = 1
    for img_name, bb, gt in train_data[:10]:
        print(count)
        img = skimage.io.imread(TRAIN_PATH + img_name)/255.0
        t = time.time()
        state = State(vgg_handler, img, bb)
        features.append(state.get_features())
        print("time:", time.time() - t)
        count +=1
    np.save(filename, features)


# --------------------------------------------------------------------------------

# Q-learning algorithm

def qlearn(training_data, NumEpochs, printing=True):
    pass

def qlearn_old(training_data, NumEpochs, printing=True):
    perceptron = Perceptron(num_actions=4, input_vector_length=2, learning_rate=0.2) 
    states = np.copy(training_data[:,1])

    ground_truths = training_data[:,0]
    sses = [sse(states, ground_truths)]
    actions_taken = 0

    for epoch_number in range(NumEpochs):
        print("Epoch: ", epoch_number+1, "of", NumEpochs)
        for example_num in range(len(training_data)):
            # load example image from file 
            # initialize ground truth from label
            # initialize state from image label

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

def show_image(img, fig=0):
    plt.figure(fig)
    plt.imshow(img)
    plt.show()


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



# --------------------------------------------------------------------------------

if __name__ == '__main__':
#run_and_plot()
    #testBox()
    #testPerceptron2()
    #epsilon_choose_test(.7)
    #testState()
    save_training_features('first10features_warp')
    