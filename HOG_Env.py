import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
from scipy.special import expit
from skimage import feature, color, exposure
import sys
from qlearn import *
from tools import *


# Some constants



actions = ['left','right','up','down','bigger','smaller','fatter','taller','stop']
NUM_ACTIONS = len(actions)
HISTORY_LENGTH = 10
IMAGE_SIZE = (128, 128)
ORIENTATIONS = 9
PIXELS_PER_CELL = (16, 16)
CELLS_PER_BLOCK = (3,3)
BLOCK_NORM = 'L2'

def num_features(w,pbc,cpb,bins):
    return np.power(cpb,2)*bins*np.power(w//pbc - cpb + 1, 2)

NUM_FEATURES = num_features(IMAGE_SIZE[0],PIXELS_PER_CELL[0],CELLS_PER_BLOCK[0],ORIENTATIONS)
STATE_LENGTH = NUM_FEATURES + NUM_ACTIONS*HISTORY_LENGTH

SHIFT_FRAC = 0.1  # Fraction of width (height) shifted left/right (up/down)
ZOOM_FRAC = 0.1  # Fraction box zooms by in bigger, smaller, taller, fatter actions

PRINTING = False



# HOG Stuff
# --------------------------------------------------------------------------------
def hog(img, visual):
    resized = resize(img, IMAGE_SIZE)
    gray = color.rgb2gray(np.array(resized))
    return feature.hog(gray, orientations = ORIENTATIONS,
                        pixels_per_cell = PIXELS_PER_CELL,
                        cells_per_block = CELLS_PER_BLOCK,
                        feature_vector = True,
                        block_norm = BLOCK_NORM,
                        visualise = visual)

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

        self.done = False

    def __str__(self):
        return self.image_name + '\n' + str(self.box)

    def get_features(self,visual=False):
        crop = get_crop(self.image, self.box)
        return hog(crop, visual)

    def get_vector(self):
        features = self.get_features()
        return np.concatenate((features, np.concatenate(self.history)))

    def get_skew(self):
        return self.image.crop(self.box.toVector2())

    # Actions: (Don't call these explicitly, otherwise, state history  and features 
    #           won't update correctly. Instead use state.take_action(action_number).)
    def left(self):
        self.box = self.box.adjust_x(-SHIFT_FRAC * self.box.width)
    def right(self):
        self.box = self.box.adjust_x(SHIFT_FRAC * self.box.width)
    def up(self):
        self.box = self.box.adjust_y(-SHIFT_FRAC * self.box.height) 
    def down(self):
        self.box = self.box.adjust_y(SHIFT_FRAC * self.box.height) 
    def bigger(self):
        self.box = self.box.zoom(1.0+ZOOM_FRAC)
    def smaller(self):
        self.box = self.box.zoom(1.0-ZOOM_FRAC)
    def fatter(self):
        self.box = self.box.adjust_width(1+ZOOM_FRAC)
    def taller(self):
        self.box = self.box.adjust_height(1+ZOOM_FRAC)
    def stop(self):
        self.done = True

    def take_action(self, action_number):
        self.actions[action_number]()
        self.box = self.box.round()
        self.history.appendleft(onehot(action_number, self.num_actions))
        self.readable_history.appendleft(self.actions[action_number].__name__)

# --------------------------------------------------------------------------------

# class HOG_Env:
#     def __init__(self, image_path=IMAGE_PATH, label_path = LABEL_PATH):
#         self.actions = actions
#         self.num_actions = num_actions
#         self.num_features = NUM_FEATURES
#         self.state_length = STATE_LENGTH
#         self.state = None
#         self.episode = None
#         self.label_path = label_path
#         self.image_path = image_path    

#     def load(self, example):
#         imgf, bb, self.gt = parse_label(read_label(self.label_path + example + '.labl'))
#         img = load_image(self.image_path + imgf + '.jpg')
#         size = img.size
#         self.state = State(img, bb, HISTORY_LENGTH)
#         self.episode = (self.gt, [bb])


#     def get_state(self):
#         return self.state.get_vector()

#     def take_action(self, a):
#         iou = self.state.box.iou(self.gt)
#         self.state.take_action(a)
#         self.episode[1].append(self.state.box)
#         new_iou = self.state.box.iou(self.gt)
#         r = np.sign(new_iou-iou)
#         return self.state.get_vector(), r

#     def show(self):
#         gt = self.episode[0]
#         hist = self.episode[1]
#         boxes = [gt]
#         boxes.append(hist[-1])
#         colors = ['y', 'r']
#         if len(hist) > 1:
#             boxes.append(hist[-2])
#             colors.append('w')
#         plot_img_boxes(self.state.image, boxes, colors)
#         show_hog(get_crop(self.state.image, self.state.box))
        



# --------------------------------------------------------------------------------

def plot_img_boxes(img, boxes, colors=None):
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    if colors is None:
        colors = ['r']*len(boxes)
   
    for box, col in zip(boxes, colors):
        rect = patches.Rectangle((box.x,box.y),box.width,box.height, linewidth=2,edgecolor=col,facecolor='none')
        ax.add_patch(rect)
    box_lefts = [box.x for box in boxes]
    box_tops = [box.y for box in boxes]
    box_rights = [box.x + box.width for box in boxes]
    box_bottoms = [box.y + box.height for box in boxes]
    margin = 100
    img_box = img.getbbox()
    left_xlim = max(0, min(box_lefts) - margin)
    right_xlim = min(img_box[2], max(box_rights) + margin)
    top_ylim = max(0, min(box_tops) - margin)
    bottom_ylim = min(img_box[3], max(box_bottoms) + margin)
    ax.set_xlim(left_xlim, right_xlim)
    ax.set_ylim(bottom_ylim, top_ylim)
    return ax
    #plt.show()

def show_hog(image):
    resized = resize(image, IMAGE_SIZE)
    gray = color.rgb2gray(np.array(resized))
    fd, hog_image = feature.hog(gray, orientations = ORIENTATIONS,
                        pixels_per_cell = PIXELS_PER_CELL,
                        cells_per_block = CELLS_PER_BLOCK,
                        feature_vector = True,
                        block_norm = BLOCK_NORM,
                        visualise = True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(gray, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    #hog_image_rescaled = hog_image

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()
    
