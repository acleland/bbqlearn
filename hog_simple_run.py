from qlearn import *
from HOG_Env import *
from tools import *
import random
import pickle

qlearn = Qlearn(HOG_Env())
train_list = get_labels(1, 390)
tlist = random.sample(train_list, 5)
t = time.time()
qlearn.run('hog_simple_run', tlist, 
    num_epochs=2, 
    actions_per_episode=15,
    visual = False)
dt = time.time() - t
print('\ntime', dt)
print('train list', tlist)

