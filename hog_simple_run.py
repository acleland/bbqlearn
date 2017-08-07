from qlearn import *
from HOG_Env import *
from tools import *

qlearn = Qlearn(HOG_Env())
train_list, _ = get_train_validation()
t = time.time()
qlearn.run('hog_full_train_set_epochs100_ape15', train_list, 
    num_epochs=100, 
    actions_per_episode=15,
    visual = False)
dt = time.time() - t
print('\ntime', dt)

