from qlearn import *

import random
import pickle

qlearn = Qlearn()
train_list = get_imgfiles(1, 390)
tlist = random.sample(train_list, 10)
t = time.time()
qlearn.run('new_simple_run', tlist, 
    num_epochs=10, 
    actions_per_episode=15,
    visual = False)
dt = time.time() - t
print('\ntime', dt)
print('train list', tlist)

