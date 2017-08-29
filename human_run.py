from qlearn_anneal import *
from HOG_Env import *
from tools import *

qlearn = Qlearn(HOG_Env())
t = time.time()
qlearn.run('human_one_image_100epochs', ['pdw1a'], 
    num_epochs=1000, 
    actions_per_episode=15,
    visual = False)
dt = time.time() - t
print('\ntime', dt)

