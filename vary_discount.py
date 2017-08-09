from qlearn import *
from HOG_Env import *
from tools import *
import random
import pickle

tlist = pickle.load(open('random_100.p','rb'))

t = time.time()
for disc_rate in [.1,.3,.5,.7,.9]:
    print('discount rate', disc_rate)
    print()
    qlearn = Qlearn(HOG_Env())
    qlearn.run('disc_'+str(disc_rate), tlist, 
        num_epochs=2,
        learning_rate=0.2,
        discount_factor=disc_rate, 
        actions_per_episode=2,
        visual = False)
dt = time.time() - t
print('time', dt)
# print('train list', tlist)


