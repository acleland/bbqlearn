from qlearn import *


qlearn = Qlearn()
t = time.time()
qlearn.run('one_image_epochs100_ape15', ['pdw1a'], 
    num_epochs=100, 
    actions_per_episode=15,
    visual = False)
dt = time.time() - t
print('\ntime', dt)

