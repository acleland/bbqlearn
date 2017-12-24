from qlearn import *

def eps(epoch):
    return 0.5

t0 = time.time()
print(time.asctime())
tlist = get_imgfiles(1,400)
Qlearn().run(save_path='cropped_de50e',
            train_list = tlist,
            subject = 'dogs',
            num_epochs = 200,
            actions_per_episode = 15,
            learning_rate = 0.2,
            discount_factor = 0.9,
            epsilon_func = eps,
            visual = False)
dt = time.time() - t0
print(time.asctime())
print('run time', dt, 'secs')
