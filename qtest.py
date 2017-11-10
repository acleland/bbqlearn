#!/usr/local/bin/python3
from qlearn import *

import sys

ACTIONS_PER_EPISODE = 15


# --------------------------------------------------------------------------------
# Q Testing (try to improve bounding boxes with trained q perceptron)

class Qtest:
    def __init__(self, perceptron):
        self.perceptron = perceptron

    def adjust_box(self, image, box, n=ACTIONS_PER_EPISODE, printing=True):
        state = State(image, box)
        for _ in range(n):
            self.one_step(state, printing)
        return state.box

    def one_step(self, state, printing=True):
        # Side effect alert: this function will modify state by taking a best action according to the q function
        # Get best action according to q function
        s = state.get_vector()
        q_s = self.perceptron.getQvector(s)
        action = random_argmax(q_s)
        if printing:
            print('state vector s', s)
            print('state box', state.box)
            print('state history', state.readable_history)
            print('Q(s)', q_s)
            print('best action:', action)
        # Execute best action to get s'
        state.take_action(action)
        if printing:
            print("new state vector s'", state.vector)
            print("s' box", state.box)
        


def test_get_ious_boxes(perceptron, test_label_files, 
                        n=ACTIONS_PER_EPISODE, 
                        img_path=IMAGE_PATH, 
                        label_path = DOGS):
    

    # Set up tester
    tester = Qtest(perceptron)

    # Set up data collectors
    iou_data = []
    changes = []
    initial_ious = []
    adjusted_ious = []
    box_data = []

    count = 0

    for testfile in test_label_files:
        image = load_image(img_path + testfile + '.jpg')
        gt, skews = get_gt_skews(testfile, label_path) 
        count += 1 
        print('Image', count, '/', len(test_label_files))
        episode = 0

        for skew in skews:
            episode += 1

            original_box = skew.toVector()
            ground_truth = gt.toVector()
            initial_iou = skew.iou(gt)
            
            adjusted_box = tester.adjust_box(image, skew, n, printing=False)
            final_box = adjusted_box.toVector()
            new_iou = adjusted_box.iou(gt)
            change = new_iou - initial_iou
            if initial_iou == 0:
                print('zero init iou found in', testfile)
            
            iou_data.append((testfile, initial_iou, new_iou, change))
            changes.append(change)
            initial_ious.append(initial_iou)
            adjusted_ious.append(new_iou)
            box_data.append((testfile, ground_truth, original_box, final_box))
            print('Episode', episode, 'Old IOU', initial_iou, 'New IOU', new_iou, 'Change', change)

    avg_change = np.mean(changes)
    print('Average change in IOU:', avg_change)
    print('Average initial IOU', np.mean(initial_ious))
    print('Average final IOU', np.mean(adjusted_ious))
    iou_data_array = np.asarray(iou_data, dtype=[('fname','|S10'), ('init_iou', 'f8'), ('final_iou', 'f8'), ('change_iou', 'f8')])
    return iou_data_array, box_data



def iou_fig(fig_save_name, init_ious, final_ious):
    plt.plot(init_ious, final_ious, '.', init_ious, init_ious, '-')
    plt.xlabel('Initial IOUs')
    plt.ylabel('Final IOUs')
    plt.savefig(fig_save_name, bbox_inches='tight')
    plt.close()


def success_rate(iou_data):
    count = 0
    for d in iou_data:
        if d[3] > 0:
            count += 1
    return count/len(iou_data)

def get_learning_curve(wbe, epochs, subject):
    test_list = get_imgfiles(401,500)
    if subject.upper() in ['DOGS', 'DOG', 'D']:
        LABEL_SUBJECT = DOGS
    else:
        LABEL_SUBJECT = HUMANS

    suc_rates = []
    for epoch in epochs:
        perc = Perceptron.from_weights(wbe[epoch])
        test_iou_data, test_box_data = test_get_ious_boxes(perc, 
                                                    test_list,
                                                    n=ACTIONS_PER_EPISODE,
                                                    img_path = IMAGE_PATH,
                                                    label_path = LABEL_SUBJECT)
        suc_rates.append(success_rate(test_iou_data))

    return suc_rates

def lc_fig(fig_save_name, epochs, suc_rates):
    plt.plot(epochs, suc_rates)
    plt.xlabel('Epochs trained')
    plt.ylabel('Success Rate')
    plt.savefig(fig_save_name, bbox_inches='tight')
    plt.close()


  

# --------------------------------------------------------------------------------



if __name__ == '__main__':
    subject = sys.argv[1]
    filepath = sys.argv[2]

    if subject.upper() in ['DOGS', 'DOG', 'D']:
        LABEL_SUBJECT = DOGS
    else:
        LABEL_SUBJECT = HUMANS

    perc = Perceptron.load(filepath + 'perceptron.npy')
    train_list = pickle.load(open(filepath + 'train_list.p','rb'))
    test_list = get_imgfiles(401,500)


    test_iou_data, test_box_data = test_get_ious_boxes(perc, 
                                                    test_list,
                                                    n=ACTIONS_PER_EPISODE,
                                                    img_path = IMAGE_PATH,
                                                    label_path = LABEL_SUBJECT)
    test_init_ious = test_iou_data['init_iou']
    test_final_ious = test_iou_data['final_iou']
    iou_fig(filepath + 'test_iou_fig.pdf', test_init_ious, test_final_ious)
    np.save(filepath + 'test_iou_data.npy', test_iou_data)
    pickle.dump(test_box_data, open(filepath + 'test_boxdata.p', 'wb'))

    
    



