from bbqlearn import *

# --------------------------------------------------------------------------------
# Q Testing (try to improve bounding boxes with trained q perceptron)

class Qtest:
    def __init__(self, perceptron):
        self.perceptron = perceptron

    def adjust_box(self, image, box, n=ACTIONS_PER_EPISODE, printing=True):
        state = State2(image, box)
        for _ in range(n):
            self.one_step(state, printing)
        return state.box

    def one_step(self, state, printing=True):
        # Side effect alert: this function will modify state by taking a best action according to the q function
        # Get best action according to q function
        s = state.vector
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
        

def test(perceptron, test_label_files, img_path=TRAIN_PATH):
    tester = Qtest(perceptron)
    initial_ious = []
    adjusted_ious = []
    changes = []

    for testfile in test_label_files:
        imgf, bb, gt = parse_label(read_label(img_path + testfile + '.labl'))
        initial_iou = bb.iou(gt)
        initial_ious.append(initial_iou)
        image = load_image(img_path + imgf + '.jpg')
        print('Image file', imgf)
        print('Initial Box', bb, 'Ground truth', gt, 'IOU', initial_iou)
        print('Getting new box...')
        adjusted_box = tester.adjust_box(image, bb, n=ACTIONS_PER_EPISODE, printing=True)
        new_iou = adjusted_box.iou(gt)
        adjusted_ious.append(new_iou)
        change = new_iou - initial_iou
        changes.append(change)
        percent_change = (new_iou - initial_iou)/initial_iou
        print('New box', adjusted_box, 'Adjusted IOU', new_iou, 'Percent change', percent_change)

    avg_change = np.mean(changes)
    print('Average change in IOU:', avg_change)
    print('Average initial IOU', np.mean(initial_ious))
    print('Average final IOU', np.mean(adjusted_ious))




    return initial_ious, adjusted_ious


# --------------------------------------------------------------------------------


if __name__ == '__main__':

    five_epochs = pickle.load(open('Run_Data/five_epochs.p', 'rb'))

    testlist = five_epochs.train_list[:1]
    print(testlist)

    init_ious, final_ious = test(five_epochs.perceptron, testlist)


