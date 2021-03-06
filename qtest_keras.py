from bbqlearn_keras import *

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
            # print("new state vector s'", state.vector)
            print("s' box", state.box)
        

def test(perceptron, test_label_files, n=ACTIONS_PER_EPISODE, img_path=TRAIN_PATH, printing=True):
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
        adjusted_box = tester.adjust_box(image, bb, n, printing=printiing)
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



def test_get_ious_boxes(perceptron, test_label_files, n=ACTIONS_PER_EPISODE, img_path=TRAIN_PATH):
    tester = Qtest(perceptron)
    iou_data = []
    changes = []
    initial_ious = []
    adjusted_ious = []
    box_data = []

    for testfile in test_label_files:
        imgf, bb, gt = parse_label(read_label(img_path + testfile + '.labl'))
        original_box = bb.toVector()
        ground_truth = gt.toVector()
        initial_iou = bb.iou(gt)
        image = load_image(img_path + imgf + '.jpg')
        print('Image file', imgf)
        print('Initial Box', bb, 'Ground truth', gt, 'IOU', initial_iou)
        print('Getting new box...')
        adjusted_box = tester.adjust_box(image, bb, n, printing=True)
        final_box = adjusted_box.toVector()
        new_iou = adjusted_box.iou(gt)
        change = new_iou - initial_iou
        percent_change = (new_iou - initial_iou)/initial_iou
        iou_data.append((testfile, initial_iou, new_iou, change))
        changes.append(change)
        initial_ious.append(initial_iou)
        adjusted_ious.append(new_iou)
        box_data.append((testfile, ground_truth, original_box, final_box))
        print('New box', adjusted_box, 'Adjusted IOU', new_iou, 'Percent change', percent_change)

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


# --------------------------------------------------------------------------------


if __name__ == '__main__':
    simpletest2('Run_Data/five_epochs.p', 'simpledata', 'simplefig.pdf', 5)
    



