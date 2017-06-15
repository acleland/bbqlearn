from bbqlearn import *


five_epochs = pickle.load(open('Run_Data/five_epochs.p', 'rb'))

testlist = five_epochs.train_list[:10]
print(testlist)

init_ious, final_ious = test(five_epochs.perceptron, testlist)


