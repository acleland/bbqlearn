from PIL import Image
import glob, os, re
TRAIN_PATH = '../Data/Train/'
SAVE_PATH = '../Data/Downsized/'
SCALE = .25

for infile in glob.glob(TRAIN_PATH + "*.labl"):
    label_name = re.search('Train/(.+)\.labl', infile).group(1)
    with open(infile) as f:
        label = f.read()
    label = label.split('|')
    scaled_label = [label[0]]
    for label_val in label[1:]:
        scaled_label.append(str(round(float(label_val)*SCALE)))
    scaled_label_string = '|'.join(scaled_label)
    with open(SAVE_PATH + label_name + '.labl', 'w') as f:
        f.write(scaled_label_string)
    print(scaled_label_string)
    