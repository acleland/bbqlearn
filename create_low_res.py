from PIL import Image
import glob, os, re
TRAIN_PATH = '../Data/Train/'
SAVE_PATH = '../Data/Downsized/'
SCALE = .25

for infile in glob.glob(TRAIN_PATH + "*.jpg"):
    image_name = re.search('Train/(.+)\.jpg', infile).group(1)
    file, ext = os.path.splitext(infile)
    print(image_name)
    im = Image.open(infile)
    w = im.size[0] * SCALE
    h = im.size[1] * SCALE
    im.thumbnail((w,h))
    im.save(SAVE_PATH + image_name + ".jpg", "JPEG")