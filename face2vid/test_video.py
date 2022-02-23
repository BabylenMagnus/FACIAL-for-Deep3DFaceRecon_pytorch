from options.test_options import TestOptions
from data.data_loader import create_data_loader
from models.models import create_model
import util.util as util
import cv2
from cv2 import VideoWriter_fourcc
import os
import glob
from os.path import join

opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = create_data_loader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#test images = %d' % dataset_size)
model = create_model(opt).cuda()
if opt.verbose:
    print(model)

from skimage.io import imsave

img_root = '../examples/test_image/' + opt.test_id_name
if not os.path.exists(img_root):
    os.makedirs(img_root)

for i, data in enumerate(dataset):
    if not i % 50:
        print(i)
    label = data['label']
    cur_frame = model.inference(label)
    prev_frame = cur_frame.data[0]

    if i + 7 <= len(dataset):
        frame_index = (i + 7)
    else:
        frame_index = i + 7 - len(dataset)
    imsave(img_root + '/{:06d}.jpg'.format(frame_index), util.tensor2im(prev_frame))

fps = 30

fourcc = VideoWriter_fourcc('M', 'J', 'P', 'G')
videoWriter = cv2.VideoWriter(img_root + '/test_1.avi', fourcc, fps, (512, 512))
im_names = os.listdir(img_root)
im_names = sorted(glob.glob(join(img_root, '*[0-9]*.jpg')))

for im_name in range(len(im_names)):
    frame = cv2.imread(im_names[im_name])

    frame = cv2.resize(frame, (512, 512))
    print(im_name)
    videoWriter.write(frame)
print(videoWriter)
videoWriter.release()
