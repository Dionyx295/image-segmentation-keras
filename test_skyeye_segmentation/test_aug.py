import pytest
import time
import os
import shutil
from PIL import Image

from skyeye_segmentation.controller.skyeye_func import ImageAugmentationWorker


def test_aug():

    im_src = "test_data/img_eval/"
    se_src = "test_data/seg_eval/"

    im_dest = "test_img_aug/"
    if os.path.exists(im_dest):
        shutil.rmtree(im_dest)
    os.mkdir(im_dest)
    se_dest = "test_seg_aug/"
    if os.path.exists(se_dest):
        shutil.rmtree(se_dest)
    os.mkdir(se_dest)

    worker = ImageAugmentationWorker(nb_img=5, img_src=im_src, seg_src=se_src,
                                     img_dest=im_dest, seg_dest=se_dest,
                                     size=(256, 256),
                                     rotation=90, width=0.25, height=0.25,
                                     shear=10, zoom=0.1, fill='reflect')

    worker.run()

    assert os.path.exists(im_dest)
    files = os.listdir(im_dest)
    assert len(files) == 5

    for file in files:
        im = Image.open(os.path.join(im_dest, file), "r")
        assert im.size[0] == 256
        assert im.size[1] == 256

    assert os.path.exists(se_dest)
    files = os.listdir(se_dest)
    assert len(files) == 5

    for file in files:
        im = Image.open(os.path.join(se_dest, file), "r")
        assert im.size[0] == 256
        assert im.size[1] == 256
