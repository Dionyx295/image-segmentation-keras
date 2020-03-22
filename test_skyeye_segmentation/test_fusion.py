import pytest
import os
import shutil
from PIL import Image

from skyeye_segmentation.controller.skyeye_func import MaskFusionWorker

def test_fusion():
    classes_folders = ["test_data/dataset/charb/", "test_data/dataset/talus/"]
    scales = [1, 2]
    save = "test_fusion/"

    if os.path.exists(save):
        shutil.rmtree(save)
    os.mkdir(save)

    worker = MaskFusionWorker(class_pathes=classes_folders,
                              class_scales=scales,
                              size=(256, 256),
                              save_to=save)

    worker.run()

    assert os.path.exists(save)

    files = os.listdir(save)
    assert len(files) == 4
    assert files[0] == "1.png"
    assert files[1] == "8.png"
    assert files[2] == "z03.png"
    assert files[3] == "z09.png"

    im = Image.open("test_fusion/1.png", "r")
    assert im.size[0] == 256
    assert im.size[1] == 256

    pixels = list(im.getdata())
    assert pixels[0] == 0
    assert pixels[163] == 1

    if os.path.exists(save):
        shutil.rmtree(save)

