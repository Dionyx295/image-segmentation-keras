import pytest
import os
import shutil
from PIL import Image

from skyeye_segmentation.controller.skyeye_func import PredictWorker


def test_predict():

    im_src = "test_data/img_eval/"

    save = "test_data/predict/"
    if os.path.exists(save):
        shutil.rmtree(save)
    os.mkdir(save)

    check = "test_data/models/model.0"

    worker = PredictWorker(inp_dir=im_src, out_dir=save,
                           checkpoints_path=check, colors=None,
                           sup_dir=save)

    worker.run()

    files = os.listdir(save)

    assert os.path.exists("test_data/predict/1.png")
    assert os.path.exists("test_data/predict/1-sup.png")
    assert os.path.exists("test_data/predict/1_prob_128x128.csv")

    im = Image.open("test_data/predict/1.png", "r")
    assert im.size[0] == 256
    assert im.size[1] == 256

    im = Image.open("test_data/predict/1-sup.png", "r")
    assert im.size[0] == 389
    assert im.size[1] == 389

    assert os.path.exists("test_data/predict/z09.png")
    assert os.path.exists("test_data/predict/z09-sup.png")
    assert os.path.exists("test_data/predict/z09_prob_128x128.csv")

    im = Image.open("test_data/predict/z09.png", "r")
    assert im.size[0] == 241
    assert im.size[1] == 251

    im = Image.open("test_data/predict/z09-sup.png", "r")
    assert im.size[0] == 374
    assert im.size[1] == 389
    im.close()

    if os.path.exists(save):
        shutil.rmtree(save)
