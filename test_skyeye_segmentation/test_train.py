import pytest
import os

from skyeye_segmentation.controller.skyeye_func import TrainWorker


def test_train():

    im_src = "test_img_aug/"
    se_src = "test_seg_aug/"
    check = "test_data/models/test"

    worker = TrainWorker(existing=None, new="fcn_8", width=256,
                         height=256, img_src=im_src,
                         seg_src=se_src, batch=1, steps=1,
                         epochs=1, checkpoint=check,
                         nb_class=2)
    worker.run()
    assert os.path.exists("test_data/models/test.0")
    os.remove("test_data/models/test.0")
    assert os.path.exists("test_data/models/test_config.json")
    os.remove("test_data/models/test_config.json")

    worker = TrainWorker(existing=None, new="unet", width=256,
                         height=256, img_src=im_src,
                         seg_src=se_src, batch=1, steps=1,
                         epochs=1, checkpoint=check,
                         nb_class=2)
    worker.run()
    assert os.path.exists("test_data/models/test.0")
    os.remove("test_data/models/test.0")
    assert os.path.exists("test_data/models/test_config.json")
    os.remove("test_data/models/test_config.json")

    worker = TrainWorker(existing=None, new="segnet", width=256,
                         height=256, img_src=im_src,
                         seg_src=se_src, batch=1, steps=1,
                         epochs=1, checkpoint=check,
                         nb_class=2)
    worker.run()
    assert os.path.exists("test_data/models/test.0")
    os.remove("test_data/models/test.0")
    assert os.path.exists("test_data/models/test_config.json")
    os.remove("test_data/models/test_config.json")