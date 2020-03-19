import pytest

from skyeye_segmentation.controller.skyeye_func import EvalWorker


def test_eval():

    im_src = "test_data/img_eval"
    se_src = "test_data/seg_eval"

    check = "test_data/models/model.0"

    worker = EvalWorker(inp_images_dir=im_src, annotations_dir=se_src,
                        checkpoints_path=check)

    worker.run()