import pytest
import os
from skimage import io

from skyeye_segmentation.controller.skyeye_func import CharbPredictWorker


# (not tested) ---------------------------------
def clear_dir(directory):
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isdir(path):
            clear_dir(path)
        else:
            os.unlink(path)
# ----------------------------------------------


def test_predict():

    model_name = "test_train/models/vggCharb.h5"
    images_path = "test_predict/img"
    save_seg_path = "test_predict/seg"
    save_sup_path = "test_predict/sup"

    vig_size = 32
    intervalle = 1
    batch_size = 32
    mode = '1px'

    clear_dir(save_seg_path)
    clear_dir(save_sup_path)

    worker = CharbPredictWorker(model_name=model_name, images_path=images_path,
                                save_seg_path=save_seg_path, save_sup_path=save_sup_path,
                                vig_size=vig_size, intervalle=intervalle, batch_size=batch_size, mode=mode)

    worker.run()

    for file in os.listdir(images_path):
        assert os.path.exists("test_predict/seg/"+file)
        assert os.path.exists("test_predict/sup/"+file)
    print(f'[test_paths] segmentations and superpositions created... OK!')

    for file in os.listdir(images_path):
        seg = io.imread("test_predict/seg/"+file)
        sup = io.imread("test_predict/sup/"+file)

        assert seg.shape == (400, 400, 4)
        assert sup.shape == (400, 400, 4)
    print(f'[test_size] expected (400, 400, 4), verified for each image... OK!')


test_predict()

print("\n[PREDICTIONS] All tests passed with success!\n")