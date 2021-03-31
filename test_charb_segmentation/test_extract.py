import pytest
import time
import os
import skimage.io as io
import numpy as np
from pathlib import Path

from skyeye_segmentation.controller.skyeye_func import ExtractionWorker
from skyeye_segmentation.controller.skyeye_func import add_borders


# (not tested) ---------------------------------
def clear_dir(directory):
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isdir(path):
            clear_dir(path)
        else:
            os.unlink(path)
# ----------------------------------------------


def test_add_border(img_size, vig_size, final_size):
    img = np.ones((img_size, img_size))
    with_borders = add_borders(img, vig_size)
    assert with_borders.shape == (final_size, final_size, 1)
    for x in range(with_borders.shape[0]):
        for y in range(with_borders.shape[1]):
            if (x >= vig_size // 2 - 1) and (x < (img_size + (vig_size // 2) - 1)) and (y >= vig_size // 2 - 1) and (
                    y < (img_size + (vig_size // 2) - 1)):
                assert with_borders[x, y] == 1
            else:
                assert with_borders[x, y] == 0
    print(f'[test_add_border] img_size={img_size}, vig_size={vig_size}, final_size={final_size}... OK!')


def test_extract(vig_size, increment, mode, expected_charb, expected_back):
    src_img = "test_img"
    src_seg = "test_seg"
    id_charb = 2
    prop_train = 0.7
    dst = "test_thumbnails"

    if os.path.exists(dst):
        clear_dir(dst)
    Path(dst).mkdir(exist_ok=True)

    start = time.time()

    worker = ExtractionWorker(src_img=src_img, src_seg=src_seg, id_charb=id_charb,
                              prop_train=prop_train,
                              dst=dst, vig_size=vig_size, increment=increment, mode=mode)

    worker.run()

    end = time.time()

    # Test if the execution did not last more than 5 seconds
    assert end - start < 5
    print(f'[test_execution_time] duration = {end-start:.2f}s < 5s... OK!')

    # Test if directories are created
    assert os.path.exists(dst + "/training/charb")
    assert os.path.exists(dst + "/training/back")
    assert os.path.exists(dst + "/validation/charb")
    assert os.path.exists(dst + "/validation/back")
    print(f'[test_directories] directory subtree created... OK!')

    # Test if the extraction worker extract the good number of thumbnails
    charb = os.listdir(dst + "/validation/charb")
    back = os.listdir(dst + "/validation/back")
    assert len(charb) == expected_charb
    assert len(back) == expected_back
    print(f'[test_number_thumbnails_back] expected {expected_back}, found {len(back)}... OK!')
    print(f'[test_number_thumbnails_charb] expected {expected_charb}, found {len(charb)}... OK!')

    # Test if the thumbnails are of the good size
    for file in charb:
        img = io.imread(os.path.join(dst + "/validation/charb/", file), as_gray=True)
        assert img.shape[0] == vig_size
        assert img.shape[1] == vig_size
    print(f'[test_thumbnail_size] expected {vig_size}, verified for each thumbnail... OK!')

    seg = io.imread(src_seg + "/LRM_sol_50cm_98.png")
    seg = seg[:, :, 0]
    seg = add_borders(seg, vig_size)

    # Test if classes were well determined
    for file in charb:
        i1 = file.index('50cm_') + 6
        i2 = file.index('.png')
        data = file[i1:i2]
        data = data.split('_')
        x = int(data[1])
        y = int(data[2])
        if mode == '1px':
            assert seg[x + (vig_size // 2) - 1, y + (vig_size // 2) - 1] == id_charb
        elif mode == '4px':
            center = seg[x + vig_size // 2 - 1:x + vig_size // 2 + 1, y + vig_size // 2 - 1:y + vig_size // 2 + 1]
            is_charb = (np.count_nonzero(center == id_charb) == 4)
            assert is_charb is True
    print("[test_class_charb] verified for each thumbnail... OK!")

    # Test if classes were well determined
    for file in back:
        i1 = file.index('50cm_') + 6
        i2 = file.index('.png')
        data = file[i1:i2]
        data = data.split('_')
        x = int(data[1])
        y = int(data[2])
        if mode == '1px':
            assert seg[x + (vig_size // 2) - 1, y + (vig_size // 2) - 1] != id_charb
        elif mode == '4px':
            center = seg[x + vig_size // 2 - 1:x + vig_size // 2 + 1, y + vig_size // 2 - 1:y + vig_size // 2 + 1]
            is_charb = (np.count_nonzero(center == id_charb) == 4)
            assert is_charb is False

    print("[test_class_back] verified for each thumbnail... OK!")


test_add_border(img_size=8, vig_size=6, final_size=12)
test_add_border(img_size=400, vig_size=32, final_size=430)
test_add_border(img_size=400, vig_size=64, final_size=462)

test_extract(vig_size=32, increment=1, mode='1px', expected_charb=687, expected_back=2061)
test_extract(vig_size=64, increment=2, mode='4px', expected_charb=123, expected_back=369)

print("\n[EXTRACTION] All tests passed with success!\n")
