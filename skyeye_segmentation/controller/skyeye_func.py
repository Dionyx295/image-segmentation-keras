import os
import numpy as np
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte
from PIL import Image
from PyQt5.QtCore import QRunnable, pyqtSlot, pyqtSignal

from skyeye_segmentation.controller.worker_signals import WorkerSignals

'''
    Worker wrapper for the mask fusion func
'''
class MaskFusionWorker(QRunnable):

    def __init__(self, *args, **kwargs):
        super(MaskFusionWorker, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        self.mask_fusion(**self.kwargs)

    '''
        Fusion of binary masks into a colored unique one
    '''
    def mask_fusion(self, class_pathes="", class_scales="", size=(400,400), save_to=""):
        nb_files = len(os.listdir(class_pathes[0]))
        file_processed = 0

        for file in os.listdir(class_pathes[0]):
            print(class_pathes[0] + file)
            ext = file.split(".")[len(file.split("."))-1]
            ext = ext.lower()
            if ext != "tif" and ext != "png" and ext != "jpg" and ext != "jpeg":
                continue

            mask_array = io.imread(class_pathes[0] + file)
            new_mask = Image.new(mode='L', size=(mask_array.shape[0], mask_array.shape[1]), color="black")
            new_mask_array = np.array(np.transpose(new_mask))

            # For each class
            for path, scale in zip(class_pathes, class_scales):
                mask_array = io.imread(path + file)
                # For each pixel
                for x in range(mask_array.shape[0]):  # Width
                    for y in range(mask_array.shape[1]):  # Height
                        if mask_array[x, y].all() == False:  # Black pixel
                            new_mask_array[x, y] = scale

            new_image = save_to + file.split(".")[0] + ".png"
            new_mask_array = trans.resize(new_mask_array, size, anti_aliasing=False)
            io.imsave(new_image, img_as_ubyte(new_mask_array))
            file_processed += 1
            progression = (int)(file_processed*100/nb_files)
            self.signals.progressed.emit(progression)

        self.signals.finished.emit("Création des masques terminée !")