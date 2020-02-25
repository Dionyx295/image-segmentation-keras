import os
import random
import numpy as np
import skimage.io as io
import skimage.transform as trans
from matplotlib import pyplot
from skimage import img_as_ubyte
from PIL import Image
from PyQt5.QtCore import QRunnable, pyqtSlot
from keras.preprocessing.image import ImageDataGenerator

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

'''
    Worker wrapper for the image augmentation func
'''
class ImageAugmentationWorker(QRunnable):

    def __init__(self, *args, **kwargs):
        super(ImageAugmentationWorker, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        self.augment_data(**self.kwargs)

    '''
        Data augmentation of images and masks using keras ImageDataGenerator
    '''
    def augment_data(self, nb_img=1, img_src="", seg_src="", img_dest="", seg_dest="", size=(10,10),
                     rotation=90, width=0.25, height=0.25, shear=10, zoom=0.1, fill='reflect'):
        image_gen = ImageDataGenerator(rotation_range=rotation,
                                       width_shift_range=width,
                                       height_shift_range=height,
                                       shear_range=shear,
                                       zoom_range=zoom,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode=fill,
                                       dtype="uint8")

        rand_seed = random.randint(1, 9999999)

        classes_path = os.path.basename(img_src)
        classes_dir = os.path.dirname(img_src)
        img_generator = image_gen.flow_from_directory(classes_dir,
                                                      size,
                                                      'rgb',
                                                      classes=[classes_path],
                                                      class_mode='categorical',
                                                      batch_size=1,
                                                      shuffle=False,
                                                      seed=rand_seed,
                                                      save_to_dir=None,
                                                      save_prefix='',
                                                      save_format='png',
                                                      follow_links=False,
                                                      subset=None,
                                                      interpolation='nearest')
        classes_path = os.path.basename(seg_src)
        classes_dir = os.path.dirname(seg_src)
        mask_generator = image_gen.flow_from_directory(classes_dir,
                                                       size,
                                                       'rgb',
                                                       classes=[classes_path],
                                                       class_mode='categorical',
                                                       batch_size=1,
                                                       shuffle=False,
                                                       seed=rand_seed,
                                                       save_to_dir=None,
                                                       save_prefix='',
                                                       save_format='png',
                                                       follow_links=False,
                                                       subset=None,
                                                       interpolation='nearest')

        file_processed=0

        ## Manual saving for uint8 conversion
        # Img
        fig = pyplot.figure(figsize=(8, 8))
        for i in range(1, nb_img + 1):
            img = img_generator.next()
            image = img[0][0].astype('uint8')
            pyplot.imsave(img_dest + "/" + str(i) + ".png", image)
            file_processed += 1
            progression = (100*file_processed)/(2*nb_img)
            self.signals.progressed.emit(progression)

        # Masks
        fig = pyplot.figure(figsize=(8, 8))
        for i in range(1, nb_img + 1):
            img = mask_generator.next()
            image = img[0][0].astype('uint8')
            pyplot.imsave(seg_dest + "/" + str(i) + ".png", image)
            file_processed += 1
            progression = (100 * file_processed) / (2 * nb_img)
            self.signals.progressed.emit(progression)

        self.signals.finished.emit("Augmentation terminée !")