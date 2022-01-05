"""Module containing all functionality workers."""

import glob
import json
import math
import os
import random
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import six
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
from PIL import Image
from PyQt5.QtCore import QRunnable, pyqtSlot
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from keras_segmentation.data_utils.data_loader import \
    verify_segmentation_dataset, image_segmentation_generator, \
    get_image_array, class_colors, get_class_colors
from keras_segmentation.models.all_models import model_from_name
from keras_segmentation.models.config import IMAGE_ORDERING
from keras_segmentation.predict import model_from_checkpoint_path, \
    get_pairs_from_paths, get_segmentation_array, predict
from skyeye_segmentation.controller.worker_signals import WorkerSignals
from skyeye_segmentation.skyeye_charb import charb_models


class MaskFusionWorker(QRunnable):
    """
        Worker wrapper for the mask fusion func
    """

    def __init__(self, *args, **kwargs):
        super(MaskFusionWorker, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """Run the functionality, triggered by Qt"""

        try:
            self.mask_fusion(**self.kwargs)
        except Exception as exc:
            self.signals.error.emit(traceback.format_exc())

    def mask_fusion(self, class_pathes="", class_scales="", size=(400, 400),
                    save_to=""):
        """Fusion of binary masks into a colored unique one"""

        nb_files = len(os.listdir(class_pathes[0]))
        file_processed = 0

        for file in os.listdir(class_pathes[0]):
            print(class_pathes[0] + file)
            ext = file.split(".")[len(file.split(".")) - 1]
            ext = ext.lower()
            if ext not in ('tif', 'png', 'jpg', 'jpeg'):
                continue

            mask_array = io.imread(class_pathes[0] + file)
            new_mask = Image.new(mode='L', size=(mask_array.shape[0],
                                                 mask_array.shape[1]),
                                 color="black")
            new_mask_array = np.array(np.transpose(new_mask))

            # For each class
            for path, scale in zip(class_pathes, class_scales):
                path_file = os.path.join(path, file)
                mask_array = io.imread(path_file)
                # For each pixel
                for coord_x in range(mask_array.shape[0]):  # Width
                    for coord_y in range(mask_array.shape[1]):  # Height
                        if mask_array[coord_x, coord_y].all() == 0:  # Black pixel
                            new_mask_array[coord_x, coord_y] = scale

            new_image = os.path.join(save_to, file.split(".")[0] + ".png")
            new_mask_array = trans.resize(new_mask_array, size,
                                          anti_aliasing=False)
            io.imsave(new_image, img_as_ubyte(new_mask_array))
            file_processed += 1
            progression = int(file_processed * 100 / nb_files)
            self.signals.progressed.emit(progression)

        self.signals.finished.emit("Création des masques terminée !")


class ImageAugmentationWorker(QRunnable):
    """Worker wrapper for the image augmentation func"""

    def __init__(self, *args, **kwargs):
        super(ImageAugmentationWorker, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """Run the functionality, triggered by Qt"""

        try:
            self.augment_data(**self.kwargs)
        except Exception as exc:
            self.signals.error.emit(traceback.format_exc())

    def augment_data(self, nb_img=1, img_src="", seg_src="", img_dest="",
                     seg_dest="", size=(10, 10),
                     rotation=90, width=0.25, height=0.25, shear=10, zoom=0.1,
                     fill='reflect'):
        """Data augmentation of imgs and segs using keras ImageDataGenerator"""

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

        file_processed = 0

        # Manual saving for uint8 conversion
        # Img
        for i in range(1, nb_img + 1):
            img = img_generator.next()
            image = img[0][0].astype('uint8')
            pyplot.imsave(img_dest + "/" + str(i) + ".png", image)
            file_processed += 1
            progression = (100 * file_processed) / (2 * nb_img)
            self.signals.progressed.emit(progression)

        # Masks
        for i in range(1, nb_img + 1):
            img = mask_generator.next()
            image = img[0][0].astype('uint8')
            pyplot.imsave(seg_dest + "/" + str(i) + ".png", image)
            file_processed += 1
            progression = (100 * file_processed) / (2 * nb_img)
            self.signals.progressed.emit(progression)

        self.signals.finished.emit("Augmentation terminée !")


def find_latest_checkpoint(checkpoints_path, fail_safe=True):
    """Loads the weights from the latest model in the specified folder."""

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f:
                                       get_epoch_number_from_path(f)
                                       .isdigit()
                                       , all_checkpoint_files))
    if not all_checkpoint_files:
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        return None

    # Find the checkpoint file with the maximum epoch
    lt_checkpoint = max(all_checkpoint_files,
                        key=lambda f: int(get_epoch_number_from_path(f)))
    return lt_checkpoint


class TrainWorker(QRunnable):
    """Worker wrapper for the train func"""

    def __init__(self, *args, **kwargs):
        super(TrainWorker, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self.session = tf.Session()
        self.graph = tf.get_default_graph()

    @pyqtSlot()
    def run(self):
        """Run the functionality, triggered by Qt"""

        try:
            self.train(**self.kwargs)
        except Exception as exc:
            self.signals.error.emit(traceback.format_exc())

    def train(self, existing="", new="", width=0, height=0, img_src="",
              seg_src="", batch=0, steps=0, epochs=0,
              checkpoint="", nb_class=0, validate=False, val_images=None,
              val_annotations=None, val_batch_size=1,
              auto_resume_checkpoint=False, load_weights=None,
              verify_dataset=True, optimizer_name='adadelta',
              do_augment=False):
        """Launches the training process with the train config"""

        with self.graph.as_default():
            with self.session.as_default():
                self.signals.log.emit("Début de la session d'entrainement")

                # Getting model
                if existing:
                    try:
                        checkpoint_nb = existing.split('.')[-1]
                        index = -(int)(len(checkpoint_nb) + 1)
                        existing = existing[0:index]
                        model = model_from_checkpoint_path_nb(existing,
                                                              checkpoint_nb)
                        self.signals.log.emit("Modèle chargé : {}"
                                              .format(existing))
                    except Exception as exc:
                        self.signals.error.emit("Impossible de charger le "
                                                "modèle existant !\n" +
                                                traceback.format_exc())
                        return
                else:
                    try:
                        vgg = new.find("vgg")
                        resnet50 = new.find("resnet50")
                        mobilenet = new.find("mobilenet")
                        pspnet = new.find("pspnet")
                        pspnet_50 = new.find("pspnet_50")
                        pspnet_101 = new.find("pspnet_101")

                        # Specifics models constraints
                        if vgg != -1 or resnet50 != -1:
                            if height % 32 != 0 or width % 32 != 0:
                                self.signals.error.emit("Pour un modèle "
                                                        "vgg/resnet50, "
                                                        "les dimensions "
                                                        "d'entrée doivent être "
                                                        "des multiples de 32.")
                                return
                        if mobilenet != -1:
                            if height != 224 or width != 224:
                                self.signals.error.emit(
                                    "Pour un modèle mobilenet, les dimensions "
                                    "d'entrée doivent être (224,224).")
                                return
                        if pspnet != -1:
                            if pspnet_50 != -1 or pspnet_101 != -1:
                                if not (height == 473 and width == 473) \
                                        and not (height == 713
                                                 and width == 713):
                                    self.signals.error.emit(
                                        "Pour un modèle pspnet_50 ou "
                                        "pspnet_101, les dimensions d'entrée "
                                        "doivent être "
                                        "(473,473) ou (713,713).")
                                    return
                            else:
                                if height % 192 != 0 or width % 192 != 0:
                                    self.signals.error.emit("Pour un modèle "
                                                            "pspnet, les "
                                                            "dimensions "
                                                            "d'entrée doivent "
                                                            "être des multiples"
                                                            " de 192.")
                                    return

                        model = model_from_name[new](nb_class,
                                                     input_height=height,
                                                     input_width=width)
                    except Exception as exc:
                        self.signals.error.emit("Impossible de créer un nouveau"
                                                " modèle {} !\n{}".
                                                format(new, traceback.
                                                       format_exc()))
                        return

                output_width = model.output_width
                output_height = model.output_height

                # Model compilation
                if optimizer_name is not None:
                    # weights = [0.1, 10, 20]
                    # loss_func = weighted_categorical_crossentropy(weights)
                    # print("Weighted loss : " + str(weights))
                    loss_func = "categorical_crossentropy"
                    model.compile(loss=loss_func,
                                  optimizer=optimizer_name,
                                  metrics=['accuracy'])

                if checkpoint is not None:
                    with open(checkpoint + "_config.json", "w") as file:
                        json.dump({
                            "model_class": model.model_name,
                            "n_classes": nb_class,
                            "input_height": height,
                            "input_width": width,
                            "output_height": height,
                            "output_width": width
                        }, file)

                if load_weights is not None and len(load_weights) > 0:
                    print("Loading weights from ", load_weights)
                    model.load_weights(load_weights)

                if auto_resume_checkpoint and (checkpoint is not None):
                    latest_checkpoint = find_latest_checkpoint(checkpoint)
                    if latest_checkpoint is not None:
                        print("Loading the weights from latest checkpoint ",
                              latest_checkpoint)
                        model.load_weights(latest_checkpoint)

                if verify_dataset:
                    print("Verifying training dataset")
                    self.signals.log.emit("Vérification du jeu d'entrainement")
                    verified = verify_segmentation_dataset(img_src, seg_src,
                                                           nb_class)
                    if not verified:
                        self.signals.log.emit("Erreur lors de la vérification"
                                              ", vérifiez le jeu "
                                              "d'entrainement (correspondance"
                                              " image/segmentation, nb de"
                                              " classes, format..).")
                        self.signals.log.emit("")
                        self.signals.error.emit("Erreur lors de la "
                                                "vérification du jeu d'"
                                                "entrainement.")
                        return

                    self.signals.log.emit("Jeu d'entrainement vérifié !")
                    self.signals.log.emit("")
                    if validate:
                        print("Verifying validation dataset")
                        verified = verify_segmentation_dataset(val_images,
                                                               val_annotations,
                                                               nb_class)
                        if not verified:
                            self.signals.log.emit(
                                "Erreur lors de la vérification"
                                ", vérifiez le jeu "
                                "de validation.")
                            self.signals.log.emit("")
                            self.signals.error.emit("Erreur lors de la "
                                                    "vérification du jeu de "
                                                    "de validation "
                                                    "(correspondance image/segm"
                                                    "entation, nb de classes, "
                                                    "format..).")
                            return

                train_gen = image_segmentation_generator(
                    img_src, seg_src, batch, nb_class,
                    height, width, output_height, output_width,
                    do_augment=do_augment)

                if validate:
                    val_gen = image_segmentation_generator(
                        val_images, val_annotations, val_batch_size,
                        nb_class, height, width, output_height, output_width)

                if not validate:
                    for epoch in range(epochs):
                        print("Starting Epoch ", epoch)
                        self.signals.log.emit("Début de l'époque {}".format(epoch))
                        history = model.fit_generator(train_gen, steps,
                                                      epochs=1)
                        msg = ""
                        for key, value in history.history.items():
                            msg += "{}:{}  ".format(str(key), str(value))
                        self.signals.log.emit(msg)

                        if checkpoint is not None:
                            model.save_weights(checkpoint + "." + str(epoch))
                            print("saved ", checkpoint + ".model." + str(epoch))
                            self.signals.log.emit("Modèle sauvegardé : "
                                                  "{}.model.{}"
                                                  .format(checkpoint, str(epoch)))
                        print("Finished Epoch", epoch)
                        self.signals.log.emit("époque {} terminée".format(epoch))
                        self.signals.log.emit("")
                        progression = 100 * (epoch + 1) / epochs
                        self.signals.progressed.emit(progression)
                else:
                    for epoch in range(epochs):
                        print("Starting Epoch ", epoch)
                        self.signals.log.emit("Début de l'époque {}".format(epoch))
                        history = model.fit_generator(train_gen, steps,
                                                      validation_data=val_gen,
                                                      validation_steps=200,
                                                      epochs=1)

                        msg = ""
                        for key, value in history.history.items():
                            msg += "{}:{}  ".format(str(key), str(value))
                        self.signals.log.emit(msg)

                        if checkpoint is not None:
                            model.save_weights(checkpoint + "." + str(epoch))
                            print("saved ", checkpoint + ".model." + str(epoch))
                            self.signals.log.emit("Modèle sauvegardé : "
                                                  "{}.model.{}"
                                                  .format(checkpoint, str(epoch)))
                        print("Finished Epoch", epoch)
                        self.signals.log.emit("époque {} terminée\n".format(epoch))
                        self.signals.log.emit("")
                        progression = 100 * (epoch + 1) / epochs
                        self.signals.progressed.emit(progression)

                self.signals.finished.emit("Entrainement terminé !")


class EvalWorker(QRunnable):
    """Worker wrapper for the evaluate func"""

    def __init__(self, *args, **kwargs):
        super(EvalWorker, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self.session = tf.Session()
        self.graph = tf.get_default_graph()

    @pyqtSlot()
    def run(self):
        """Run the functionality, triggered by Qt"""

        try:
            self.evaluate(**self.kwargs)
        except Exception as exc:
            self.signals.error.emit(traceback.format_exc())

    def evaluate(self, model=None, inp_images=None, annotations=None,
                 inp_images_dir=None, annotations_dir=None,
                 checkpoints_path=None):
        """Evaluate the loaded model for an imgs set and segs"""

        with self.graph.as_default():
            with self.session.as_default():
                self.signals.log.emit("Début de la session d'évaluation")
                if model is None:
                    if checkpoints_path is None:
                        self.signals.log.emit("Impossible de trouver le modèle"
                                              " à évaluer.")
                        self.signals.log.emit("")
                        self.signals.error.emit("Impossible de trouver le "
                                                "modèle à évaluer")
                    try:
                        checkpoint_nb = checkpoints_path.split('.')[-1]
                        index = -(int)(len(checkpoint_nb) + 1)
                        existing = checkpoints_path[0:index]
                        model = model_from_checkpoint_path_nb(existing,
                                                              checkpoint_nb)
                        self.signals.log.emit("Modèle chargé : {}"
                                              .format(checkpoints_path))
                    except Exception as exc:
                        self.signals.finished.emit("Impossible de charger le "
                                                   "modèle existant !" +
                                                   traceback.format_exc())
                        return

                if inp_images is None:
                    paths = get_pairs_from_paths(inp_images_dir,
                                                 annotations_dir)
                    paths = list(zip(*paths))
                    inp_images = list(paths[0])
                    annotations = list(paths[1])

                tpm = np.zeros(model.n_classes)
                fpm = np.zeros(model.n_classes)
                fnm = np.zeros(model.n_classes)
                n_pixels = np.zeros(model.n_classes)

                file_processed = 0
                for inp, ann in tqdm(zip(inp_images, annotations)):
                    pred = predict(model, inp)

                    ground = get_segmentation_array(ann, model.n_classes,
                                                    model.output_width,
                                                    model.output_height,
                                                    no_reshape=True)
                    ground = ground.argmax(-1)

                    pred = pred.flatten()
                    ground = ground.flatten()

                    matrix = confusion_matrix(ground, pred)
                    self.signals.log.emit("Image {}".format(str(inp)))
                    self.signals.log.emit("Matrice de confusion :\n{}\n"
                                          .format(str(matrix)))

                    for cl_i in range(model.n_classes):
                        tpm[cl_i] += np.sum((pred == cl_i) * (ground == cl_i))
                        fpm[cl_i] += np.sum((pred == cl_i) * (ground != cl_i))
                        fnm[cl_i] += np.sum((pred != cl_i) * (ground == cl_i))
                        n_pixels[cl_i] += np.sum(ground == cl_i)

                    file_processed += 1
                    progression = 100 * file_processed / len(inp_images)
                    self.signals.progressed.emit(progression)

                cl_wise_score = tpm / (tpm + fpm + fnm + 0.000000000001)
                n_pixels_norm = n_pixels / np.sum(n_pixels)
                frequency_weighted_iu = np.sum(cl_wise_score * n_pixels_norm)
                mean_iu = np.mean(cl_wise_score)
                self.signals.log.emit("frequency_weighted_IU {}"
                                      .format(str(frequency_weighted_iu)))
                self.signals.log.emit("mean_IU {}".format(str(mean_iu)))
                self.signals.log.emit("class_wise_IU {}"
                                      .format(str(cl_wise_score)))
                self.signals.log.emit("")
                self.signals.finished.emit("Evaluation terminée !")


def create_pascal_label_colormap():
    """Creates the colormap for the superposition"""

    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Convert segs to colors"""
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    colormap = create_pascal_label_colormap()
    if np.max(label) > len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


class PredictWorker(QRunnable):
    """Worker wrapper for the predict func"""

    def __init__(self, *args, **kwargs):
        super(PredictWorker, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self.session = tf.Session()
        self.graph = tf.get_default_graph()

    @pyqtSlot()
    def run(self):
        """Run the functionality, triggered by Qt"""

        try:
            self.predict_multiple(**self.kwargs)
        except Exception as exc:
            self.signals.error.emit(traceback.format_exc())

    def predict(self, model=None, inp=None, out_fname=None,
                checkpoints_path=None, clrs=None, out_prob_file=None):
        """Make prediction from an img and loaded model"""

        if model is None and (checkpoints_path is not None):
            model = model_from_checkpoint_path(checkpoints_path)

        assert inp is not None
        assert isinstance(inp, (np.ndarray, six.string_types)), \
            "Input should be the CV image or the input file name"

        if isinstance(inp, six.string_types):
            inp = cv2.imread(inp)

        assert len(inp.shape) == 3, "Image should be h,w,3 "
        orininal_h = inp.shape[0]
        orininal_w = inp.shape[1]

        output_width = model.output_width
        output_height = model.output_height
        input_width = model.input_width
        input_height = model.input_height
        n_classes = model.n_classes

        img_ar = get_image_array(inp, input_width, input_height,
                                 ordering=IMAGE_ORDERING)
        pred = model.predict(np.array([img_ar]))[0]

        # Creating probabilities file
        if out_prob_file is not None:
            out_prob_file += "_prob_{}x{}.csv".format(output_width,
                                                      output_height)

            with open(out_prob_file, 'w+') as file:
                # Header
                header = "x y "
                for i in range(0, n_classes):
                    header += "C{} ".format(str(i))
                header += "class\n"
                file.write(header)

                # Pixel per pixel
                coord_x = 0
                coord_y = 0
                for pixel in pred:
                    line = "{} {}".format(coord_x, coord_y)
                    for class_prob in pixel:
                        line += " {}".format(str(class_prob))
                    line += " {}".format(str(np.argmax(pixel))) + "\n"

                    file.write(line)

                    coord_x += 1
                    if coord_x >= output_width:
                        coord_x = 0
                        coord_y += 1

        pred = pred.reshape((output_height, output_width, n_classes)).argmax(axis=2)

        seg_img = np.zeros((output_height, output_width, 3))

        if clrs is None:
            #colors = class_colors # old version, difficult to distinguish colors
            colors = get_class_colors(n_classes)
        else:
            colors = clrs

        for color in range(n_classes):
            seg_img[:, :, 0] += ((pred[:, :] == color) * (colors[color][0])) \
                .astype('uint8')
            seg_img[:, :, 1] += ((pred[:, :] == color) * (colors[color][1])) \
                .astype('uint8')
            seg_img[:, :, 2] += ((pred[:, :] == color) * (colors[color][2])) \
                .astype('uint8')

        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

        if out_fname is not None:
            cv2.imwrite(out_fname, seg_img)

        return pred

    def predict_multiple(self, model=None, inps=None, inp_dir=None,
                         out_dir=None, checkpoints_path=None, colors=None,
                         sup_dir=None):
        """Make multiple predictions from an img set"""

        with self.graph.as_default():
            with self.session.as_default():
                if model is None and (checkpoints_path is not None):
                    try:
                        checkpoint_nb = checkpoints_path.split('.')[-1]
                        index = -(int)(len(checkpoint_nb) + 1)
                        existing = checkpoints_path[0:index]
                        model = model_from_checkpoint_path_nb(existing,
                                                              checkpoint_nb)
                        self.signals.log.emit("Modèle chargé : {}"
                                              .format(checkpoints_path))
                    except Exception as exc:
                        self.signals.finished.emit("Impossible de charger le"
                                                   " modèle existant !\n"
                                                   + traceback.format_exc())
                        return None

                if inps is None and (inp_dir is not None):
                    inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + \
                           glob.glob(os.path.join(inp_dir, "*.png")) + \
                           glob.glob(os.path.join(inp_dir, "*.jpeg")) + \
                           glob.glob(os.path.join(inp_dir, "*.tif"))

                assert isinstance(inps, list)
                all_prs = []

                files_nb = len(inps)
                file_processed = 0
                self.signals.log.emit("Prédiction de {} images..."
                                      .format(str(files_nb)))
                for i, inp in enumerate(tqdm(inps)):
                    if out_dir is None:
                        out_fname = None
                    else:
                        """
                        if isinstance(inp, six.string_types):
                            out_fname = os.path.join(out_dir,
                                                     os.path.basename(inp))
                        else:
                            out_fname = os.path.join(out_dir, str(i) + ".jpg")
                        """
                        out_fname = os.path \
                            .join(out_dir, os.path
                                  .splitext(os.path.basename(inp))[0] + ".png")

                    out_prob = os.path.splitext(out_fname)[0]
                    pred = self.predict(model, inp, out_fname, clrs=colors,
                                        out_prob_file=out_prob)

                    all_prs.append(pred)

                    file_processed += 1
                    progression = 100 * file_processed / (files_nb * 2)
                    self.signals.progressed.emit(progression)
                    self.signals.log.emit("{}".format(out_fname))

                self.create_superpositions(img_src=inp_dir, seg_src=out_dir,
                                           save_dir=sup_dir)

                self.signals.log.emit("")
                self.signals.finished.emit("Prédictions terminées !")
                return all_prs

    def create_superpositions(self, img_src, seg_src, save_dir):
        """Creates the superpositions images"""

        files_nb = len(os.listdir(seg_src))
        files_processed = 0
        self.signals.log.emit("Création des {} superpositions..."
                              .format(str(files_nb)))
        files = os.listdir(img_src)
        for filename in files:
            imgfile = os.path.join(img_src, filename)
            pngfile = os.path.join(seg_src, filename)
            img = cv2.imread(imgfile, 1)
            img = img[:, :, ::-1]
            seg_map = cv2.imread(pngfile, 0)
            seg_image = label_to_color_image(seg_map).astype(np.uint8)
            saved_img = os.path.join(save_dir, os.path.splitext(filename)[0] +
                                     "-sup.png")
            pyplot.figure()
            pyplot.imshow(seg_image, cmap="nipy_spectral", interpolation='none')
            pyplot.imshow(img, interpolation='none', alpha=0.5)
            pyplot.axis('off')
            pyplot.savefig(saved_img)
            self.signals.log.emit(saved_img)
            progression = 50 + 100 * files_processed / (files_nb * 2)
            self.signals.progressed.emit(progression)


def model_from_checkpoint_path_nb(checkpoints_path, checkpoint_nb):
    """Loads the weights from the n° model in the specified folder."""

    assert (os.path.isfile(checkpoints_path + "_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(checkpoints_path + "_config.json", "r").read())
    weights = checkpoints_path + "." + str(checkpoint_nb)
    assert (os.path.isfile(weights)
            ), "Weights file not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", weights)
    model.load_weights(weights)
    return model


###########################################################################
### Workers (tab "Charbonnières")                                       ###
###########################################################################

def add_borders(img, vig_size=32):
    """[CHARB] - Add black borders (=background) to images in order to extract thumbnails in the edges"""

    img_size = img.shape
    final_size = (img_size[0] + vig_size - 2, img_size[1] + vig_size - 2, 1)
    final_img = np.zeros(final_size, dtype='uint8')

    for x in range(img_size[0]):
        for y in range(img_size[1]):
            final_img[x + vig_size // 2 - 1, y + vig_size // 2 - 1] = img[x, y]

    return final_img


class ExtractionWorker(QRunnable):
    """[CHARB] - Worker wrapper for the thumbnail extraction function"""

    def __init__(self, *args, **kwargs):
        super(ExtractionWorker, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self.session = tf.Session()
        self.graph = tf.get_default_graph()

    @pyqtSlot()
    def run(self):
        """Run the functionality, triggered by Qt"""

        try:
            self.make_dataset(**self.kwargs)
        except Exception as exc:
            self.signals.error.emit(traceback.format_exc())

    def extract_thumbnails(self, src_img, src_seg, dst, id_charb,
                           vig_size=32, increment=1, mode='1px'):
        """[CHARB] - Extract thumbnails of one image"""

        # Variables de stats
        nb_charb = 0
        nb_back = 0

        # On lit l'image et sa segmentation
        try:
            img = io.imread(src_img)
            seg = io.imread(src_seg)
        except:
            self.signals.log.emit(f'ATTENTION: Impossible d\'ouvrir "{src_img}"')
            return False

        # Récupération du nom de l'image
        name = os.path.basename(src_img)
        name = name[:name.index('.')]

        # On peut garder seulement la première couche de la segmentation
        seg = seg[:, :, 0]

        # Ajout des bords à l'image et à sa segmentation
        seg = add_borders(seg, vig_size)
        img = add_borders(img, vig_size)

        img_size = img.shape

        for x in range(0, img_size[0] - vig_size, increment):
            for y in range(0, img_size[1] - vig_size, increment):

                # Création des vignettes
                vig_img = img[x:x + vig_size, y:y + vig_size]
                vig_seg = seg[x:x + vig_size, y:y + vig_size]

                if mode == '4px':
                    center = vig_seg[vig_size // 2 - 1:vig_size // 2 + 1, vig_size // 2 - 1:vig_size // 2 + 1]
                    is_charb = (np.count_nonzero(center == id_charb) == 4)
                elif mode == '1px':
                    is_charb = (vig_seg[vig_size // 2 - 1, vig_size // 2 - 1] == id_charb)
                else:
                    self.signals.error.emit(f'ERREUR: Le mode doit être "1px" ou "4px"')
                    return False

                # Si c'est une charbonnière
                if is_charb:
                    io.imsave(dst + "charb/" + str(name) + "_" + str(x) + "_" + str(y) + ".png", vig_img)
                    nb_charb += 1
                # Sinon (on rajoute juste un pourcentage pour limiter le dataset) -> on classifie comme 'back'
                # TODO: changer ce système d'aléatoire par quelque chose de plus... stable
                elif np.random.rand() < 0.05 and nb_back < 3 * nb_charb:
                    io.imsave(dst + "back/" + str(name) + "_" + str(x) + "_" + str(y) + ".png", vig_img)
                    nb_back += 1

        self.signals.log.emit(f'{src_img} -> {nb_charb} charb, {nb_back} back')
        return nb_back, nb_charb

    def clear_dir(self, directory):
        """[CHARB] - Recursively clear directory"""

        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if os.path.isdir(path):
                self.clear_dir(path)
            else:
                os.unlink(path)

    def make_dataset(self, src_img, src_seg, dst, prop_train=70, id_charb=2, vig_size=32, increment=1, mode='1px'):
        """[CHARB] - Create a dataset by extracting thumbnails from multiples images"""

        # Vide le dossier destination
        self.clear_dir(dst)
        self.signals.log.emit(f'Dataset ({dst}) vidé!')

        # Crée les sous-dossiers "training" et "validation"
        Path(dst + "/training/charb").mkdir(parents=True, exist_ok=True)
        Path(dst + "/training/back").mkdir(parents=True, exist_ok=True)
        Path(dst + "/validation/charb").mkdir(parents=True, exist_ok=True)
        Path(dst + "/validation/back").mkdir(parents=True, exist_ok=True)

        images = os.listdir(src_img)
        nb_images = len(images)

        prop_train = prop_train / 100

        range_train = range(0, int(prop_train * nb_images))
        range_eval = range(int(prop_train * nb_images), nb_images)

        stats = {
            "train": [0, 0],
            "eval": [0, 0],
        }

        for i in range(nb_images):

            img = src_img + "/" + images[i]
            seg = src_seg + "/" + images[i]

            if i in range_train:
                nb_back, nb_charb = self.extract_thumbnails(img, seg, dst + "/training/", id_charb=id_charb,
                                                            vig_size=vig_size, increment=increment, mode=mode)
                stats["train"][0] += nb_back
                stats["train"][1] += nb_charb
            elif i in range_eval:
                nb_back, nb_charb = self.extract_thumbnails(img, seg, dst + "/validation/", id_charb=id_charb,
                                                            vig_size=vig_size, increment=increment, mode=mode)
                stats["eval"][0] += nb_back
                stats["eval"][1] += nb_charb

            progression = int((i * 100) / nb_images)
            self.signals.progressed.emit(progression)

        self.signals.log.emit(f'\nTraining set: \t {stats["train"][0]} back, {stats["train"][1]} charb')
        self.signals.log.emit(f'Evaluation set: \t {stats["eval"][0]} back, {stats["eval"][1]} charb\n')

        self.signals.finished.emit("Extraction terminée !")


class CharbTrainWorker(QRunnable):
    """[CHARB] - Worker wrapper for the training function"""

    def __init__(self, *args, **kwargs):
        super(CharbTrainWorker, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self.session = tf.Session()
        self.graph = tf.get_default_graph()

    @pyqtSlot()
    def run(self):
        """Run the functionality, triggered by Qt"""

        try:
            self.train(**self.kwargs)
        except Exception as exc:
            self.signals.error.emit(traceback.format_exc())

    def train(self, model_name, train_data_path, eval_data_path, vig_size,
              batch_size, steps_per_epoch, validation_steps, epochs, shuffle,
              save_model_in):
        """[CHARB] - Create and train a model using thumbnails in the 'trainDataPath' directory """

        with self.graph.as_default():
            with self.session.as_default():

                self.signals.log.emit("Chargement des données...")

                img_data_gen = ImageDataGenerator(rescale=1. / 255)

                train_data = img_data_gen.flow_from_directory(directory=train_data_path,
                                                              target_size=(vig_size, vig_size),
                                                              batch_size=batch_size,
                                                              color_mode='grayscale',
                                                              shuffle=True)

                val_data = img_data_gen.flow_from_directory(directory=eval_data_path,
                                                            target_size=(vig_size, vig_size),
                                                            batch_size=batch_size,
                                                            color_mode='grayscale',
                                                            shuffle=True)

                self.signals.log.emit(f'Train dataset: {len(train_data) * batch_size} images')
                self.signals.log.emit(f'Evaluation dataset: {len(val_data) * batch_size} images')

                self.signals.log.emit("Début de la session d'entrainement\n")

                # Crée et compile le modèle renseigné par l'utilisateur
                if model_name == 'vggCharb':
                    model = charb_models.vgg4((vig_size, vig_size, 1))
                else:
                    self.signals.error.emit(f'ERREUR: Le modèle "{model_name}" n\' existe pas')
                    return

                # --- Checkpoint et EarlyStopping ---
                # TODO: possibilité de gérer ces params (earlystopping) depuis l'interface? ou on enlève ça?
                # NOTE: PC tom = 'val_acc', PC poly = 'val_accuracy'
                #
                # # Sauvegarde le modèle si le paramètre 'val_acc' est meilleur que dans la dernière époque
                # checkpoint = ModelCheckpoint(saveModelAs + "/" + modelName + ".h5", monitor='val_accuracy', verbose=1,
                #                              save_best_only=True, save_weights_only=False, mode='auto', period=1)
                #
                # # Stoppe le training si le paramètre 'val_acc' ne s'améliore pas pendant 'patience' époques
                # early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

                if steps_per_epoch == 0:
                    steps_per_epoch = len(train_data)

                start = time.time()

                max_acc = -math.inf  # maxAccuracy

                # Pour chaque époque
                for epoch in range(epochs):
                    self.signals.log.emit(f'Début de l\'époque {epoch}')

                    # Entraînement
                    hist = model.fit_generator(generator=train_data, validation_data=val_data,
                                               steps_per_epoch=steps_per_epoch,
                                               validation_steps=validation_steps, epochs=1,
                                               shuffle=shuffle,  # callbacks=[checkpoint, early]
                                               verbose=1)

                    # Affichage des métriques (loss et accuracy)
                    msg = ""
                    for key, value in hist.history.items():
                        msg += "{}:{}  ".format(str(key), str(value))
                    self.signals.log.emit(msg)

                    # Si l'accuracy a augmenté, on sauvegarde le modèle
                    if hist.history["val_acc"][0] > max_acc:
                        self.signals.log.emit(
                            f'La précision a augmenté ({hist.history["val_acc"][0]} > {max_acc}) => modèle sauvegardé')
                        max_acc = hist.history["val_acc"][0]
                        model.save(save_model_in + "/" + model_name + ".h5")

                    self.signals.log.emit("")

                    progression = 100 * (epoch + 1) / epochs
                    self.signals.progressed.emit(progression)

                end = time.time()
                duration = (end - start) / 60

                self.signals.log.emit(f'Entrainement terminé en {duration:.0f}mn')

                # plt.plot(hist.history['acc'])  # [!] Note: PC tom = 'acc' / PC poly = 'accuracy'
                # plt.plot(hist.history['val_acc'])  # [!] Note: idem
                # plt.plot(hist.history['loss'])
                # plt.plot(hist.history['val_loss'])
                # plt.title('Model accuracy')
                # plt.ylabel('Accuracy')
                # plt.xlabel('Epochs')
                # plt.legend(['acc', 'val_acc', 'loss', 'val_loss'])
                # plt.show()

                self.signals.finished.emit(f'Entrainement terminé')


class CharbEvalWorker(QRunnable):
    """[CHARB] - Worker wrapper for the eval function"""

    def __init__(self, *args, **kwargs):
        super(CharbEvalWorker, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self.session = tf.Session()
        self.graph = tf.get_default_graph()

    @pyqtSlot()
    def run(self):
        """Run the functionality, triggered by Qt"""

        try:
            self.eval(**self.kwargs)
        except Exception as exc:
            self.signals.error.emit(traceback.format_exc())

    def eval(self, existing_model_path, eval_data_path, vig_size, batch_size):
        """[CHARB] - Evaluate a model using thumbnails in the 'testDataPath' directory"""

        with self.graph.as_default():
            with self.session.as_default():
                self.signals.log.emit("Chargement des données...")

                img_data_gen = ImageDataGenerator(rescale=1. / 255)

                test_data = img_data_gen.flow_from_directory(directory=eval_data_path,
                                                             target_size=(vig_size, vig_size),
                                                             batch_size=batch_size,
                                                             color_mode='grayscale')

                self.signals.log.emit(f'Test dataset: {test_data.__len__() * batch_size} images\n')

                self.signals.log.emit("Début de la session d'évaluation...\n")

                # Chargement du modèle
                saved_model = load_model(existing_model_path)

                # Prédictions
                start = time.time()
                predictions = saved_model.predict(test_data)
                end = time.time()

                self.signals.log.emit(f'Évaluation terminée en {end - start:.2f}s')

                test_labels = []
                nb_examples = len(test_data.filenames)
                nb_calls = math.ceil(nb_examples / (1.0 * batch_size))
                for i in range(0, int(nb_calls)):
                    test_labels.extend(test_data[i][1])
                test_labels = np.argmax(test_labels, axis=1)

                report = classification_report(y_true=test_labels,
                                               y_pred=predictions.argmax(axis=1),
                                               target_names=['back', 'charb'],
                                               output_dict=False)

                conf_matrix = confusion_matrix(test_labels, predictions.argmax(axis=1))

                self.signals.log.emit(report)

                self.signals.log.emit('_____| GROUND TRUTH  |')
                self.signals.log.emit('PRED | back  | char  |')
                self.signals.log.emit(f'back |{conf_matrix[0, 0]:6d} |{conf_matrix[1, 0]:6d} |')
                self.signals.log.emit(f'char |{conf_matrix[0, 1]:6d} |{conf_matrix[1, 1]:6d} |\n')

                self.signals.finished.emit(f'Évaluation terminée')


class CharbPredictWorker(QRunnable):
    """[CHARB] - Worker wrapper for the predict function"""

    def __init__(self, *args, **kwargs):
        super(CharbPredictWorker, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self.session = tf.Session()
        self.graph = tf.get_default_graph()

    @pyqtSlot()
    def run(self):
        """Run the functionality, triggered by Qt"""

        try:
            self.predictDataset(**self.kwargs)
        except Exception as exc:
            self.signals.error.emit(traceback.format_exc())

    def predict(self, model, img_name, vig_size=32, increment=1, batch_size=32, mode='1px'):
        """[CHARB] - Predict the segmentation of an image specified by 'imgName'"""

        # On ouvre l'image et on la converit en array
        img = image.load_img(img_name, color_mode='grayscale')
        img = image.img_to_array(img)

        # On crée la segmentation (de la même taille)
        img_size = img.shape
        seg = np.zeros((img_size[0], img_size[1]), dtype='float')

        # On ajoute des bords noirs à l'image
        img = add_borders(img, vig_size)
        img_size = img.shape

        # Variables de gestion de la boucle
        finished = False  # True si on a finit de prédire tous les pixels de l'image
        x = 0
        y = 0

        # Tant qu'on a pas finit de prédire tous les pixels...
        while not finished:

            # On créé un batch vide et un tableau de positions
            batch = []
            pos = []

            # Et on remplit le batch (tant qu'on a pas atteind batchSize ou qu'on a pas finit)
            while len(batch) < batch_size and not finished:

                # On extrait la vignette et on l'ajoute au batch (et on se rappelle de sa position)
                thumbnail = img[x:x + vig_size, y:y + vig_size]
                batch.append(thumbnail)
                pos.append((x, y))

                # Gestion de la position
                x += increment  # à chaque itération, on incrémente x
                if x >= img_size[0] - vig_size + 1:  # si x est au bord de l'image...
                    x = 0  # on le remet à 0
                    y += increment  # et on incrémente y
                if y >= img_size[1] - vig_size + 1:  # si y est au bord de l'image...
                    finished = True  # on a finit

            # Prédiction du batch
            batch = np.asarray(batch)
            output = model.predict_proba(batch)

            # Reconstitution segmentation
            for j in range(len(pos)):
                if mode == '1px':
                    seg[pos[j]] = output[j, 1]
                elif mode == '4px':
                    seg[pos[j][0], pos[j][1]] = output[j, 1]
                    seg[pos[j][0] + 1, pos[j][1]] = output[j, 1]
                    seg[pos[j][0], pos[j][1] + 1] = output[j, 1]
                    seg[pos[j][0] + 1, pos[j][1] + 1] = output[j, 1]
                else:
                    self.signals.error.emit("ERREUR: mode incorrect (doit être 1px ou 4px)")
                    return

        return seg

    def binary(self, segmentation, threshold=0.5):
        """[CHARB] - Return a binary representation of the segmentation (0=background, 1=charb)"""

        img_size = segmentation.shape
        bin_seg = np.zeros(img_size, dtype='uint8')

        for x in range(img_size[0]):
            for y in range(img_size[1]):
                if segmentation[x, y] > threshold:
                    bin_seg[x, y] = 1

        return bin_seg

    def superposition(self, segmentation, lidar, threshold=0.5):
        """[CHARB] - Return a superposition of the segmentation and the source image"""

        img_size = segmentation.shape
        final = np.zeros((img_size[0], img_size[1], 4), dtype='uint8')
        lidar = lidar.astype('uint8')

        for x in range(img_size[0]):
            for y in range(img_size[1]):
                if segmentation[x, y] > threshold:
                    final[x, y] = (lidar[x, y], min(lidar[x, y] + 70, 255), lidar[x, y], 200)
                else:
                    final[x, y] = (lidar[x, y], lidar[x, y], lidar[x, y], 255)

        return final

    def predictDataset(self, model_name, images_path, save_seg_path, save_sup_path, vig_size, intervalle, batch_size,
                       mode):
        """[CHARB] - Predict all images in a directory"""

        with self.graph.as_default():
            with self.session.as_default():
                start = time.time()

                # On récupère les images
                images = os.listdir(images_path)
                self.signals.log.emit(f'Nombre d\'images à prédire : {len(images)}\n')

                # On charge le modèle
                model = load_model(model_name)
                self.signals.log.emit("Début de la session de prédictions\n")

                # Pour chaque image
                for i in range(len(images)):
                    self.signals.log.emit(f'Prédiction de l\'image {images[i]}...')

                    # On prédit sa segmentation
                    pred = self.predict(model=model, img_name=images_path + "/" + images[i],
                                        vig_size=vig_size, increment=intervalle, batch_size=batch_size, mode=mode)

                    self.signals.log.emit(f'OK! Création de la segmentation et de la superposition.\n')

                    # On charge l'image source (lidar)
                    lidar = image.load_img(images_path + "/" + images[i], color_mode='grayscale')
                    lidar = image.img_to_array(lidar)

                    # On créé la segmentation et la superposition
                    seg = self.binary(pred)
                    sup = self.superposition(pred, lidar)

                    # Et on les sauvegarde
                    plt.imsave(save_seg_path + "/" + images[i], seg, vmin=0, vmax=1, cmap='gray')
                    plt.imsave(save_sup_path + "/" + images[i], sup)

                    progression = int(100 * (i + 1) / len(images))
                    self.signals.progressed.emit(progression)

                end = time.time()
                self.signals.log.emit(f'Prédictions terminées en {(end - start) / 60:.2f}mn!')
                self.signals.finished.emit(f'Prédictions terminées')
