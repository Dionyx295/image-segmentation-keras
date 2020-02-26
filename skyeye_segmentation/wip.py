from keras.preprocessing.image import ImageDataGenerator
from keras_segmentation.models.segnet import segnet
from keras_segmentation.models.unet import unet
from keras_segmentation.models.fcn import fcn_8, fcn_32
from keras_segmentation.models.pspnet import pspnet
from keras_segmentation.predict import *
from keras_segmentation.data_utils.augmentation import *
from PIL import Image
from matplotlib import pyplot
from random import randint
from scipy.optimize import curve_fit

import numpy as np
import os, sys
import keras_segmentation.data_utils.preprocess as pp

def Resize(path="data/train_images/", size=(400,400)):
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize(size, Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=100)

def AugmentData():
    image_gen = ImageDataGenerator(rotation_range=90,
                                   width_shift_range=0.25,
                                   height_shift_range=0.25,
                                   shear_range=10,
                                   zoom_range=[0.9, 1.25],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='reflect',
                                   dtype="uint8")

    rand_seed = randint(1, 9999999)

    img_generator = image_gen.flow_from_directory("data_manuel_full",
                                              (400, 400),
                                              'rgb',
                                              classes=["train_images"],
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
    mask_generator = image_gen.flow_from_directory("data_manuel_full",
                                                  (400, 400),
                                                  'rgb',
                                                  classes=["train_segmentation"],
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

    img_num = 500

    # Img
    fig = pyplot.figure(figsize=(8, 8))
    for i in range(1, img_num+1):
        img = img_generator.next()
        image = img[0][0].astype('uint8')
        pyplot.imsave("data_manuel_full/aug_images/" + str(i) + ".png", image)
        #fig.add_subplot(3, 3, i)
        #pyplot.imshow(image)
    #pyplot.show()

    # Masks
    fig = pyplot.figure(figsize=(8, 8))
    for i in range(1, img_num+1):
        img = mask_generator.next()
        image = img[0][0].astype('uint8')
        pyplot.imsave("data_manuel_full/aug_segmentation/"+str(i)+".png", image)
        #fig.add_subplot(3, 3, i)
        #pyplot.imshow(image)
    #pyplot.show()
    print("Augmentation done !")


def Train(model):
    model.train(
        train_images =  "data_manuel_talus/aug_images/",
        train_annotations = "data_manuel_talus/aug_segmentation",
        checkpoints_path = "models/segnet/segnet-talus",
        epochs=50,
        steps_per_epoch=63,
        batch_size=8
    )

def Evaluate(model):
    unlabelled = (255, 255, 255)
    charb = (0, 155, 0)
    talus = (155, 0, 0)
    tertres = (0, 0, 155)
    class_colors = np.array([unlabelled, charb, talus, tertres])
    all_pr = predict_multiple(model=model,
                              inp_dir=os.getcwd()+"/data_auto/test_images/",
                              out_dir=os.getcwd()+"/data_auto/predict/",
                              colors=class_colors)

    # evaluating the model
    results = model.evaluate_segmentation( inp_images_dir=os.getcwd()+"/data_auto/test_images/" ,
                                           annotations_dir=os.getcwd()+"/data_auto/test_segmentation/" )
    print(results)
    #print(results['class_wise_IU'][1])

def exponentail_basic(x, a, b, c):
    return a * np.exp(-b * np.asarray(x)) + c

def polynomial(x, a, b, c, d, e):
    return a + b*np.asarray(x) + c*np.asarray(x)**2 + d*np.asarray(x)**3 + e*np.asarray(x)**4

def power(x, a, b):
    return a*np.asarray(x)**b

def graph_from_file(file_path):
    count = 0
    x = []
    y = []
    with open(os.getcwd() + file_path, 'r') as f:
        for line in f:
            count += 1
            x.append(count)
            y.append(float(line))
    pyplot.ylabel('IU')
    pyplot.xlabel('Epoch')
    pyplot.scatter(x, y, c='blue')
    popt, pcov = curve_fit(polynomial, x, y)
    print(popt)
    pyplot.plot(x, polynomial(x, *popt), c='red')
    pyplot.show()

def Evaluate_perf(models_path, perf_file):
    i = 0
    while os.path.isfile(os.getcwd() + models_path + "." + str(i)):
        model = model_from_checkpoint_path_nb(os.getcwd() + models_path, i)
        i += 1

        # evaluating the model
        results = model.evaluate_segmentation(inp_images_dir=os.getcwd() + "/data_auto/test_images/",
                                          annotations_dir=os.getcwd() + "/data_auto/test_segmentation/")
        print(results['class_wise_IU'][1])
        file = open(os.getcwd() + perf_file, "a")
        file.write(str(results['class_wise_IU'][1]) + "\n")
        file.close()

def model_from_checkpoint_path_nb(checkpoints_path, checkpoint_nb):

    from keras_segmentation.models.all_models import model_from_name
    assert (os.path.isfile(checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(checkpoints_path+"_config.json", "r").read())
    weights = checkpoints_path+"."+str(checkpoint_nb)
    assert (os.path.isfile(weights)
            ), "Weights file not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", weights)
    model.load_weights(weights)
    return model

'''
pp.mask_fusion(class_pathes=["data_manuel_full/talus/","data_manuel_full/charb/","data_manuel_full/tertres/",],
            class_scales=[1, 2, 3],
            size=(400,400),
            save_to="data_manuel_full/train_segmentation/")
'''


#AugmentData()

model = fcn_8(n_classes=3, input_height=400, input_width=400)

#model = model_from_checkpoint_path(os.getcwd() + "\\models\\fcn_32\\fcn_32-auto")


model.train(
        train_images =  "data/train_images",
        train_annotations = "data/train_segmentation",
        checkpoints_path = "models/fcn_8",
        epochs=1,
        steps_per_epoch=10,
        batch_size=4)

Evaluate(model)
#Evaluate_perf("\\models\\fcn_32\\fcn_32-auto", "\\models\\fcn_32\\fcn_32-auto-perf.txt")

#graph_from_file("\models\\fcn_32\\fcn_32-full-perf.txt")




