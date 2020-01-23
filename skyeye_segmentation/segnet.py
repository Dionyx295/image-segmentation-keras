from keras.preprocessing.image import ImageDataGenerator
from keras_segmentation.models.segnet import segnet
from keras_segmentation.predict import *
from keras_segmentation.data_utils.augmentation import *
from PIL import Image
from matplotlib import pyplot
from random import randint
import os, sys

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

    img_generator = image_gen.flow_from_directory("data",
                                              (400, 400),
                                              'rgb',
                                              classes=["to_aug_images"],
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
    mask_generator = image_gen.flow_from_directory("data",
                                                  (400, 400),
                                                  'rgb',
                                                  classes=["to_aug_segmentation"],
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

    img_num = 1500

    # Img
    fig = pyplot.figure(figsize=(8, 8))
    for i in range(1, img_num):
        img = img_generator.next()
        image = img[0][0].astype('uint8')
        pyplot.imsave("data/aug_images/" + str(i) + ".png", image)
        #fig.add_subplot(3, 3, i)
        #pyplot.imshow(image)
    #pyplot.show()

    # Masks
    fig = pyplot.figure(figsize=(8, 8))
    for i in range(1, img_num):
        img = mask_generator.next()
        image = img[0][0].astype('uint8')
        pyplot.imsave("data/aug_segmentation/"+str(i)+".png", image)
        #fig.add_subplot(3, 3, i)
        #pyplot.imshow(image)
    #pyplot.show()


def Train(model):
    model.train(
        train_images =  "data/aug_images/",
        train_annotations = "data/aug_segmentation",
        checkpoints_path = "models/segnet",
        validate=False,
        val_images="data/val_images",
        val_annotations="data/val_segmentation",
        val_batch_size=4,
        epochs=1,
        steps_per_epoch=300,
        batch_size=4
    )

def Predict(model):
    unlabelled = (255, 255, 255)
    talus = (155, 0, 0)
    charbonniere = (0, 155, 0)
    class_colors = np.array([unlabelled, talus, charbonniere])
    all_pr = predict_multiple(model=model,
                              inp_dir=os.getcwd()+"/data/test_images/",
                              out_dir=os.getcwd()+"/data/predict/",
                              colors=class_colors)

    import matplotlib.pyplot as plt

    for pr in all_pr:
        plt.imshow(pr)

    # evaluating the model
    print(model.evaluate_segmentation( inp_images_dir=os.getcwd()+"/data/test_images/" , annotations_dir=os.getcwd()+"/data/test_segmentation/" ) )

#AugmentData()

#model = segnet(n_classes=3, input_height=400, input_width=400)
model = model_from_checkpoint_path(os.getcwd() + "\models\segnet")
#Train(model)
Predict(model)