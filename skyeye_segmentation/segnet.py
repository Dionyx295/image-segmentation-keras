from keras_segmentation.models.segnet import segnet
from keras_segmentation.predict import *
from PIL import Image
import os, sys

def resize(path="data/train_images/", size=(400,400)):
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize(size, Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=100)

#model = segnet(n_classes=3, input_height=400, input_width=400)

#model.train(
#    train_images =  "data/train_images/",
#    train_annotations = "data/train_segmentation",
#    checkpoints_path = "models/segnet" ,
#    epochs=8,
#    steps_per_epoch=300,
#    batch_size=2
#)

print(os.getcwd())
model = model_from_checkpoint_path(os.getcwd() + "\models\segnet")

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