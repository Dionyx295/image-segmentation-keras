import pytest
import os

from skyeye_segmentation.controller.skyeye_func import CharbTrainWorker


def test_train():

    if os.path.exists("test_train/models/vggCharb.h5"):
        os.remove("test_train/models/vggCharb.h5")

    train_data_path = "test_train/training"
    eval_data_path = "test_train/validation"
    save_model_in = "test_train/models"

    vig_size = 32
    batch_size = 32
    steps_per_epoch = 5
    validation_steps = 5
    epochs = 2
    shuffle = True

    worker = CharbTrainWorker(model_name="vggCharb",
                              train_data_path=train_data_path, eval_data_path=eval_data_path,
                              vig_size=vig_size, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps, epochs=epochs, shuffle=shuffle,
                              save_model_in=save_model_in)
    worker.run()

    assert os.path.exists("test_train/models/vggCharb.h5")


test_train()

print("\n[TRAINING] All tests passed with success!\n")
