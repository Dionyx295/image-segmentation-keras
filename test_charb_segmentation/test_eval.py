import pytest

from skyeye_segmentation.controller.skyeye_func import CharbEvalWorker


def test_eval():

    eval_data_path = "test_train/validation"
    existing_model_path = "test_train/models/vggCharb.h5"

    vig_size = 32
    batch_size = 32

    worker = CharbEvalWorker(existing_model_path=existing_model_path, eval_data_path=eval_data_path,
                             vig_size=vig_size, batch_size=batch_size)

    worker.run()


test_eval()

print("\n[EVALUATION] All tests passed with success!\n")
