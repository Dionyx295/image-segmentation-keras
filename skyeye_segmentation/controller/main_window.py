"""Main controller, handles connections, ui management."""
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QtWarningMsg, QThreadPool, QtCriticalMsg, QSettings, \
    pyqtSlot

from keras_segmentation.models.all_models import model_from_name
from skyeye_segmentation.controller.error_handler import errormsg
from skyeye_segmentation.view import main_window
from skyeye_segmentation.controller.skyeye_func import MaskFusionWorker, \
    ImageAugmentationWorker, TrainWorker, EvalWorker, PredictWorker

# Imports SkyEye "Charbonnières"
from skyeye_segmentation.controller.skyeye_func import ExtractionWorker, CharbTrainWorker, \
    CharbEvalWorker, CharbPredictWorker


class MainWindow(QMainWindow):
    """Main window controller class"""

    log_file = ""

    def __init__(self):
        super().__init__()

        # Creating log file and redirecting stdout
        self.stdout_original = sys.stdout
        work_dir = os.path.join(Path(os.getcwd()), "log")
        if not os.path.isdir(work_dir):
            os.mkdir(work_dir)
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y %Hh%Mm%Ss")

        MainWindow.log_file = os.path.join(work_dir, "logs_" + dt_string + ".txt")
        # [!] UNCOMMENT TO PRINT IN LOG FILE
        # sys.stdout = open(MainWindow.log_file, 'a')
        # sys.stderr = open(MainWindow.log_file, 'a')

        self.qt_ui = main_window.Ui_MainWindow()
        self.qt_ui.setupUi(self)
        self.thread_pool = QThreadPool()

        self.settings = QSettings("Polytech Tours", "SkyEye")
        self.load_settings()

        # Connections to slots
        # File and folder browsers
        self.qt_ui.mask_prep_browse_button.clicked \
            .connect(self.on_mask_prep_browse_button_click)
        self.qt_ui.aug_images_source_browse_button.clicked \
            .connect(self.on_aug_images_source_browse_button_click)
        self.qt_ui.aug_seg_source_browse_button.clicked \
            .connect(self.on_aug_seg_source_browse_button_click)
        self.qt_ui.aug_images_dest_browse_button.clicked \
            .connect(self.on_aug_images_dest_browse_button_click)
        self.qt_ui.aug_seg_dest_browse_button.clicked \
            .connect(self.on_aug_seg_dest_browse_button_click)
        self.qt_ui.train_images_browse_button.clicked \
            .connect(self.on_train_images_browse_button_click)
        self.qt_ui.train_seg_browse_button.clicked \
            .connect(self.on_train_seg_browse_button_click)
        self.qt_ui.eval_images_browse_button.clicked \
            .connect(self.on_eval_images_browse_button_click)
        self.qt_ui.eval_seg_browse_button.clicked \
            .connect(self.on_eval_seg_browse_button_click)
        self.qt_ui.existing_model_path_browse_button.clicked \
            .connect(self.on_existing_model_path_browse_button_click)
        self.qt_ui.save_model_path_browse_button.clicked \
            .connect(self.on_save_model_path_browse_button_click)
        self.qt_ui.predict_model_path_browse_button.clicked \
            .connect(self.on_predict_model_path_browse_button_click)
        self.qt_ui.predict_images_browse_button.clicked \
            .connect(self.on_predict_images_browse_button_click)
        self.qt_ui.saved_seg_browse_button.clicked \
            .connect(self.on_saved_seg_browse_button_click)
        self.qt_ui.saved_sup_browse_button.clicked \
            .connect(self.on_saved_sup_browse_button_click)

        # Classes prep
        self.qt_ui.add_class_button.clicked \
            .connect(self.on_add_class_button_click)
        self.qt_ui.remove_class_button.clicked \
            .connect(self.on_remove_class_button_click)
        self.qt_ui.fusion_button.clicked \
            .connect(self.on_fusion_button_click)

        # Augmentation
        self.qt_ui.aug_button.clicked \
            .connect(self.on_aug_button_click)

        # Training
        self.qt_ui.train_button.clicked \
            .connect(self.on_train_button_click)

        # Evaluation
        self.qt_ui.eval_button.clicked \
            .connect(self.on_eval_button_click)

        # Prediction
        self.qt_ui.predict_button.clicked \
            .connect(self.on_predict_button_click)

        # UI management
        # Mask fusion
        self.qt_ui.classes_list.model().rowsInserted \
            .connect(self.check_fusion_available)
        self.qt_ui.classes_list.model().rowsRemoved \
            .connect(self.check_fusion_available)
        self.qt_ui.resize_height_spinbox.valueChanged \
            .connect(self.check_fusion_available)
        self.qt_ui.resize_width_spinbox.valueChanged \
            .connect(self.check_fusion_available)
        self.qt_ui.mask_prep_field.textChanged \
            .connect(self.check_fusion_available)
        # Image augmentation
        self.qt_ui.aug_images_source_field.textChanged \
            .connect(self.check_aug_available)
        self.qt_ui.aug_seg_source_field.textChanged \
            .connect(self.check_aug_available)
        self.qt_ui.aug_images_dest_field.textChanged \
            .connect(self.check_aug_available)
        self.qt_ui.aug_seg_dest_field.textChanged \
            .connect(self.check_aug_available)
        self.qt_ui.aug_images_nb_spinbox.valueChanged \
            .connect(self.check_aug_available)
        self.qt_ui.aug_rotation_range_spinbox.valueChanged \
            .connect(self.check_aug_available)
        self.qt_ui.aug_horizontal_spinbox.valueChanged \
            .connect(self.check_aug_available)
        self.qt_ui.aug_vertical_spinbox.valueChanged \
            .connect(self.check_aug_available)
        self.qt_ui.aug_zoom_spinbox.valueChanged \
            .connect(self.check_aug_available)
        self.qt_ui.aug_shear_spinbox.valueChanged \
            .connect(self.check_aug_available)
        self.qt_ui.aug_width_spinbox.valueChanged \
            .connect(self.check_aug_available)
        self.qt_ui.aug_height_spinbox.valueChanged \
            .connect(self.check_aug_available)
        # Training
        self.qt_ui.train_images_field.textChanged \
            .connect(self.check_train_available)
        self.qt_ui.train_seg_field.textChanged \
            .connect(self.check_train_available)
        self.qt_ui.model_combobox.currentTextChanged \
            .connect(self.clear_existing_model)
        self.qt_ui.width_model_spinbox.valueChanged \
            .connect(self.clear_existing_model)
        self.qt_ui.height_model_spinbox.valueChanged \
            .connect(self.clear_existing_model)
        self.qt_ui.width_model_spinbox.valueChanged \
            .connect(self.check_train_available)
        self.qt_ui.height_model_spinbox.valueChanged \
            .connect(self.check_train_available)
        self.qt_ui.existing_model_path_field.textChanged \
            .connect(self.check_train_available)
        self.qt_ui.batch_size_spinbox.valueChanged \
            .connect(self.check_train_available)
        self.qt_ui.step_epoch_spinbox.valueChanged \
            .connect(self.check_train_available)
        self.qt_ui.epochs_spinbox.valueChanged \
            .connect(self.check_train_available)
        self.qt_ui.save_model_path_field.textChanged \
            .connect(self.check_train_available)
        # Evaluation
        self.qt_ui.eval_images_field.textChanged \
            .connect(self.check_eval_available)
        self.qt_ui.eval_seg_field.textChanged \
            .connect(self.check_eval_available)
        self.qt_ui.existing_model_path_field.textChanged \
            .connect(self.check_eval_available)
        # Prediction
        self.qt_ui.predict_model_path_field.textChanged \
            .connect(self.check_predict_available)
        self.qt_ui.predict_images_field.textChanged \
            .connect(self.check_predict_available)
        self.qt_ui.saved_seg_field.textChanged \
            .connect(self.check_predict_available)
        self.qt_ui.saved_sup_field.textChanged \
            .connect(self.check_predict_available)

        ###########################################################################
        ### Connections to slots (tab "Charbonnières")                          ###
        ###########################################################################

        # Thumbnail extraction
        self.qt_ui.charb_extra_images_browse_button.clicked \
            .connect(self.on_charb_extra_images_browse_button_click)
        self.qt_ui.charb_extra_seg_browse_button.clicked \
            .connect(self.on_charb_extra_seg_browse_button_click)
        self.qt_ui.charb_extra_dataset_browse_button.clicked \
            .connect(self.on_charb_extra_dataset_browse_button_click)
        self.qt_ui.charb_extra_proportion_slider.valueChanged \
            .connect(self.on_charb_extra_proportion_slider_change)
        self.qt_ui.charb_extra_extract_button.clicked \
            .connect(self.on_charb_extra_extract_button_click)

        # Training and evaluation
        self.qt_ui.charb_train_traindata_browse_button.clicked \
            .connect(self.on_charb_train_traindata_browse_button_click)
        self.qt_ui.charb_train_evaldata_browse_button.clicked \
            .connect(self.on_charb_train_evaldata_browse_button_click)
        self.qt_ui.charb_train_loadmodel_browse_button.clicked \
            .connect(self.on_charb_train_loadmodel_browse_button_click)
        self.qt_ui.charb_train_savemodel_browse_button.clicked \
            .connect(self.on_charb_train_savemodel_browse_button_click)
        self.qt_ui.charb_train_train_button.clicked \
            .connect(self.on_charb_train_train_button_click)
        self.qt_ui.charb_train_eval_button.clicked \
            .connect(self.on_charb_train_eval_button_click)

        # Predictions
        self.qt_ui.charb_pred_model_browse_button.clicked \
            .connect(self.on_charb_pred_model_browse_button_click)
        self.qt_ui.charb_pred_images_browse_button.clicked \
            .connect(self.on_charb_pred_images_browse_button_click)
        self.qt_ui.charb_pred_seg_browse_button.clicked \
            .connect(self.on_charb_pred_seg_browse_button_click)
        self.qt_ui.charb_pred_sup_browse_button.clicked \
            .connect(self.on_charb_pred_sup_browse_button_click)
        self.qt_ui.charb_pred_predict_button.clicked \
            .connect(self.on_charb_pred_predict_button_click)

        # UI Management
        # Thumbnail extraction
        self.qt_ui.charb_extra_images_field.textChanged \
            .connect(self.charb_check_extract_available)
        self.qt_ui.charb_extra_seg_field.textChanged \
            .connect(self.charb_check_extract_available)
        self.qt_ui.charb_extra_dataset_field.textChanged \
            .connect(self.charb_check_extract_available)
        self.qt_ui.charb_extra_vigsize_spinbox.valueChanged \
            .connect(self.charb_check_extract_available)
        self.qt_ui.charb_extra_intervalle_spinbox.valueChanged \
            .connect(self.charb_check_extract_available)
        self.qt_ui.charb_extra_1px_checkbox.stateChanged \
            .connect(self.charb_check_extract_available)
        self.qt_ui.charb_extra_4px_checkbox.stateChanged \
            .connect(self.charb_check_extract_available)
        # Training
        self.qt_ui.charb_train_traindata_field.textChanged \
            .connect(self.charb_check_train_available)
        self.qt_ui.charb_train_evaldata_field.textChanged \
            .connect(self.charb_check_train_available)
        self.qt_ui.charb_train_vigsize_spinbox.valueChanged \
            .connect(self.charb_check_train_available)
        self.qt_ui.charb_train_model_combobox.currentTextChanged \
            .connect(self.charb_check_train_available)
        self.qt_ui.charb_train_batchsize_spinbox.valueChanged \
            .connect(self.charb_check_train_available)
        self.qt_ui.charb_train_stepperepoch_spinbox.valueChanged \
            .connect(self.charb_check_train_available)
        self.qt_ui.charb_train_epochs_spinbox.valueChanged \
            .connect(self.charb_check_train_available)
        self.qt_ui.charb_train_savemodel_field.textChanged \
            .connect(self.charb_check_train_available)
        # Evaluation
        self.qt_ui.charb_train_evaldata_field.textChanged \
            .connect(self.charb_check_eval_available)
        self.qt_ui.charb_train_loadmodel_field.textChanged \
            .connect(self.charb_check_eval_available)
        # Predictions
        self.qt_ui.charb_pred_model_field.textChanged \
            .connect(self.charb_check_predict_available)
        self.qt_ui.charb_pred_images_field.textChanged \
            .connect(self.charb_check_predict_available)
        self.qt_ui.charb_pred_seg_field.textChanged \
            .connect(self.charb_check_predict_available)
        self.qt_ui.charb_pred_sup_field.textChanged \
            .connect(self.charb_check_predict_available)
        self.qt_ui.charb_pred_batchsize_spinbox.valueChanged \
            .connect(self.charb_check_predict_available)
        self.qt_ui.charb_pred_vigsize_spinbox.valueChanged \
            .connect(self.charb_check_predict_available)
        self.qt_ui.charb_pred_intervalle_spinbox.valueChanged \
            .connect(self.charb_check_predict_available)
        self.qt_ui.charb_pred_1px_checkbox.stateChanged \
            .connect(self.charb_check_predict_available)
        self.qt_ui.charb_pred_4px_checkbox.stateChanged \
            .connect(self.charb_check_predict_available)

        ###########################################################################
        ###########################################################################
        ###########################################################################

        # Dictionnary to bind classes and their path
        self.classes_folders = {}

        # List of available models
        model_names = model_from_name.keys()
        self.qt_ui.model_combobox.addItems(model_names)

        # Error message handler
        # qInstallMessageHandler(errormsg)

        self.check_all_available()
        self.show()

    def closeEvent(self, event):
        """QMainWindow closeEvent override"""
        # Confirmation
        if self.qt_ui.progress_bar.isEnabled():
            msg = "Voulez-vous quitter et arrêter le traitement en cours ?"
        else:
            msg = "Voulez-vous vraiment quitter ?"
        reply = QMessageBox.question(self, 'Quitter ?',
                                     msg, QMessageBox.Yes, QMessageBox.Cancel)
        if reply == QMessageBox.Yes:
            raise SystemExit(0)  # Terminates the process and other threads

        event.ignore()

    def load_settings(self):
        """Settings loader"""
        # Path fields
        self.qt_ui.aug_images_source_field \
            .setText(self.settings.value("aug_images_source_field"))
        self.qt_ui.aug_seg_source_field \
            .setText(self.settings.value("aug_seg_source_field"))
        self.qt_ui.aug_images_dest_field \
            .setText(self.settings.value("aug_images_dest_field"))
        self.qt_ui.aug_seg_dest_field \
            .setText(self.settings.value("aug_seg_dest_field"))
        self.qt_ui.train_images_field \
            .setText(self.settings.value("train_images_field"))
        self.qt_ui.train_seg_field \
            .setText(self.settings.value("train_seg_field"))
        self.qt_ui.eval_images_field \
            .setText(self.settings.value("eval_images_field"))
        self.qt_ui.eval_seg_field \
            .setText(self.settings.value("eval_seg_field"))
        self.qt_ui.existing_model_path_field \
            .setText(self.settings.value("existing_model_path_field"))
        self.qt_ui.save_model_path_field \
            .setText(self.settings.value("save_model_path_field"))
        self.qt_ui.predict_model_path_field \
            .setText(self.settings.value("predict_model_path_field"))
        self.qt_ui.predict_images_field \
            .setText(self.settings.value("predict_images_field"))
        self.qt_ui.saved_seg_field \
            .setText(self.settings.value("saved_seg_field"))
        self.qt_ui.saved_sup_field \
            .setText(self.settings.value("saved_sup_field"))

        # Augmentation parameters
        value = self.settings.value("aug_images_nb_spinbox")
        if value:
            self.qt_ui.aug_images_nb_spinbox.setValue(value)
        value = self.settings.value("aug_rotation_range_spinbox")
        if value:
            self.qt_ui.aug_rotation_range_spinbox.setValue(value)
        value = self.settings.value("aug_horizontal_spinbox")
        if value:
            self.qt_ui.aug_horizontal_spinbox.setValue(value)
        value = self.settings.value("aug_vertical_spinbox")
        if value:
            self.qt_ui.aug_vertical_spinbox.setValue(value)
        value = self.settings.value("aug_shear_spinbox")
        if value:
            self.qt_ui.aug_shear_spinbox.setValue(value)
        value = self.settings.value("aug_zoom_spinbox")
        if value:
            self.qt_ui.aug_zoom_spinbox.setValue(value)
        value = self.settings.value("aug_width_spinbox")
        if value:
            self.qt_ui.aug_width_spinbox.setValue(value)
        value = self.settings.value("aug_height_spinbox")
        if value:
            self.qt_ui.aug_height_spinbox.setValue(value)
        value = self.settings.value("aug_fill_combobox")
        if value:
            self.qt_ui.aug_fill_combobox.setCurrentText(value)

        # Train settings
        value = self.settings.value("width_model_spinbox")
        if value:
            self.qt_ui.width_model_spinbox.setValue(value)
        value = self.settings.value("height_model_spinbox")
        if value:
            self.qt_ui.height_model_spinbox.setValue(value)
        value = self.settings.value("batch_size_spinbox")
        if value:
            self.qt_ui.batch_size_spinbox.setValue(value)
        value = self.settings.value("step_epoch_spinbox")
        if value:
            self.qt_ui.step_epoch_spinbox.setValue(value)
        value = self.settings.value("epochs_spinbox")
        if value:
            self.qt_ui.epochs_spinbox.setValue(value)
        value = self.settings.value("nb_class_spinbox")
        if value:
            self.qt_ui.nb_class_spinbox.setValue(value)

        ###########################################################################
        ### Load settings (tab "Charbonnières")                                 ###
        ###########################################################################

        # Thumbnails extraction
        self.qt_ui.charb_extra_images_field \
            .setText(self.settings.value("charb_extra_imagesPath", ""))
        self.qt_ui.charb_extra_seg_field \
            .setText(self.settings.value("charb_extra_segmentationsPath", ""))
        self.qt_ui.charb_extra_dataset_field \
            .setText(self.settings.value("charb_extra_datasetPath", ""))
        self.qt_ui.charb_extra_idcharb_spindbox \
            .setValue(self.settings.value("charb_common_idCharb", 2))
        self.qt_ui.charb_extra_vigsize_spinbox \
            .setValue(self.settings.value("charb_common_vigSize", 32))
        self.qt_ui.charb_extra_intervalle_spinbox \
            .setValue(self.settings.value("charb_common_intervalle", 1))
        self.qt_ui.charb_extra_proportion_slider \
            .setValue(self.settings.value("charb_extra_propTrain", 70))
        self.qt_ui.charb_extra_proportion_label \
            .setText(f'{self.settings.value("charb_extra_propTrain", 70)} '
                     f'/ {100 - self.settings.value("charb_extra_propTrain", 70)}')
        if self.settings.value("charb_common_mode", "1px") == "1px":
            self.qt_ui.charb_extra_1px_checkbox.setChecked(True)
        else:
            self.qt_ui.charb_extra_4px_checkbox.setChecked(True)

        # Training
        self.qt_ui.charb_train_model_combobox \
            .setCurrentText(self.settings.value("charb_train_modelName", "vgg4"))
        self.qt_ui.charb_train_traindata_field \
            .setText(self.settings.value("charb_train_trainDataPath", ""))
        self.qt_ui.charb_train_evaldata_field \
            .setText(self.settings.value("charb_train_evalDataPath", ""))
        self.qt_ui.charb_train_stepperepoch_spinbox \
            .setValue(self.settings.value("charb_train_stepsPerEpoch", 100))
        self.qt_ui.charb_train_epochs_spinbox \
            .setValue(self.settings.value("charb_train_epochs", 100))
        self.qt_ui.charb_train_shuffle_checkbox \
            .setChecked(self.settings.value("charb_train_shuffle", True))
        self.qt_ui.charb_train_vigsize_spinbox \
            .setValue(self.settings.value("charb_common_vigSize", 32))
        self.qt_ui.charb_train_batchsize_spinbox \
            .setValue(self.settings.value("charb_common_batchSize", 32))
        self.qt_ui.charb_train_savemodel_field \
            .setText(self.settings.value("charb_train_saveModelAs", ""))

        # Evaluation
        self.qt_ui.charb_train_loadmodel_field \
            .setText(self.settings.value("charb_eval_existingModelPath", ""))

        # Prediction
        self.qt_ui.charb_pred_model_field \
            .setText(self.settings.value("charb_pred_modelName", ""))
        self.qt_ui.charb_pred_images_field \
            .setText(self.settings.value("charb_pred_imagesPath", ""))
        self.qt_ui.charb_pred_seg_field \
            .setText(self.settings.value("charb_pred_saveSegPath", ""))
        self.qt_ui.charb_pred_sup_field \
            .setText(self.settings.value("charb_pred_saveSupPath", ""))
        self.qt_ui.charb_pred_vigsize_spinbox \
            .setValue(self.settings.value("charb_common_vigSize", 32))
        self.qt_ui.charb_pred_batchsize_spinbox \
            .setValue(self.settings.value("charb_common_batchSize", 32))
        self.qt_ui.charb_pred_intervalle_spinbox \
            .setValue(self.settings.value("charb_common_intervalle", 2))
        # self.qt_ui.charb_pred_idCharb_spinbox \
        #     .setText(self.settings.value("charb_common_idCharb", 2"))
        if self.settings.value("charb_common_mode", "1px") == "1px":
            self.qt_ui.charb_pred_1px_checkbox.setChecked(True)
        else:
            self.qt_ui.charb_pred_4px_checkbox.setChecked(True)

    # Slots
    @pyqtSlot()
    def on_mask_prep_browse_button_click(self):
        """File and folder browsers"""
        folder_name = QFileDialog. \
            getExistingDirectory(self, "Sauvegarder les masques dans",
                                 options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.mask_prep_field.setText(folder_name)

    @pyqtSlot()
    def on_aug_images_source_browse_button_click(self):
        """Images to aug browser"""
        folder_name = QFileDialog \
            .getExistingDirectory(self, "Images sources",
                                  options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.aug_images_source_field.setText(folder_name)
            self.settings.setValue("aug_images_source_field", folder_name)
            self.settings.sync()

    @pyqtSlot()
    def on_aug_seg_source_browse_button_click(self):
        """Segs to aug browser"""
        folder_name = QFileDialog \
            .getExistingDirectory(self, "Segmentations sources",
                                  options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.aug_seg_source_field.setText(folder_name)
            self.settings.setValue("aug_seg_source_field", folder_name)
            self.settings.sync()

    @pyqtSlot()
    def on_aug_images_dest_browse_button_click(self):
        """Aug images destination browser"""
        folder_name = QFileDialog \
            .getExistingDirectory(self,
                                  "Sauvegarder les images augmentées dans",
                                  options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.aug_images_dest_field.setText(folder_name)
            self.settings.setValue("aug_images_dest_field", folder_name)
            self.settings.sync()

    @pyqtSlot()
    def on_aug_seg_dest_browse_button_click(self):
        """Aug segs destination browser"""
        folder_name = QFileDialog \
            .getExistingDirectory(self,
                                  "Sauvegarder les masques augmentés dans",
                                  options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.aug_seg_dest_field.setText(folder_name)
            self.settings.setValue("aug_seg_dest_field", folder_name)
            self.settings.sync()

    @pyqtSlot()
    def on_train_images_browse_button_click(self):
        """Images to train on browser"""
        folder_name = QFileDialog \
            .getExistingDirectory(self, "Images d'entrainement",
                                  options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.train_images_field.setText(folder_name)
            self.settings.setValue("train_images_field", folder_name)
            self.settings.sync()

    @pyqtSlot()
    def on_train_seg_browse_button_click(self):
        """Segs to train on browser"""
        folder_name = QFileDialog \
            .getExistingDirectory(self, "Segmentations d'entrainement",
                                  options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.train_seg_field.setText(folder_name)
            self.settings.setValue("train_seg_field", folder_name)
            self.settings.sync()

    @pyqtSlot()
    def on_eval_images_browse_button_click(self):
        """Img to eval browser"""
        folder_name = QFileDialog \
            .getExistingDirectory(self, "Images d'évaluation",
                                  options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.eval_images_field.setText(folder_name)
            self.settings.setValue("eval_images_field", folder_name)
            self.settings.sync()

    @pyqtSlot()
    def on_eval_seg_browse_button_click(self):
        """Segs to eval browser"""
        folder_name = QFileDialog \
            .getExistingDirectory(self, "Segmentations d'évaluation",
                                  options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.eval_seg_field.setText(folder_name)
            self.settings.setValue("eval_seg_field", folder_name)
            self.settings.sync()

    @pyqtSlot()
    def on_existing_model_path_browse_button_click(self):
        """Existing model browser"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Charger un modèle")
        if file_name:
            self.qt_ui.existing_model_path_field.setText(file_name)
            self.settings.setValue("existing_model_path_field", file_name)
            self.settings.sync()

    @pyqtSlot()
    def on_save_model_path_browse_button_click(self):
        """Save model destination browser"""
        folder_name = QFileDialog \
            .getExistingDirectory(self, "Sauvegarder le modèle",
                                  options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.save_model_path_field.setText(folder_name)
            self.settings.setValue("save_model_path_field", folder_name)
            self.settings.sync()

    @pyqtSlot()
    def on_predict_model_path_browse_button_click(self):
        """Existing model to predict browser"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Charger un modèle")
        if file_name:
            self.qt_ui.predict_model_path_field.setText(file_name)
            self.settings.setValue("predict_model_path_field", file_name)
            self.settings.sync()

    @pyqtSlot()
    def on_predict_images_browse_button_click(self):
        """Img to predict browser"""
        folder_name = QFileDialog \
            .getExistingDirectory(self, "Images à prédire",
                                  options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.predict_images_field.setText(folder_name)
            self.settings.setValue("predict_images_field", folder_name)
            self.settings.sync()

    @pyqtSlot()
    def on_saved_seg_browse_button_click(self):
        """Seg images destination browser"""
        folder_name = QFileDialog \
            .getExistingDirectory(self, "Enregistrer les segmentations dans",
                                  options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.saved_seg_field.setText(folder_name)
            self.settings.setValue("saved_seg_field", folder_name)
            self.settings.sync()

    @pyqtSlot()
    def on_saved_sup_browse_button_click(self):
        """Overlay images destination browser"""
        folder_name = QFileDialog \
            .getExistingDirectory(self, "Enregistrer les superpositions dans",
                                  options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.saved_sup_field.setText(folder_name)
            self.settings.setValue("saved_sup_field", folder_name)
            self.settings.sync()

    # Classes prep
    @pyqtSlot()
    def on_add_class_button_click(self):
        """Adding a class to fusion"""
        folder_name = QFileDialog \
            .getExistingDirectory(self,
                                  "Choisissez un dossier contenant les masques "
                                  "d'une classe",
                                  options=QFileDialog.ShowDirsOnly)
        if folder_name:
            new_class = os.path.basename(folder_name)
            if new_class not in self.classes_folders:
                self.classes_folders[new_class] = folder_name + "/"
                self.qt_ui.classes_list.addItem(new_class)
            else:
                errormsg(typerr=QtWarningMsg,
                         msgerr="Une classe du nom de {} existe déjà dans la "
                                "liste !".format(new_class))

    @pyqtSlot()
    def on_remove_class_button_click(self):
        """Removing a class to fusion"""
        selection = self.qt_ui.classes_list.selectedItems()
        for item in selection:
            self.qt_ui.classes_list.takeItem(self.qt_ui.classes_list.row(item))
            del self.classes_folders[item.text()]

    @pyqtSlot()
    def on_fusion_button_click(self):
        """Fusion launcher"""

        # Parameters
        n_classes = len(self.classes_folders)
        scales = np.linspace(1, n_classes, n_classes).tolist()
        width = int(self.qt_ui.resize_width_spinbox.text())
        height = int(self.qt_ui.resize_height_spinbox.text())
        save_dir = self.qt_ui.mask_prep_field.text()

        worker = MaskFusionWorker(class_pathes=list(self.classes_folders
                                                    .values()),
                                  class_scales=scales,
                                  size=(width, height),
                                  save_to=save_dir)

        # Launching treatment
        self.set_progress_bar_state(True)
        self.check_all_available()  # Lock other buttons
        worker.signals.progressed.connect(self.update_progress_bar)
        worker.signals.finished.connect(self.treatment_done)

        self.thread_pool.start(worker)

    @pyqtSlot()
    def on_aug_button_click(self):
        """Image augmentation launcher"""

        # Saving settings
        self.settings.setValue("aug_images_nb_spinbox",
                               self.qt_ui.aug_images_nb_spinbox.value())
        self.settings.setValue("aug_rotation_range_spinbox",
                               self.qt_ui.aug_rotation_range_spinbox.value())
        self.settings.setValue("aug_horizontal_spinbox",
                               self.qt_ui.aug_horizontal_spinbox.value())
        self.settings.setValue("aug_vertical_spinbox",
                               self.qt_ui.aug_vertical_spinbox.value())
        self.settings.setValue("aug_shear_spinbox",
                               self.qt_ui.aug_shear_spinbox.value())
        self.settings.setValue("aug_zoom_spinbox",
                               self.qt_ui.aug_zoom_spinbox.value())
        self.settings.setValue("aug_width_spinbox",
                               self.qt_ui.aug_width_spinbox.value())
        self.settings.setValue("aug_height_spinbox",
                               self.qt_ui.aug_height_spinbox.value())
        self.settings.setValue("aug_fill_combobox",
                               self.qt_ui.aug_fill_combobox.currentText())

        # Parameters
        img_src = self.qt_ui.aug_images_source_field.text()
        seg_src = self.qt_ui.aug_seg_source_field.text()
        img_dest = self.qt_ui.aug_images_dest_field.text()
        seg_dest = self.qt_ui.aug_seg_dest_field.text()
        nb_img = self.qt_ui.aug_images_nb_spinbox.value()
        rotation = self.qt_ui.aug_rotation_range_spinbox.value()
        width = self.qt_ui.aug_horizontal_spinbox.value() / 100
        height = self.qt_ui.aug_vertical_spinbox.value() / 100
        shear = self.qt_ui.aug_shear_spinbox.value()
        zoom = self.qt_ui.aug_zoom_spinbox.value() / 100
        fill = self.qt_ui.aug_fill_combobox.currentText()
        size = (self.qt_ui.aug_width_spinbox.value(),
                self.qt_ui.aug_height_spinbox.value())

        worker = ImageAugmentationWorker(nb_img=nb_img, img_src=img_src,
                                         seg_src=seg_src, img_dest=img_dest,
                                         seg_dest=seg_dest, size=size,
                                         rotation=rotation, width=width,
                                         height=height, shear=shear, zoom=zoom,
                                         fill=fill)

        # Launching treatment
        self.set_progress_bar_state(True)
        self.check_all_available()  # Lock other buttons
        worker.signals.progressed.connect(self.update_progress_bar)
        worker.signals.finished.connect(self.treatment_done)
        worker.signals.error.connect(self.error_appened)
        self.thread_pool.start(worker)

    @pyqtSlot()
    def on_train_button_click(self):
        """Training launcher"""

        # Saving settings
        self.settings.setValue("width_model_spinbox",
                               self.qt_ui.width_model_spinbox.value())
        self.settings.setValue("height_model_spinbox",
                               self.qt_ui.height_model_spinbox.value())
        self.settings.setValue("batch_size_spinbox",
                               self.qt_ui.batch_size_spinbox.value())
        self.settings.setValue("step_epoch_spinbox",
                               self.qt_ui.step_epoch_spinbox.value())
        self.settings.setValue("epochs_spinbox",
                               self.qt_ui.epochs_spinbox.value())
        self.settings.setValue("nb_class_spinbox",
                               self.qt_ui.nb_class_spinbox.value())

        # Parameters
        img_src = self.qt_ui.train_images_field.text()
        seg_src = self.qt_ui.train_seg_field.text()
        existing = self.qt_ui.existing_model_path_field.text()
        new = self.qt_ui.model_combobox.currentText()
        width = self.qt_ui.width_model_spinbox.value()
        height = self.qt_ui.height_model_spinbox.value()
        batch = self.qt_ui.batch_size_spinbox.value()
        steps = self.qt_ui.step_epoch_spinbox.value()
        epochs = self.qt_ui.epochs_spinbox.value()
        checkpoint = self.qt_ui.save_model_path_field.text()
        nb_class = self.qt_ui.nb_class_spinbox.value()

        worker = TrainWorker(existing=existing, new=new, width=width,
                             height=height, img_src=img_src,
                             seg_src=seg_src, batch=batch, steps=steps,
                             epochs=epochs, checkpoint=checkpoint,
                             nb_class=nb_class)

        # Launching treatment
        self.set_progress_bar_state(True)
        self.check_all_available()  # Lock other buttons
        worker.signals.progressed.connect(self.update_progress_bar)
        worker.signals.log.connect(self.append_train_log)
        worker.signals.finished.connect(self.treatment_done)
        worker.signals.error.connect(self.error_appened)
        self.thread_pool.start(worker)

    @pyqtSlot()
    def on_eval_button_click(self):
        """Evaluation launcher"""

        # Parameters
        img_src = self.qt_ui.eval_images_field.text()
        seg_src = self.qt_ui.eval_seg_field.text()
        existing = self.qt_ui.existing_model_path_field.text()

        worker = EvalWorker(inp_images_dir=img_src, annotations_dir=seg_src,
                            checkpoints_path=existing)

        # Launching treatment
        self.set_progress_bar_state(True)
        self.check_all_available()  # Lock other buttons
        worker.signals.progressed.connect(self.update_progress_bar)
        worker.signals.log.connect(self.append_train_log)
        worker.signals.finished.connect(self.treatment_done)
        worker.signals.error.connect(self.error_appened)
        self.thread_pool.start(worker)

    @pyqtSlot()
    def on_predict_button_click(self):
        """Prediction launcher"""

        # Parameters
        existing = self.qt_ui.predict_model_path_field.text()
        img_src = self.qt_ui.predict_images_field.text()
        seg_dest = self.qt_ui.saved_seg_field.text()
        sup_dest = self.qt_ui.saved_sup_field.text()

        worker = PredictWorker(inp_dir=img_src, out_dir=seg_dest,
                               checkpoints_path=existing, colors=None,
                               sup_dir=sup_dest)

        # Launching treatment
        self.set_progress_bar_state(True)
        self.check_all_available()  # Lock other buttons
        worker.signals.progressed.connect(self.update_progress_bar)
        worker.signals.log.connect(self.append_predict_log)
        worker.signals.finished.connect(self.treatment_done)
        worker.signals.error.connect(self.error_appened)
        self.thread_pool.start(worker)

    ###########################################################################
    ### Slots (tab "Charbonnières")                                         ###
    ###########################################################################

    # Thumbnail extraction

    @pyqtSlot()
    def on_charb_extra_images_browse_button_click(self):
        """[CHARB] - Images for thumbnails generation browser"""
        folder_name = QFileDialog. \
            getExistingDirectory(self, "Images",
                                 options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.charb_extra_images_field.setText(folder_name)

    @pyqtSlot()
    def on_charb_extra_seg_browse_button_click(self):
        """[CHARB] - Segmentations for thumbnails generation browser"""
        folder_name = QFileDialog. \
            getExistingDirectory(self, "Segmentations",
                                 options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.charb_extra_seg_field.setText(folder_name)

    @pyqtSlot()
    def on_charb_extra_dataset_browse_button_click(self):
        """[CHARB] - Output dataset browser"""
        folder_name = QFileDialog. \
            getExistingDirectory(self, "Dataset",
                                 options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.charb_extra_dataset_field.setText(folder_name)

    @pyqtSlot()
    def on_charb_extra_proportion_slider_change(self):
        trainSize = self.qt_ui.charb_extra_proportion_slider.value()
        evalSize = 100 - trainSize
        self.qt_ui.charb_extra_proportion_label.setText(f'{trainSize} / {evalSize}')

    @pyqtSlot()
    def on_charb_extra_extract_button_click(self):
        """[CHARB] - Thumbnails extraction launcher"""

        # Getting parameters
        imagesPath = self.qt_ui.charb_extra_images_field.text()
        segmentationsPath = self.qt_ui.charb_extra_seg_field.text()
        datasetPath = self.qt_ui.charb_extra_dataset_field.text()
        idCharb = self.qt_ui.charb_extra_idcharb_spindbox.value()
        vigSize = int(self.qt_ui.charb_extra_vigsize_spinbox.value())
        intervalle = int(self.qt_ui.charb_extra_intervalle_spinbox.value())
        propTrain = self.qt_ui.charb_extra_proportion_slider.value()
        if self.qt_ui.charb_extra_1px_checkbox.isChecked():
            mode = '1px'
        elif self.qt_ui.charb_extra_4px_checkbox.isChecked():
            mode = '4px'

        # Saving settings
        self.settings.setValue("charb_extra_imagesPath", imagesPath)
        self.settings.setValue("charb_extra_segmentationsPath", segmentationsPath)
        self.settings.setValue("charb_extra_datasetPath", datasetPath)
        self.settings.setValue("charb_extra_propTrain", propTrain)

        self.settings.setValue("charb_common_idCharb", idCharb)
        self.settings.setValue("charb_common_vigSize", vigSize)
        self.settings.setValue("charb_common_intervalle", intervalle)
        self.settings.setValue("charb_common_mode", mode)

        # Creating extraction worker
        worker = ExtractionWorker(src_img=imagesPath, src_seg=segmentationsPath, idCharb=idCharb, propTrain=propTrain,
                                  dst=datasetPath, vigSize=vigSize, increment=intervalle, mode=mode)

        # Launching treatment
        self.set_progress_bar_state(True)
        self.check_all_available()  # Lock other buttons
        worker.signals.progressed.connect(self.update_progress_bar)
        worker.signals.log.connect(self.charb_append_extraction_log)
        worker.signals.finished.connect(self.treatment_done)
        worker.signals.error.connect(self.error_appened)
        self.thread_pool.start(worker)

    # Training and evaluation

    @pyqtSlot()
    def on_charb_train_traindata_browse_button_click(self):
        """[CHARB] - Training dataset browser"""
        folder_name = QFileDialog. \
            getExistingDirectory(self, "Jeu d'entraînement",
                                 options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.charb_train_traindata_field.setText(folder_name)

    @pyqtSlot()
    def on_charb_train_evaldata_browse_button_click(self):
        """[CHARB] - Evaluation dataset browser"""
        folder_name = QFileDialog. \
            getExistingDirectory(self, "Jeu d'évaluation",
                                 options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.charb_train_evaldata_field.setText(folder_name)

    @pyqtSlot()
    def on_charb_train_loadmodel_browse_button_click(self):
        """[CHARB] - Existing model browser"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Charger un modèle")
        if file_name:
            self.qt_ui.charb_train_loadmodel_field.setText(file_name)

    @pyqtSlot()
    def on_charb_train_savemodel_browse_button_click(self):
        """[CHARB] - Save model browser"""
        folder_name = QFileDialog. \
            getExistingDirectory(self, "Sauvegarder le modèle dans le dossier suivant",
                                 options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.charb_train_savemodel_field.setText(folder_name)

    @pyqtSlot()
    def on_charb_train_train_button_click(self):
        """[CHARB] - Training launcher"""

        # Parameters
        modelName = self.qt_ui.charb_train_model_combobox.currentText()
        trainDataPath = self.qt_ui.charb_train_traindata_field.text()
        evalDataPath = self.qt_ui.charb_train_evaldata_field.text()
        vigSize = int(self.qt_ui.charb_train_vigsize_spinbox.value())
        batchSize = int(self.qt_ui.charb_train_batchsize_spinbox.value())
        steps_per_epoch = int(self.qt_ui.charb_train_stepperepoch_spinbox.value())
        validation_steps = 20   # TODO?
        epochs = int(self.qt_ui.charb_train_epochs_spinbox.value())
        shuffle = self.qt_ui.charb_train_shuffle_checkbox.isChecked()
        saveModelAs = self.qt_ui.charb_train_savemodel_field.text()

        # Saving settings
        self.settings.setValue("charb_train_modelName", modelName)
        self.settings.setValue("charb_train_trainDataPath", trainDataPath)
        self.settings.setValue("charb_train_evalDataPath", evalDataPath)
        self.settings.setValue("charb_train_stepsPerEpoch", steps_per_epoch)
        self.settings.setValue("charb_train_validationSteps", validation_steps)
        self.settings.setValue("charb_train_epochs", epochs)
        self.settings.setValue("charb_train_shuffle", shuffle)
        self.settings.setValue("charb_train_saveModelAs", saveModelAs)

        self.settings.setValue("charb_common_vigSize", vigSize)
        self.settings.setValue("charb_common_batchSize", batchSize)

        # Creating training worker
        worker = CharbTrainWorker(modelName=modelName,
                                  trainDataPath=trainDataPath, evalDataPath=evalDataPath,
                                  vigSize=vigSize, batchSize=batchSize, steps_per_epoch=steps_per_epoch,
                                  validation_steps=validation_steps, epochs=epochs, shuffle=shuffle,
                                  saveModelAs=saveModelAs)

        # Launching treatment
        self.set_progress_bar_state(True)
        self.check_all_available()  # Lock other buttons
        worker.signals.progressed.connect(self.update_progress_bar)
        worker.signals.log.connect(self.charb_append_train_log)
        worker.signals.finished.connect(self.treatment_done)
        worker.signals.error.connect(self.error_appened)
        self.thread_pool.start(worker)

    @pyqtSlot()
    def on_charb_train_eval_button_click(self):
        """[CHARB] - Evaluation launcher"""

        # Getting parameters
        existingModelPath = self.qt_ui.charb_train_loadmodel_field.text()
        evalDataPath = self.qt_ui.charb_train_evaldata_field.text()
        vigSize = int(self.qt_ui.charb_train_vigsize_spinbox.value())
        batchSize = int(self.qt_ui.charb_train_batchsize_spinbox.value())

        # Saving settings
        self.settings.setValue("charb_eval_existingModelPath", existingModelPath)
        self.settings.setValue("charb_eval_evalDataPath", evalDataPath)

        self.settings.setValue("charb_common_vigSize", vigSize)
        self.settings.setValue("charb_common_batchSize", batchSize)

        # Creating evaluation worker
        worker = CharbEvalWorker(existingModelPath=existingModelPath, evalDataPath=evalDataPath,
                                 vigSize=vigSize, batchSize=batchSize)

        # Launching treatment
        self.set_progress_bar_state(True)
        self.check_all_available()  # Lock other buttons
        worker.signals.progressed.connect(self.update_progress_bar)
        worker.signals.log.connect(self.charb_append_train_log)
        worker.signals.finished.connect(self.treatment_done)
        worker.signals.error.connect(self.error_appened)
        self.thread_pool.start(worker)

    # Predictions

    @pyqtSlot()
    def on_charb_pred_model_browse_button_click(self):
        """[CHARB] - Existing model browser"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Charger un modèle")
        if file_name:
            self.qt_ui.charb_pred_model_field.setText(file_name)

    @pyqtSlot()
    def on_charb_pred_images_browse_button_click(self):
        """[CHARB] - Source images browser"""
        folder_name = QFileDialog. \
            getExistingDirectory(self, "Images à prédire",
                                 options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.charb_pred_images_field.setText(folder_name)

    @pyqtSlot()
    def on_charb_pred_seg_browse_button_click(self):
        """[CHARB] - Segmentations browser"""
        folder_name = QFileDialog. \
            getExistingDirectory(self, "Sauvegarder les segmentations dans...",
                                 options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.charb_pred_seg_field.setText(folder_name)

    @pyqtSlot()
    def on_charb_pred_sup_browse_button_click(self):
        """[CHARB] - Superpositions browser"""
        folder_name = QFileDialog. \
            getExistingDirectory(self, "Sauvegarder les superpositions dans...",
                                 options=QFileDialog.ShowDirsOnly)
        if folder_name:
            self.qt_ui.charb_pred_sup_field.setText(folder_name)

    @pyqtSlot()
    def on_charb_pred_predict_button_click(self):
        """[CHARB] - Predictions launcher"""

        # Getting parameters
        modelName = self.qt_ui.charb_pred_model_field.text()
        imagesPath = self.qt_ui.charb_pred_images_field.text()
        saveSegPath = self.qt_ui.charb_pred_seg_field.text()
        saveSupPath = self.qt_ui.charb_pred_sup_field.text()
        vigSize = int(self.qt_ui.charb_pred_vigsize_spinbox.value())
        intervalle = int(self.qt_ui.charb_pred_intervalle_spinbox.value())
        batchSize = int(self.qt_ui.charb_pred_batchsize_spinbox.value())
        if self.qt_ui.charb_pred_1px_checkbox.isChecked():
            mode = '1px'
        elif self.qt_ui.charb_pred_4px_checkbox.isChecked():
            mode = '4px'

        # Saving settings
        self.settings.setValue("charb_pred_modelName", modelName)
        self.settings.setValue("charb_pred_imagesPath", imagesPath)
        self.settings.setValue("charb_pred_saveSegPath", saveSegPath)
        self.settings.setValue("charb_pred_saveSupPath", saveSupPath)

        self.settings.setValue("charb_common_vigSize", vigSize)
        self.settings.setValue("charb_common_intervalle", intervalle)
        self.settings.setValue("charb_common_batchSize", batchSize)
        self.settings.setValue("charb_common_mode", mode)

        # Creating prediction worker
        worker = CharbPredictWorker(modelName=modelName, imagesPath=imagesPath,
                                    saveSegPath=saveSegPath, saveSupPath=saveSupPath,
                                    vigSize=vigSize, intervalle=intervalle, batchSize=batchSize, mode=mode)

        # Launching treatment
        self.set_progress_bar_state(True)
        self.check_all_available()  # Lock other buttons
        worker.signals.progressed.connect(self.update_progress_bar)
        worker.signals.log.connect(self.charb_append_predict_log)
        worker.signals.finished.connect(self.treatment_done)
        worker.signals.error.connect(self.error_appened)
        self.thread_pool.start(worker)

    ###########################################################################
    ###########################################################################
    ###########################################################################

    # UI management

    def update_progress_bar(self, value):
        """Update the current progression of the progress bar"""
        self.qt_ui.progress_bar.setValue(value)

    def treatment_done(self, msg=None):
        """Called when a treatment is done, notify and disable progress_bar"""

        # unlock other treatments
        self.set_progress_bar_state(False)

        self.check_all_available()

        if msg:
            QMessageBox.information(self,
                                    "Terminé",
                                    "{}\n".format(msg))

    def error_appened(self, msg=""):
        """Called when an error happens"""
        errormsg(typerr=QtCriticalMsg, msgerr=msg)
        print("ER:", msg)
        self.treatment_done()

    def clear_existing_model(self):
        '''
            Called when settings parameters for a new model,
            clears the existing model field to avoid confusion
        '''
        self.qt_ui.existing_model_path_field.setText("")

    def check_all_available(self):
        """Lock or unlock the treatment starting buttons"""
        self.check_fusion_available()
        self.check_aug_available()
        self.check_train_available()
        self.check_eval_available()
        self.check_predict_available()
        # Checks for tab "Charbonnières"
        self.charb_check_extract_available()
        self.charb_check_train_available()
        self.charb_check_eval_available()
        self.charb_check_predict_available()

    def set_progress_bar_state(self, enabled):
        """Lock or unlock the progress bar"""
        self.qt_ui.progress_bar.setEnabled(enabled)
        self.qt_ui.progress_bar.setValue(0)

    def append_train_log(self, line):
        """Train logging"""
        self.qt_ui.train_logs_textedit.appendPlainText(str(line))
        print(line)
        sys.stdout.flush()

    def append_predict_log(self, line):
        """Predict logging"""
        self.qt_ui.predict_logs_textedit.appendPlainText(str(line))
        print(line)
        sys.stdout.flush()

    def check_fusion_available(self):
        """
            Check that all the required fields are completed
            to launch a mask fusion
        """
        self.qt_ui.fusion_button.setEnabled(False)

        # A treatment is in progress
        if self.qt_ui.progress_bar.isEnabled():
            return

        # There is at least one class
        if self.qt_ui.classes_list.count() <= 0:
            return

        # Width and height are not zero
        width = self.qt_ui.resize_width_spinbox.value()
        height = self.qt_ui.resize_height_spinbox.value()
        if width < 1 or height < 1:
            return

        # Save path is accessible
        path = self.qt_ui.mask_prep_field.text()
        if not os.path.exists(path):
            return

        self.qt_ui.fusion_button.setEnabled(True)

    def check_aug_available(self):
        """
            Check that all the required fields are completed
            to launch an image augmentation
        """
        self.qt_ui.aug_button.setEnabled(False)

        # A treatment is in progress
        if self.qt_ui.progress_bar.isEnabled():
            return

        # Paths are accessibles
        path = self.qt_ui.aug_images_source_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.aug_seg_source_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.aug_images_dest_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.aug_seg_dest_field.text()
        if not os.path.exists(path):
            return

        # Parameters are OK
        nb_img = self.qt_ui.aug_images_nb_spinbox.value()
        rotation_range = self.qt_ui.aug_rotation_range_spinbox.value()
        horizontal = self.qt_ui.aug_horizontal_spinbox.value()
        vertical = self.qt_ui.aug_vertical_spinbox.value()
        zoom = self.qt_ui.aug_zoom_spinbox.value()
        shear = self.qt_ui.aug_shear_spinbox.value()
        width = self.qt_ui.aug_width_spinbox.value()
        height = self.qt_ui.aug_height_spinbox.value()
        if (nb_img <= 0 or
                rotation_range < 0 or rotation_range > 360 or
                horizontal < 0 or horizontal > 100 or
                vertical < 0 or vertical > 100 or
                zoom < 0 or zoom > 100 or
                shear < 0 or shear > 360 or
                width < 1 or height < 1):
            return

        self.qt_ui.aug_button.setEnabled(True)

    def check_train_available(self):
        """
            Check that all the required fields are completed
            to launch a training
        """
        self.qt_ui.train_button.setEnabled(False)

        # A treatment is in progress
        if self.qt_ui.progress_bar.isEnabled():
            return

        # Paths are accessibles
        path = self.qt_ui.train_images_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.train_seg_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.existing_model_path_field.text()
        if path and not os.path.exists(path):
            return
        path = self.qt_ui.save_model_path_field.text()
        if not os.path.exists(os.path.dirname(path)):
            return

        # Parameters are OK
        batch = self.qt_ui.batch_size_spinbox.value()
        steps = self.qt_ui.step_epoch_spinbox.value()
        epochs = self.qt_ui.epochs_spinbox.value()
        nb_class = self.qt_ui.nb_class_spinbox.value()
        if batch < 1 or steps < 1 or epochs < 1 or \
                nb_class < 1 or nb_class > 255:
            return

        self.qt_ui.train_button.setEnabled(True)

    def check_eval_available(self):
        """
            Check that all the required fields are completed
             to launch an evaluation
        """
        self.qt_ui.eval_button.setEnabled(False)

        # A treatment is in progress
        if self.qt_ui.progress_bar.isEnabled():
            return

        # Paths are accessibles
        path = self.qt_ui.eval_images_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.eval_seg_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.existing_model_path_field.text()
        if not os.path.exists(path):
            return

        nb_class = self.qt_ui.nb_class_spinbox.value()
        if nb_class < 1 or nb_class > 255:
            return

        self.qt_ui.eval_button.setEnabled(True)

    def check_predict_available(self):
        '''
            Check that all the required fields are completed
            to launch a prediction
        '''
        self.qt_ui.predict_button.setEnabled(False)

        # A treatment is in progress
        if self.qt_ui.progress_bar.isEnabled():
            return

        # Paths are accessibles
        path = self.qt_ui.predict_model_path_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.predict_images_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.saved_seg_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.saved_sup_field.text()
        if not os.path.exists(path):
            return

        self.qt_ui.predict_button.setEnabled(True)

    ###########################################################################
    ### UI management (tab "Charbonnières")                                 ###
    ###########################################################################

    def charb_append_extraction_log(self, line):
        """[CHARB] - Extraction logging"""
        self.qt_ui.charb_extra_log.appendPlainText(str(line))
        print(line)
        sys.stdout.flush()

    def charb_append_train_log(self, line):
        """[CHARB] - Train/Eval logging"""
        self.qt_ui.charb_train_log.appendPlainText(str(line))
        print(line)
        sys.stdout.flush()

    def charb_append_predict_log(self, line):
        """[CHARB] - Predict logging"""
        self.qt_ui.charb_pred_log.appendPlainText(str(line))
        print(line)
        sys.stdout.flush()

    def charb_check_extract_available(self):
        """
            [CHARB]
            Check that all the required fields are completed
            to launch thumbnail extraction
        """
        self.qt_ui.charb_extra_extract_button.setEnabled(False)

        # A treatment is in progress
        if self.qt_ui.progress_bar.isEnabled():
            return

        # Paths are accessibles
        path = self.qt_ui.charb_extra_images_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.charb_extra_seg_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.charb_extra_dataset_field.text()
        if not os.path.exists(path):
            return

        # Parameters are OK
        vigSize = self.qt_ui.charb_extra_vigsize_spinbox.value()
        intervalle = self.qt_ui.charb_extra_intervalle_spinbox.value()
        mode1px = self.qt_ui.charb_extra_1px_checkbox.isChecked()
        mode4px = self.qt_ui.charb_extra_4px_checkbox.isChecked()
        if vigSize < 32 or vigSize > 128 or vigSize % 16 != 0:
            return
        if intervalle < 1 or intervalle > 128:
            return
        if (mode1px and mode4px) or (not mode1px and not mode4px):
            return

        self.qt_ui.charb_extra_extract_button.setEnabled(True)

    def charb_check_train_available(self):
        """
            [CHARB]
            Check that all the required fields are completed
            to launch a training session
        """
        self.qt_ui.charb_train_train_button.setEnabled(False)

        # A treatment is in progress
        if self.qt_ui.progress_bar.isEnabled():
            return

        # Paths are accessibles
        path = self.qt_ui.charb_train_traindata_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.charb_train_evaldata_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.charb_train_savemodel_field.text()
        if not os.path.exists(path):
            return

        # Parameters are OK
        vigSize = self.qt_ui.charb_train_vigsize_spinbox.value()
        batchSize = self.qt_ui.charb_train_batchsize_spinbox.value()
        stepsPerEpoch = self.qt_ui.charb_train_stepperepoch_spinbox.value()
        epochs = self.qt_ui.charb_train_epochs_spinbox.value()
        if vigSize < 32 or vigSize > 128 or vigSize % 16 != 0:
            return
        if batchSize < 32 or batchSize > 256 or batchSize % 32 != 0:
            return
        if stepsPerEpoch < 1 or epochs < 1:
            return

        self.qt_ui.charb_train_train_button.setEnabled(True)

    def charb_check_eval_available(self):
        """
            [CHARB]
            Check that all the required fields are completed
            to launch an evaluation session
        """
        self.qt_ui.charb_train_eval_button.setEnabled(False)

        # A treatment is in progress
        if self.qt_ui.progress_bar.isEnabled():
            return

        # Paths are accessibles
        path = self.qt_ui.charb_train_loadmodel_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.charb_train_evaldata_field.text()
        if not os.path.exists(path):
            return

        self.qt_ui.charb_train_eval_button.setEnabled(True)

    def charb_check_predict_available(self):
        """
            [CHARB]
            Check that all the required fields are completed
            to launch predictions
        """
        self.qt_ui.charb_pred_predict_button.setEnabled(False)

        # A treatment is in progress
        if self.qt_ui.progress_bar.isEnabled():
            return

        # Paths are accessibles
        path = self.qt_ui.charb_pred_model_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.charb_pred_images_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.charb_pred_seg_field.text()
        if not os.path.exists(path):
            return
        path = self.qt_ui.charb_pred_sup_field.text()
        if not os.path.exists(path):
            return

        # Parameters are OK
        batchSize = self.qt_ui.charb_pred_batchsize_spinbox.value()
        vigSize = self.qt_ui.charb_pred_vigsize_spinbox.value()
        intervalle = self.qt_ui.charb_pred_intervalle_spinbox.value()
        mode1px = self.qt_ui.charb_pred_1px_checkbox.isChecked()
        mode4px = self.qt_ui.charb_pred_4px_checkbox.isChecked()
        if batchSize < 32 or batchSize > 256 or batchSize % 32 != 0:
            return
        if vigSize < 32 or vigSize > 128 or vigSize % 16 != 0:
            return
        if intervalle < 1 or intervalle > 128:
            return
        if (mode1px and mode4px) or (not mode1px and not mode4px):
            return

        self.qt_ui.charb_pred_predict_button.setEnabled(True)
