import sys
import os
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import pyqtSlot, qInstallMessageHandler, QtWarningMsg, QThreadPool
from skyeye_segmentation.controller.errormsg import errormsg

from skyeye_segmentation.view import main_window
from skyeye_segmentation.controller.skyeye_func import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = main_window.Ui_MainWindow()
        self.ui.setupUi(self)
        self.threadpool = QThreadPool()

        # Connections to slots
        ## File and folder browsers
        self.ui.mask_prep_browse_button.clicked.connect(self.on_mask_prep_browse_button_click)
        self.ui.aug_images_source_browse_button.clicked.connect(self.on_aug_images_source_browse_button_click)
        self.ui.aug_seg_source_browse_button.clicked.connect(self.on_aug_seg_source_browse_button_click)
        self.ui.aug_images_dest_browse_button.clicked.connect(self.on_aug_images_dest_browse_button_click)
        self.ui.aug_seg_dest_browse_button.clicked.connect(self.on_aug_seg_dest_browse_button_click)
        self.ui.train_images_browse_button.clicked.connect(self.on_train_images_browse_button_click)
        self.ui.train_seg_browse_button.clicked.connect(self.on_train_seg_browse_button_click)
        self.ui.eval_images_browse_button.clicked.connect(self.on_eval_images_browse_button_click)
        self.ui.eval_seg_browse_button.clicked.connect(self.on_eval_seg_browse_button_click)
        self.ui.existing_model_path_browse_button.clicked.connect(self.on_existing_model_path_browse_button_click)
        self.ui.save_model_path_browse_button.clicked.connect(self.on_save_model_path_browse_button_click)
        self.ui.predict_model_path_browse_button.clicked.connect(self.on_predict_model_path_browse_button_click)
        self.ui.predict_images_browse_button.clicked.connect(self.on_predict_images_browse_button_click)
        self.ui.saved_seg_browse_button.clicked.connect(self.on_saved_seg_browse_button_click)
        self.ui.saved_sup_browse_button.clicked.connect(self.on_saved_sup_browse_button_click)

        ## Classes prep
        self.ui.add_class_button.clicked.connect(self.on_add_class_button_click)
        self.ui.remove_class_button.clicked.connect(self.on_remove_class_button_click)
        self.ui.fusion_button.clicked.connect(self.on_fusion_button_click)

        ## Training
        self.ui.train_button.clicked.connect(self.on_train_button_click)

        ## UI management
        # Mask fusion
        self.ui.classes_list.model().rowsInserted.connect(self.check_fusion_available)
        self.ui.classes_list.model().rowsRemoved.connect(self.check_fusion_available)
        self.ui.resize_height_spinbox.valueChanged.connect(self.check_fusion_available)
        self.ui.resize_width_spinbox.valueChanged.connect(self.check_fusion_available)
        self.ui.mask_prep_field.textChanged.connect(self.check_fusion_available)
        # Image augmentation
        self.ui.aug_images_source_field.textChanged.connect(self.check_aug_available)
        self.ui.aug_seg_source_field.textChanged.connect(self.check_aug_available)
        self.ui.aug_images_dest_field.textChanged.connect(self.check_aug_available)
        self.ui.aug_seg_dest_field.textChanged.connect(self.check_aug_available)
        self.ui.aug_images_nb_spinbox.valueChanged.connect(self.check_aug_available)
        self.ui.aug_rotation_range_spinbox.valueChanged.connect(self.check_aug_available)
        self.ui.aug_horizontal_spinbox.valueChanged.connect(self.check_aug_available)
        self.ui.aug_vertical_spinbox.valueChanged.connect(self.check_aug_available)
        self.ui.aug_zoom_spinbox.valueChanged.connect(self.check_aug_available)
        self.ui.aug_shear_spinbox.valueChanged.connect(self.check_aug_available)
        self.ui.aug_width_spinbox.valueChanged.connect(self.check_aug_available)
        self.ui.aug_height_spinbox.valueChanged.connect(self.check_aug_available)
        # Training
        self.ui.train_images_field.textChanged.connect(self.check_train_available)
        self.ui.train_seg_field.textChanged.connect(self.check_train_available)
        self.ui.width_model_spinbox.valueChanged.connect(self.check_train_available)
        self.ui.height_model_spinbox.valueChanged.connect(self.check_train_available)
        self.ui.existing_model_path_field.textChanged.connect(self.check_train_available)
        self.ui.batch_size_spinbox.valueChanged.connect(self.check_train_available)
        self.ui.step_epoch_spinbox.valueChanged.connect(self.check_train_available)
        self.ui.epochs_spinbox.valueChanged.connect(self.check_train_available)
        self.ui.save_model_path_field.textChanged.connect(self.check_train_available)
        # Evaluation
        self.ui.eval_images_field.textChanged.connect(self.check_eval_available)
        self.ui.eval_seg_field.textChanged.connect(self.check_eval_available)
        self.ui.existing_model_path_field.textChanged.connect(self.check_eval_available)
        # Prediction
        self.ui.predict_model_path_field.textChanged.connect(self.check_predict_available)
        self.ui.predict_images_field.textChanged.connect(self.check_predict_available)
        self.ui.saved_seg_field.textChanged.connect(self.check_predict_available)
        self.ui.saved_sup_field.textChanged.connect(self.check_predict_available)

        # Dictionnary to bind classes and their path
        self.classes_folders = {}

        # Error message handler
        qInstallMessageHandler(errormsg)

        self.check_all_available()
        self.show()

    # Slots

    ## File and folder browsers
    @pyqtSlot()
    def on_mask_prep_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Sauvegarder les masques dans",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.mask_prep_field.setText(folderName)

    @pyqtSlot()
    def on_aug_images_source_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Images sources",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.aug_images_source_field.setText(folderName)

    @pyqtSlot()
    def on_aug_seg_source_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Segmentations sources",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.aug_seg_source_field.setText(folderName)

    @pyqtSlot()
    def on_aug_images_dest_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Sauvegarder les images augmentées dans",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.aug_images_dest_field.setText(folderName)

    @pyqtSlot()
    def on_aug_seg_dest_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Sauvegarder les masques augmentés dans",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.aug_seg_dest_field.setText(folderName)

    @pyqtSlot()
    def on_train_images_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Images d'entrainement",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.train_images_field.setText(folderName)

    @pyqtSlot()
    def on_train_seg_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Segmentations d'entrainement",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.train_seg_field.setText(folderName)

    @pyqtSlot()
    def on_eval_images_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Images d'évaluation",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.eval_images_field.setText(folderName)

    @pyqtSlot()
    def on_eval_seg_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Segmentations d'évaluation",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.eval_seg_field.setText(folderName)

    @pyqtSlot()
    def on_existing_model_path_browse_button_click(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Charger un modèle")
        if fileName:
            self.ui.existing_model_path_field.setText(fileName)

    @pyqtSlot()
    def on_save_model_path_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Sauvegarder le modèle",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.save_model_path_field.setText(folderName)

    @pyqtSlot()
    def on_predict_model_path_browse_button_click(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Charger un modèle")
        if fileName:
            self.ui.predict_model_path_field.setText(fileName)

    @pyqtSlot()
    def on_predict_images_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Images à prédire",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.predict_images_field.setText(folderName)

    @pyqtSlot()
    def on_saved_seg_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Enregistrer les segmentations dans",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.saved_seg_field.setText(folderName)

    @pyqtSlot()
    def on_saved_sup_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Enregistrer les superpositions dans",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.saved_sup_field.setText(folderName)

    ## Classes prep
    @pyqtSlot()
    def on_add_class_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Choisissez un dossier contenant les masques d'une classe",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            new_class = os.path.basename(folderName)
            if (new_class not in self.classes_folders):
                self.classes_folders[new_class] = folderName + "/"
                self.ui.classes_list.addItem(new_class)
            else:
                errormsg(typerr=QtWarningMsg,
                         msgerr="Une classe du nom de {} existe déjà dans la liste !".format(new_class),
                         contexte="")

    @pyqtSlot()
    def on_remove_class_button_click(self):
        selection = self.ui.classes_list.selectedItems()
        for item in selection:
            self.ui.classes_list.takeItem(self.ui.classes_list.row(item))
            del self.classes_folders[item.text()]

    @pyqtSlot()
    def on_fusion_button_click(self):
        # Parameters
        n_classes = len(self.classes_folders)
        scales = np.linspace(0, n_classes, n_classes + 1).tolist()
        width = int(self.ui.resize_width_spinbox.text())
        height = int(self.ui.resize_height_spinbox.text())
        save_dir = self.ui.mask_prep_field.text()

        worker = MaskFusionWorker(class_pathes=list(self.classes_folders.values()),
                                  class_scales=scales,
                                  size=(width, height),
                                  save_to=save_dir)

        # Launching treatment
        self.set_progress_bar_state(True)
        self.check_all_available() # Lock other buttons
        worker.signals.progressed.connect(self.update_progress_bar)
        worker.signals.finished.connect(self.treatment_done)
        self.threadpool.start(worker)

    @pyqtSlot()
    def on_aug_button_click(self):
        # Parameters
        img_src = self.ui.aug_images_source_field.text()
        seg_src = self.ui.aug_seg_source_field.text()
        img_dest = self.ui.aug_images_dest_field.text()
        seg_dest = self.ui.aug_seg_dest_field.text()

        worker = ImageAugmentationWorker(nb_img=1, img_src="", seg_src="", img_dest="", seg_dest="", size=(10,10),
                     rotation=90, width=0.25, height=0.25, shear=10, zoom=0.1, fill='reflect')

    @pyqtSlot()
    def on_train_button_click(self):
        self.ui.training_logs_textedit.append("Début de l'entrainement !")

    ### UI management ###
    '''
        Update the current progression of the progress bar
    '''

    def update_progress_bar(self, value):
        self.ui.progress_bar.setValue(value)

    '''
        Called when a treatment is done, notify and disable progress_bar
    '''

    def treatment_done(self, msg=""):
        # unlock other treatments
        self.set_progress_bar_state(False)

        self.check_all_available()

        if msg:
            QMessageBox.information(self,
                                    "Terminé",
                                    "{}\n".format(msg))

    '''
        Lock or unlock the treatment starting buttons
    '''
    def check_all_available(self):
        self.check_fusion_available()
        self.check_aug_available()
        self.check_train_available()
        self.check_eval_available()
        self.check_predict_available()

    '''
        Lock or unlock the progress bar
    '''
    def set_progress_bar_state(self, enabled):
        self.ui.progress_bar.setEnabled(enabled)
        self.ui.progress_bar.setValue(0)

    '''
        Check that all the required fields are completed to launch a mask fusion
    '''
    def check_fusion_available(self):
        self.ui.fusion_button.setEnabled(False)

        # A treatment is in progress
        if self.ui.progress_bar.isEnabled():
            return

        # There is at least one class
        if self.ui.classes_list.count() <= 0:
            return

        # Width and height are not zero
        width = self.ui.resize_width_spinbox.value()
        height = self.ui.resize_height_spinbox.value()
        if width < 1 or height < 1:
            return

        # Save path is accessible
        path = self.ui.mask_prep_field.text()
        if not os.path.exists(path):
            return

        self.ui.fusion_button.setEnabled(True)

    '''
        Check that all the required fields are completed to launch an image augmentation
    '''
    def check_aug_available(self):
        self.ui.aug_button.setEnabled(False)

        # A treatment is in progress
        if self.ui.progress_bar.isEnabled():
            return

        # Paths are accessibles
        path = self.ui.aug_images_source_field.text()
        if not os.path.exists(path):
            return
        path = self.ui.aug_seg_source_field.text()
        if not os.path.exists(path):
            return
        path = self.ui.aug_images_dest_field.text()
        if not os.path.exists(path):
            return
        path = self.ui.aug_seg_dest_field.text()
        if not os.path.exists(path):
            return

        # Parameters are OK
        nb_img = self.ui.aug_images_nb_spinbox.value()
        rotation_range = self.ui.aug_rotation_range_spinbox.value()
        horizontal = self.ui.aug_horizontal_spinbox.value()
        vertical = self.ui.aug_vertical_spinbox.value()
        zoom = self.ui.aug_zoom_spinbox.value()
        shear = self.ui.aug_shear_spinbox.value()
        width = self.ui.aug_width_spinbox.value()
        height = self.ui.aug_height_spinbox.value()
        if (nb_img <= 0 or
                rotation_range < 0 or rotation_range > 360 or
                horizontal < 0 or horizontal > 100 or
                vertical < 0 or vertical > 100 or
                zoom < 0 or zoom > 100 or
                shear < 0 or shear > 360 or
                width < 1 or height < 1):
            return

        self.ui.aug_button.setEnabled(True)

    '''
        Check that all the required fields are completed to launch a training
    '''
    def check_train_available(self):
        self.ui.train_button.setEnabled(False)

        # A treatment is in progress
        if self.ui.progress_bar.isEnabled():
            return

        # Paths are accessibles
        path = self.ui.train_images_field.text()
        if not os.path.exists(path):
            return
        path = self.ui.train_seg_field.text()
        if not os.path.exists(path):
            return
        path = self.ui.existing_model_path_field.text()
        if path and not os.path.exists(path):
            return
        path = self.ui.save_model_path_field.text()
        if not os.path.exists(path):
            return

        # Parameters are OK
        batch = self.ui.batch_size_spinbox.value()
        steps = self.ui.step_epoch_spinbox.value()
        epochs = self.ui.epochs_spinbox.value()
        if batch < 1 or steps < 1 or epochs < 1:
            return

        self.ui.train_button.setEnabled(True)

    '''
        Check that all the required fields are completed to launch an evaluation
    '''
    def check_eval_available(self):
        self.ui.eval_button.setEnabled(False)

        # A treatment is in progress
        if self.ui.progress_bar.isEnabled():
            return

        # Paths are accessibles
        path = self.ui.eval_images_field.text()
        if not os.path.exists(path):
            return
        path = self.ui.eval_seg_field.text()
        if not os.path.exists(path):
            return
        path = self.ui.existing_model_path_field.text()
        if not os.path.exists(path):
            return

        self.ui.eval_button.setEnabled(True)

    '''
        Check that all the required fields are completed to launch a prediction
    '''
    def check_predict_available(self):
        self.ui.predict_button.setEnabled(False)

        # A treatment is in progress
        if self.ui.progress_bar.isEnabled():
            return

        # Paths are accessibles
        path = self.ui.predict_model_path_field.text()
        if not os.path.exists(path):
            return
        path = self.ui.predict_images_field.text()
        if not os.path.exists(path):
            return
        path = self.ui.saved_seg_field.text()
        if not os.path.exists(path):
            return
        path = self.ui.saved_sup_field.text()
        if not os.path.exists(path):
            return

        self.ui.predict_button.setEnabled(True)
