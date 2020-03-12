import sys

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import qInstallMessageHandler, QtWarningMsg, QThreadPool, QtCriticalMsg, QSettings

from skyeye_segmentation.controller.errormsg import errormsg
from skyeye_segmentation.view import main_window
from skyeye_segmentation.controller.skyeye_func import *

from datetime import datetime
from pathlib import Path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Creating log file and redirecting stdout
        self.stdout_original = sys.stdout
        work_dir = os.path.join(Path(os.getcwd()).parent, "log")
        if not os.path.isdir(work_dir):
            os.mkdir(work_dir)
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y %Hh%Mm%Ss")
        log_file = os.path.join(work_dir, "logs_"+dt_string+".txt")
        sys.stdout = open(log_file, 'a')
        sys.stderr = open(log_file, 'a')

        self.ui = main_window.Ui_MainWindow()
        self.ui.setupUi(self)
        self.threadpool = QThreadPool()

        self.settings = QSettings("Polytech Tours", "SkyEye")
        self.load_settings()

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

        ## Augmentation
        self.ui.aug_button.clicked.connect(self.on_aug_button_click)

        ## Training
        self.ui.train_button.clicked.connect(self.on_train_button_click)

        ## Evaluation
        self.ui.eval_button.clicked.connect(self.on_eval_button_click)

        ## Prediction
        self.ui.predict_button.clicked.connect(self.on_predict_button_click)

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
        self.ui.model_combobox.currentTextChanged.connect(self.clear_existing_model)
        self.ui.width_model_spinbox.valueChanged.connect(self.clear_existing_model)
        self.ui.height_model_spinbox.valueChanged.connect(self.clear_existing_model)
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

        # List of available models
        model_names = model_from_name.keys()
        self.ui.model_combobox.addItems(model_names)

        # Error message handler
        #qInstallMessageHandler(errormsg)

        self.check_all_available()
        self.show()

    # QMainWindow closeEvent override
    def closeEvent(self, event):
        # Confirmation
        if self.ui.progress_bar.isEnabled():
            msg = "Voulez-vous quitter et arrêter le traitement en cours ?"
        else:
            msg = "Voulez-vous vraiment quitter ?"
        reply = QMessageBox.question(self, 'Quitter ?',
                                     msg, QMessageBox.Yes, QMessageBox.Cancel)
        if reply == QMessageBox.Yes:
            raise SystemExit(0) # Terminate the process and other potential threads
        else:
            event.ignore()

    # Settings loader
    def load_settings(self):
        # Path fields
        self.ui.aug_images_source_field.setText(self.settings.value("aug_images_source_field"))
        self.ui.aug_seg_source_field.setText(self.settings.value("aug_seg_source_field"))
        self.ui.aug_images_dest_field.setText(self.settings.value("aug_images_dest_field"))
        self.ui.aug_seg_dest_field.setText(self.settings.value("aug_seg_dest_field"))
        self.ui.train_images_field.setText(self.settings.value("train_images_field"))
        self.ui.train_seg_field.setText(self.settings.value("train_seg_field"))
        self.ui.eval_images_field.setText(self.settings.value("eval_images_field"))
        self.ui.eval_seg_field.setText(self.settings.value("eval_seg_field"))
        self.ui.existing_model_path_field.setText(self.settings.value("existing_model_path_field"))
        self.ui.save_model_path_field.setText(self.settings.value("save_model_path_field"))
        self.ui.predict_model_path_field.setText(self.settings.value("predict_model_path_field"))
        self.ui.predict_images_field.setText(self.settings.value("predict_images_field"))
        self.ui.saved_seg_field.setText(self.settings.value("saved_seg_field"))
        self.ui.saved_sup_field.setText(self.settings.value("saved_sup_field"))

        # Augmentation parameters
        value = self.settings.value("aug_images_nb_spinbox")
        if value:
            self.ui.aug_images_nb_spinbox.setValue(value)
        value = self.settings.value("aug_rotation_range_spinbox")
        if value:
            self.ui.aug_rotation_range_spinbox.setValue(value)
        value = self.settings.value("aug_horizontal_spinbox")
        if value:
            self.ui.aug_horizontal_spinbox.setValue(value)
        value = self.settings.value("aug_vertical_spinbox")
        if value:
            self.ui.aug_vertical_spinbox.setValue(value)
        value = self.settings.value("aug_shear_spinbox")
        if value:
            self.ui.aug_shear_spinbox.setValue(value)
        value = self.settings.value("aug_zoom_spinbox")
        if value:
            self.ui.aug_zoom_spinbox.setValue(value)
        value = self.settings.value("aug_width_spinbox")
        if value:
            self.ui.aug_width_spinbox.setValue(value)
        value = self.settings.value("aug_height_spinbox")
        if value:
            self.ui.aug_height_spinbox.setValue(value)
        value = self.settings.value("aug_fill_combobox")
        if value:
            self.ui.aug_fill_combobox.setCurrentText(value)

        # Train settings
        value = self.settings.value("width_model_spinbox")
        if value:
            self.ui.width_model_spinbox.setValue(value)
        value = self.settings.value("height_model_spinbox")
        if value:
            self.ui.height_model_spinbox.setValue(value)
        value = self.settings.value("batch_size_spinbox")
        if value:
            self.ui.batch_size_spinbox.setValue(value)
        value = self.settings.value("step_epoch_spinbox")
        if value:
            self.ui.step_epoch_spinbox.setValue(value)
        value = self.settings.value("epochs_spinbox")
        if value:
            self.ui.epochs_spinbox.setValue(value)
        value = self.settings.value("nb_class_spinbox")
        if value:
            self.ui.nb_class_spinbox.setValue(value)

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
            self.settings.setValue("aug_images_source_field", folderName)
            self.settings.sync()

    @pyqtSlot()
    def on_aug_seg_source_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Segmentations sources",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.aug_seg_source_field.setText(folderName)
            self.settings.setValue("aug_seg_source_field", folderName)
            self.settings.sync()

    @pyqtSlot()
    def on_aug_images_dest_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Sauvegarder les images augmentées dans",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.aug_images_dest_field.setText(folderName)
            self.settings.setValue("aug_images_dest_field", folderName)
            self.settings.sync()

    @pyqtSlot()
    def on_aug_seg_dest_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Sauvegarder les masques augmentés dans",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.aug_seg_dest_field.setText(folderName)
            self.settings.setValue("aug_seg_dest_field", folderName)
            self.settings.sync()

    @pyqtSlot()
    def on_train_images_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Images d'entrainement",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.train_images_field.setText(folderName)
            self.settings.setValue("train_images_field", folderName)
            self.settings.sync()

    @pyqtSlot()
    def on_train_seg_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Segmentations d'entrainement",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.train_seg_field.setText(folderName)
            self.settings.setValue("train_seg_field", folderName)
            self.settings.sync()

    @pyqtSlot()
    def on_eval_images_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Images d'évaluation",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.eval_images_field.setText(folderName)
            self.settings.setValue("eval_images_field", folderName)
            self.settings.sync()

    @pyqtSlot()
    def on_eval_seg_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Segmentations d'évaluation",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.eval_seg_field.setText(folderName)
            self.settings.setValue("eval_seg_field", folderName)
            self.settings.sync()

    @pyqtSlot()
    def on_existing_model_path_browse_button_click(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Charger un modèle")
        if fileName:
            self.ui.existing_model_path_field.setText(fileName)
            self.settings.setValue("existing_model_path_field", fileName)
            self.settings.sync()

    @pyqtSlot()
    def on_save_model_path_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Sauvegarder le modèle",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.save_model_path_field.setText(folderName)
            self.settings.setValue("save_model_path_field", folderName)
            self.settings.sync()

    @pyqtSlot()
    def on_predict_model_path_browse_button_click(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Charger un modèle")
        if fileName:
            self.ui.predict_model_path_field.setText(fileName)
            self.settings.setValue("predict_model_path_field", fileName)
            self.settings.sync()

    @pyqtSlot()
    def on_predict_images_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Images à prédire",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.predict_images_field.setText(folderName)
            self.settings.setValue("predict_images_field", folderName)
            self.settings.sync()

    @pyqtSlot()
    def on_saved_seg_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Enregistrer les segmentations dans",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.saved_seg_field.setText(folderName)
            self.settings.setValue("saved_seg_field", folderName)
            self.settings.sync()

    @pyqtSlot()
    def on_saved_sup_browse_button_click(self):
        folderName = QFileDialog.getExistingDirectory(self, "Enregistrer les superpositions dans",
                                                      options=QFileDialog.ShowDirsOnly)
        if folderName:
            self.ui.saved_sup_field.setText(folderName)
            self.settings.setValue("saved_sup_field", folderName)
            self.settings.sync()

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
        scales = np.linspace(1, n_classes, n_classes).tolist()
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
        # Saving settings
        self.settings.setValue("aug_images_nb_spinbox", self.ui.aug_images_nb_spinbox.value())
        self.settings.setValue("aug_rotation_range_spinbox", self.ui.aug_rotation_range_spinbox.value())
        self.settings.setValue("aug_horizontal_spinbox", self.ui.aug_horizontal_spinbox.value())
        self.settings.setValue("aug_vertical_spinbox", self.ui.aug_vertical_spinbox.value())
        self.settings.setValue("aug_shear_spinbox", self.ui.aug_shear_spinbox.value())
        self.settings.setValue("aug_zoom_spinbox", self.ui.aug_zoom_spinbox.value())
        self.settings.setValue("aug_width_spinbox", self.ui.aug_width_spinbox.value())
        self.settings.setValue("aug_height_spinbox", self.ui.aug_height_spinbox.value())
        self.settings.setValue("aug_fill_combobox", self.ui.aug_fill_combobox.currentText())

        # Parameters
        img_src = self.ui.aug_images_source_field.text()
        seg_src = self.ui.aug_seg_source_field.text()
        img_dest = self.ui.aug_images_dest_field.text()
        seg_dest = self.ui.aug_seg_dest_field.text()
        nb_img = self.ui.aug_images_nb_spinbox.value()
        rotation = self.ui.aug_rotation_range_spinbox.value()
        width = self.ui.aug_horizontal_spinbox.value() / 100
        height = self.ui.aug_vertical_spinbox.value() / 100
        shear = self.ui.aug_shear_spinbox.value()
        zoom = self.ui.aug_zoom_spinbox.value() /100
        fill = self.ui.aug_fill_combobox.currentText()
        size = (self.ui.aug_width_spinbox.value(), self.ui.aug_height_spinbox.value())
        worker = ImageAugmentationWorker(nb_img=nb_img, img_src=img_src, seg_src=seg_src, img_dest=img_dest,
                                         seg_dest=seg_dest, size=size, rotation=rotation, width=width, height=height,
                                         shear=shear, zoom=zoom, fill=fill)

        # Launching treatment
        self.set_progress_bar_state(True)
        self.check_all_available()  # Lock other buttons
        worker.signals.progressed.connect(self.update_progress_bar)
        worker.signals.finished.connect(self.treatment_done)
        worker.signals.error.connect(self.error_appened)
        self.threadpool.start(worker)

    @pyqtSlot()
    def on_train_button_click(self):
        # Saving settings
        self.settings.setValue("width_model_spinbox", self.ui.width_model_spinbox.value())
        self.settings.setValue("height_model_spinbox", self.ui.height_model_spinbox.value())
        self.settings.setValue("batch_size_spinbox", self.ui.batch_size_spinbox.value())
        self.settings.setValue("step_epoch_spinbox", self.ui.step_epoch_spinbox.value())
        self.settings.setValue("epochs_spinbox", self.ui.epochs_spinbox.value())
        self.settings.setValue("nb_class_spinbox", self.ui.nb_class_spinbox.value())

        # Parameters
        img_src = self.ui.train_images_field.text()
        seg_src = self.ui.train_seg_field.text()
        existing = self.ui.existing_model_path_field.text()
        new = self.ui.model_combobox.currentText()
        width = self.ui.width_model_spinbox.value()
        height = self.ui.height_model_spinbox.value()
        batch = self.ui.batch_size_spinbox.value()
        steps = self.ui.step_epoch_spinbox.value()
        epochs = self.ui.epochs_spinbox.value()
        checkpoint = self.ui.save_model_path_field.text()
        nb_class = self.ui.nb_class_spinbox.value()

        worker = TrainWorker(existing=existing, new=new, width=width, height=height, img_src=img_src,
                             seg_src=seg_src, batch=batch, steps=steps, epochs=epochs, checkpoint=checkpoint,
                             nb_class=nb_class)

        # Launching treatment
        self.set_progress_bar_state(True)
        self.check_all_available()  # Lock other buttons
        worker.signals.progressed.connect(self.update_progress_bar)
        worker.signals.log.connect(self.append_train_log)
        worker.signals.finished.connect(self.treatment_done)
        worker.signals.error.connect(self.error_appened)
        self.threadpool.start(worker)

    @pyqtSlot()
    def on_eval_button_click(self):
        # Parameters
        img_src = self.ui.eval_images_field.text()
        seg_src = self.ui.eval_seg_field.text()
        existing = self.ui.existing_model_path_field.text()
        worker = EvalWorker(inp_images_dir=img_src, annotations_dir=seg_src, checkpoints_path=existing)

        # Launching treatment
        self.set_progress_bar_state(True)
        self.check_all_available()  # Lock other buttons
        worker.signals.progressed.connect(self.update_progress_bar)
        worker.signals.log.connect(self.append_train_log)
        worker.signals.finished.connect(self.treatment_done)
        worker.signals.error.connect(self.error_appened)
        self.threadpool.start(worker)

    @pyqtSlot()
    def on_predict_button_click(self):
        #inp_dir=None, out_dir=None,checkpoints_path=None, colors=None

        # Parameters
        existing = self.ui.predict_model_path_field.text()
        img_src = self.ui.predict_images_field.text()
        seg_dest = self.ui.saved_seg_field.text()
        sup_dest = self.ui.saved_sup_field.text()
        worker = PredictWorker(inp_dir=img_src, out_dir=seg_dest, checkpoints_path=existing, colors=None,
                               sup_dir=sup_dest)

        # Launching treatment
        self.set_progress_bar_state(True)
        self.check_all_available()  # Lock other buttons
        worker.signals.progressed.connect(self.update_progress_bar)
        worker.signals.log.connect(self.append_predict_log)
        worker.signals.finished.connect(self.treatment_done)
        worker.signals.error.connect(self.error_appened)
        self.threadpool.start(worker)

    ### UI management ###
    '''
        Update the current progression of the progress bar
    '''
    def update_progress_bar(self, value):
        self.ui.progress_bar.setValue(value)

    '''
        Called when a treatment is done, notify and disable progress_bar
    '''
    def treatment_done(self, msg=None):
        # unlock other treatments
        self.set_progress_bar_state(False)

        self.check_all_available()

        if msg:
            QMessageBox.information(self,
                                    "Terminé",
                                    "{}\n".format(msg))

    '''
        Called when an error happens
    '''
    def error_appened(self, msg=""):
        errormsg(typerr=QtCriticalMsg, msgerr=msg, contexte="")
        print("ER:", msg)
        self.treatment_done()

    '''
        Called when settings parameters for a new model,
        clears the existing model field to avoid confusion
    '''
    def clear_existing_model(self):
        self.ui.existing_model_path_field.setText("")

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
        Train logging
    '''
    def append_train_log(self, line):
        self.ui.train_logs_textedit.appendPlainText(str(line))
        print(line)
        sys.stdout.flush()

    '''
        Predict logging
    '''
    def append_predict_log(self, line):
        self.ui.predict_logs_textedit.appendPlainText(str(line))
        print(line)
        sys.stdout.flush()

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
        if not os.path.exists(os.path.dirname(path)):
            return

        # Parameters are OK
        batch = self.ui.batch_size_spinbox.value()
        steps = self.ui.step_epoch_spinbox.value()
        epochs = self.ui.epochs_spinbox.value()
        nb_class = self.ui.nb_class_spinbox.value()
        if batch < 1 or steps < 1 or epochs < 1 or \
                nb_class < 1 or nb_class > 255:
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

        nb_class = self.ui.nb_class_spinbox.value()
        if nb_class < 1 or nb_class > 255:
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
