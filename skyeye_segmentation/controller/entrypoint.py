import sys
import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtCore import pyqtSlot, qInstallMessageHandler, QtWarningMsg
from skyeye_segmentation.controller.msgerreur import messagederreur

from skyeye_segmentation.view import main_window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = main_window.Ui_MainWindow()
        self.ui.setupUi(self)

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


        self.ui.train_button.clicked.connect(self.on_train_button_click)

        self.classes_folders = {}

        # Error message handler
        qInstallMessageHandler(messagederreur)
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
            self.ui.save_model_path_browse_button.setText(folderName)

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
            if(new_class not in self.classes_folders):
                self.classes_folders[new_class] = folderName
                self.ui.classes_list.addItem(new_class)
            else:
                messagederreur(typerr=QtWarningMsg,
                               msgerr="Une classe du nom de {} existe déjà dans la liste !".format(new_class),
                               contexte="")

    @pyqtSlot()
    def on_remove_class_button_click(self):
        selection = self.ui.classes_list.selectedItems()
        for item in selection:
            self.ui.classes_list.takeItem(self.ui.classes_list.row(item))
            del self.classes_folders[item.text()]


    @pyqtSlot()
    def on_train_button_click(self):
        self.ui.training_logs_textedit.append("Début de l'entrainement !")



app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())