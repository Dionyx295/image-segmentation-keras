"""Application entrypoint"""

import sys
from PyQt5.QtWidgets import QApplication

from skyeye_segmentation.controller.main_window import MainWindow

if __name__ == '__main__':
    APP = QApplication(sys.argv)
    W = MainWindow()
    W.show()
    sys.exit(APP.exec_())
