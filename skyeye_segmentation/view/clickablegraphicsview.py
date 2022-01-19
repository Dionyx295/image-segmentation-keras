# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:36:23 2022

@author: Jean-Malo
"""
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal


class ClickableGraphicsView(QtWidgets.QGraphicsView):
    
    clicked = pyqtSignal()
    
    def mousePressEvent(self, event):
        self.clicked.emit()
        QtWidgets.QGraphicsView.mousePressEvent(self, event)