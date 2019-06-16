# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 02:20:21 2018

@author: keshav
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QApplication, QPushButton

app = QApplication([])
app.setStyle('Fusion')
app.setStyleSheet("QPushButton { margin: 100ex; }")
palette = QPalette()
palette.setColor(QPalette.ButtonText, Qt.red)
app.setPalette(palette)
button = QPushButton('Click To Record Audio')
button.show()
app.exec_()