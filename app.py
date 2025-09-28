import sys
import cv2
from datetime import datetime
import zwoasi as asi
import numpy as np
from tifffile import (imsave, imread)  # For saving 16-bit TIFF images
import serial
import os
import json
from PyQt5.uic import loadUi
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import glob
import deconvolution_lib

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QCheckBox, QWidget, QGroupBox, QComboBox, QDial, QGridLayout, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer 
from PyQt5.QtGui import QPixmap, QImage

from serial_thread import SerialThread

class GraphWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Filament Current vs. mA Output")
        self.setGeometry(100, 100, 800, 600)

        # Create figure and axis for Matplotlib
        self.figure, self.ax = plt.subplots()
        self.ax.set_xlabel("Filament Current (A)")
        self.ax.set_ylabel("mA Output")
        self.ax.set_xlim(0, 5)  # X-axis range (Filament Current)
        self.ax.set_ylim(0, 1.5)  # Adjust as needed for mA values
        self.ax.grid()

        # Data storage
        self.filament_current_values = []
        self.ma_values = []

        # Create a Matplotlib canvas inside the PyQt window
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

class FluroXrayCTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... existing initialization code ...
        self.serial_thread = None
        self.camera = None
        # rest of init remains unchanged

    # ... all existing methods remain unchanged, except SerialThread definition removed ...

    def closeEvent(self, event):
        """Handle cleanup when the window is closed."""
        try:
            if hasattr(self, 'serial_thread') and self.serial_thread:
                try:
                    self.serial_thread.stop()
                except Exception:
                    pass
        except Exception:
            pass
        if self.camera:
            self.camera.close()
        event.accept()

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    deconvolution_lib.init_model("best_deconvolution_model.pth")
    main_window = FluroXrayCTApp()
    main_window.show()
    sys.exit(app.exec_())
