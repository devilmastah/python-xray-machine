from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QLabel, QPushButton, QComboBox, QWidget, QMessageBox
from PyQt5.QtCore import QThread
import serial.tools.list_ports
from datetime import datetime
import os

class CTScanThread(QThread):
    def __init__(self, parent, steps):
        super().__init__()
        self.parent = parent
        self.steps = steps

    def run(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(timestamp, exist_ok=True)

        for framenr in range(1, self.steps + 1):
            print(f"Capturing frame {framenr} of {self.steps}")
            result = self.parent.main_window.take_median_stack_ct(framenr, timestamp, 3)
            if not result:
                print(f"Frame {framenr} capture failed. Aborting CT scan.")
                break


class CTScanOptionsWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.setWindowTitle("CT Scan Options")
        self.setGeometry(200, 200, 400, 300)
        self.main_window = main_window

        # Initialize serial port
        self.serial_port = None

        # Layout
        layout = QVBoxLayout()

        # Serial dropdown
        self.serial_dropdown = QComboBox()
        self.serial_dropdown.addItems(self.get_serial_ports())
        layout.addWidget(QLabel("Select Serial Port:"))
        layout.addWidget(self.serial_dropdown)

        # Connect and disconnect buttons
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_serial)
        layout.addWidget(self.connect_button)

        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self.disconnect_serial)
        self.disconnect_button.setEnabled(False)
        layout.addWidget(self.disconnect_button)

        # Steps dropdown
        self.steps_dropdown = QComboBox()
        self.steps_dropdown.addItems(["60", "90", "120", "180", "360"])
        layout.addWidget(QLabel("Select Steps:"))
        layout.addWidget(self.steps_dropdown)

        # Start CT Scan button
        self.start_scan_button = QPushButton("Start CT Scan")
        self.start_scan_button.clicked.connect(self.start_ct_scan)
        layout.addWidget(self.start_scan_button)

        # Set central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def get_serial_ports(self):
        """Get a list of available serial ports."""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def connect_serial(self):
        """Connect to the selected serial port."""
        port_name = self.serial_dropdown.currentText()
        if not port_name:
            QMessageBox.warning(self, "Error", "No serial port selected.")
            return

        try:
            self.serial_port = serial.Serial(port_name, baudrate=9600, timeout=1)
            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)
            QMessageBox.information(self, "Success", f"Connected to {port_name}")
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", str(e))

    def disconnect_serial(self):
        """Disconnect the serial port."""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.serial_port = None

        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        QMessageBox.information(self, "Disconnected", "Serial port disconnected.")

    def start_ct_scan(self):
        """Handle the CT scan start button click."""
        steps = int(self.steps_dropdown.currentText())

        # Ensure the main window camera is initialized
        if not self.main_window.camera:
            QMessageBox.warning(self, "Error", "Camera not initialized.")
            return

        # Run CT scan in a separate thread
        self.ct_scan_thread = CTScanThread(self, steps)
        self.ct_scan_thread.start()
        QMessageBox.information(self, "CT Scan", "CT scan started.")

