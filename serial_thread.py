import serial
from PyQt5.QtCore import QThread, pyqtSignal

class SerialThread(QThread):
    """Background thread reading from a pyserial Serial and emitting lines as text."""
    data_received = pyqtSignal(str)

    def __init__(self, serial_port: serial.Serial):
        super().__init__()
        self.serial_port = serial_port
        self._running = True

    def run(self):
        while self._running and self.serial_port and self.serial_port.is_open:
            try:
                line = self.serial_port.readline().decode(errors='ignore').strip()
                if line:
                    self.data_received.emit(line)
            except Exception as e:
                # Emit an error marker so UI can surface/log it
                self.data_received.emit(f"SerialError:{e}")
                break

    def stop(self):
        self._running = False
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
        except Exception:
            pass
