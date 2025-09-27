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
from ct_scan_window import CTScanThread, CTScanOptionsWindow



class SerialThread(QThread):
    """Thread to handle serial data in the background."""
    data_received = pyqtSignal(str)  # Signal to emit received data

    def __init__(self, serial_port):
        super().__init__()
        self.serial_port = serial_port
        self.running = True

    def run(self):
        """Run the thread to monitor serial data."""
        while self.running:
            if self.serial_port.in_waiting > 0:
                try:
                    data = self.serial_port.readline().decode().strip()
                    if data:
                        self.data_received.emit(data)  # Emit data when received
                except Exception as e:
                    print(f"Error reading serial data: {e}")

    def stop(self):
        """Stop the thread."""
        self.running = False
        self.wait()

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

        # Start animation for real-time updates
        self.ani = animation.FuncAnimation(self.figure, self.update_plot, interval=1000)

    def add_data_point(self, filament_current, ma_output):
        """Add new data points and refresh graph."""
        self.filament_current_values.append(float(filament_current))
        self.ma_values.append(float(ma_output))
        self.canvas.draw()

    def update_plot(self, frame):
        """Update plot with new data points."""
        self.ax.clear()
        self.ax.set_xlabel("Filament Current (A)")
        self.ax.set_ylabel("mA Output")
        self.ax.set_xlim(2, 4)  # Ensure x-axis stays fixed
        self.ax.set_ylim(0, 1.5)  # Adjust this limit as needed
        self.ax.grid()
        self.ax.scatter(self.filament_current_values, self.ma_values, color="blue", label="Data Points")
        self.ax.legend()
        self.canvas.draw()


class FluroXrayCTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("fluro_xray_ct.ui", self)  # Load the .ui file
        self.setWindowTitle("Fluro X-Ray CT")
        self.graph_window = None  # Initialize graph window as None
        # Serial communication variables
        self.serial_port = None
        self.serial_thread = None

        # Camera variables
        self.camera = None
        self.timer = None
        self.dark_frame = None
        self.flat_frame = None
        self.BeamOn = 0;
        self.lastFrame = None

        self.settings_cache = None
        self.settings_filename = "settings.txt"

        # Initialize UI elements and connect signals
        self.initialize_ui()

        # Initialize the camera
        self.initialize_camera()

    def initialize_ui(self):
        """Connect UI elements to their respective functions."""
        # Populate serial dropdown
        self.serial_dropdown.addItems(self.get_serial_ports())

        # Connect sliders
        self.gain_slider.valueChanged.connect(self.update_gain)
        self.exposure_slider.valueChanged.connect(self.update_exposure)
        self.ma_setting_slider.valueChanged.connect(self.on_ma_slider_change)
        self.kv_setting_slider.valueChanged.connect(self.on_kv_slider_change)
        self.filament_amps_slider.valueChanged.connect(self.on_filament_slider_change)

        # Connect combo box
        self.processing_dropdown.currentTextChanged.connect(self.update_processing_method)

        # Connect buttons
        self.connect_button.clicked.connect(self.connect_serial)
        self.disconnect_button.clicked.connect(self.disconnect_serial)
        self.median_stack_button.clicked.connect(self.take_median_stack)
        self.start_feed_button.clicked.connect(self.start_feed)
        self.take_exposure_button.clicked.connect(self.take_exposure)
        self.take_dark_button.clicked.connect(self.take_dark)
        self.take_flat_button.clicked.connect(self.take_flat)
        self.StartXray.clicked.connect(self.beam_on)
        self.cameraCalibration.clicked.connect(self.performCalibration)
        self.cutOffHighlevel.clicked.connect(self.findCutOfPoint)
        self.SecondaryCropbutton.clicked.connect(self.secondary_crop_after_undistortion)

        # Connect checkboxes
        self.rotate_checkbox.stateChanged.connect(self.toggle_rotation)
        self.high_quality_mode_checkbox.stateChanged.connect(self.toggle_high_quality_mode)

        # Set initial states
        self.disconnect_button.setEnabled(False)
        self.frame_buffer = []
        self.high_quality_mode = False

        self.ct_scan_options_window = None
        self.ct_scan_options.clicked.connect(self.open_ct_scan_options_window)






    def open_ct_scan_options_window(self):
        """Open the CT Scan Options window."""
        if self.ct_scan_options_window is None:
            self.ct_scan_options_window = CTScanOptionsWindow(self)
        self.ct_scan_options_window.show()













######################## INTERFACE STUFF

    def toggle_rotation(self, state):
        """Handle the rotate checkbox."""
        print(f"Rotation {'Enabled' if state == Qt.Checked else 'Disabled'}")


    def performCalibration(self):
        """Capture a single still image, stop the video feed, and display it."""
        self.stop_video_feed()  # Stop video feed if running
        self.calibrate_barrel_distortion()


    # Slider Callbacks
    def on_ma_slider_change(self, value):
        """Handle mA slider change."""
        self.save_or_update_setting("ma_slider",value);
        value = value / 100;
        self.send_command(f"mAOut:{value:.2f}")


    def on_kv_slider_change(self, value):
        """Handle kV slider change."""
        self.save_or_update_setting("kv_slider",value);
        self.send_command(f"kVOut:{int(value)}")


    def on_filament_slider_change(self, value):
        self.save_or_update_setting("filament_slider",value);
        """Handle filament slider change."""
        value = value / 100;
        self.send_command(f"filLim:{value:.2f}")















########################### SETTING MANAGEMENT

    def save_or_update_setting(self, name, value):
        """
        Save or update a specific setting.

        Parameters:
            name (str): The name of the setting.
            value (str): The value of the setting.
        """
        if self.settings_cache is None:
            self._load_settings_from_file()

        # Update the cache
        self.settings_cache[name] = str(value)

        # Save all settings back to the file
        try:
            with open(self.settings_filename, "w") as file:
                for key, val in self.settings_cache.items():
                    file.write(f"{key}={val}\n")
            print(f"Setting saved: {name}={value}")
        except Exception as e:
            print(f"Error saving setting: {e}")


    def load_setting(self, name):
        # Load settings from file if cache is not initialized
        if self.settings_cache is None:
            self._load_settings_from_file()

        # Return the setting from the cache
        value = self.settings_cache.get(name)
        return value


    def _load_settings_from_file(self):
        """
        Internal method to load settings from the file into memory.
        """
        self.settings_cache = {}
        try:
            with open(self.settings_filename, "r") as file:
                for line in file:
                    if line.strip():  # Ignore empty lines
                        key, value = line.strip().split("=", 1)
                        self.settings_cache[key] = value
            #print("Settings loaded into memory.")
        except FileNotFoundError:
            print(f"Settings file not found: {self.settings_file}. Starting with an empty cache.")
        except Exception as e:
            print(f"Error loading settings: {e}")



    def initialize_camera(self):
        """Initialize the ZWO camera."""
        gain = 10
        exposure_time = 10000

        try:
            asi.init(r"./ASICamera2.dll")  # Replace with the actual path to the SDK
            num_cameras = asi.get_num_cameras()
            if num_cameras == 0:
                QMessageBox.critical(self, "Camera Error", "No cameras detected!")
                return

            cameras = asi.list_cameras()
            print(f"Detected cameras: {cameras}")
            self.camera = asi.Camera(0)  # Connect to the first camera
            self.camera.set_image_type(asi.ASI_IMG_RAW16)  # Set default exposure (10 ms)
            self.camera.set_control_value(asi.ASI_EXPOSURE, exposure_time)  # Set default exposure (10 ms)
            self.camera.set_control_value(asi.ASI_GAIN, gain)         # Set default gain

            gain = self.load_setting("gain")
            exposure_time = self.load_setting("exposuretime")

            if gain:
                gain = int(gain)
            else:
                gain = 10;
            if exposure_time:
                exposure_time = int(exposure_time)
            else:
                exposure_time = 10000

            self.update_gain(gain)
            self.update_exposure(exposure_time)
            self.gain_slider.setValue(gain)
            self.exposure_slider.setValue(exposure_time)

            roi = self.select_camera_roi()
            if roi:
                bins = self.camera.get_bin()  # Get the current binning value
                image_type = self.camera.get_image_type()  # Get the current image type
                start_x, start_y, width, height = roi
                # Adjust width and height to meet requirements
                width = (width // 8) * 8  # Round down to the nearest multiple of 8
                height = (height // 2) * 2  # Round down to the nearest multiple of 2

                
                self.camera.set_roi(start_x, start_y, width, height, bins=bins, image_type=image_type)

                #self.camera.set_control_value(asi.ASI_START_X, start_x)
                #self.camera.set_control_value(asi.ASI_START_Y, start_y)
                #self.camera.set_control_value(asi.ASI_WIDTH, width)
                #self.camera.set_control_value(asi.ASI_HEIGHT, height)
                print("Camera configured with the selected ROI.")
            else:
                print("No ROI selected. Camera configuration aborted.")

        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize camera: {e}")
        

  
        




    def update_gain(self, value):
        """Update the camera gain based on the slider value."""
        if self.camera:
            self.camera.set_control_value(asi.ASI_GAIN, value)
            self.gain_label.setText(f"Gain: {value}")
            self.save_or_update_setting("gain",value);
            print(f"Gain set to: {value}")

    def update_exposure(self, value):
        """Update the camera exposure time based on the slider value."""
        if self.camera:
            self.camera.set_control_value(asi.ASI_EXPOSURE, value)
            self.exposure_label.setText(f"Exposure: {value} µs")
            self.save_or_update_setting("exposuretime",value);
            print(f"Exposure set to: {value} µs")

    def update_processing_method(self, method):
        """Update the image processing method."""
        self.scaling_method = method.lower()  # Store the selected method as lowercase
        print(f"Processing method set to: {self.scaling_method}")

    def toggle_high_quality_mode(self, state):
        """Enable or disable high-quality video mode."""
        self.high_quality_mode = state == Qt.Checked
        print(f"High-Quality Video Mode: {'Enabled' if self.high_quality_mode else 'Disabled'}")



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
            self.serial_thread = SerialThread(self.serial_port)
            self.serial_thread.data_received.connect(self.process_serial_data)  # Connect to the handler
            self.serial_thread.start()

            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)
            self.status_label.setText(f"Status: Connected to {port_name}")

            maslidersetting = self.load_setting("ma_slider");
            if(maslidersetting):
                self.ma_setting_slider.setValue(int(maslidersetting))
            
            kvslidersetting = self.load_setting("kv_slider");
            if(maslidersetting):
                self.kv_setting_slider.setValue(int(kvslidersetting))
            
            filamentslidersetting = self.load_setting("filament_slider"); 
            if(maslidersetting):
                self.filament_amps_slider.setValue(int(filamentslidersetting))


        except Exception as e:
            QMessageBox.critical(self, "Connection Error", str(e))

    def disconnect_serial(self):
        """Disconnect the serial port."""
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.serial_thread = None

        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.serial_port = None

        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.status_label.setText("Status: Not connected")

    def process_serial_data(self, data):
        """Handle data received from the serial port."""
        print(f"Received: {data}")
        if data.startswith("Status:ReadStats:"):
            stats = data.split(":")
            kev_stats = stats[2].split("/")
            ma_stats = stats[3].split("/")
            filament_stats = stats[4].split("/")
            
            # Update mA Setting label
           
            slider_val = self.ma_setting_slider.value() / 100
            self.MASetLabel.setText(f"Set: {slider_val:.2f} mA")
            self.MAReadLabel.setText(f"Read: {ma_stats[1]} mA")

            slider_val = self.kv_setting_slider.value()
            self.KVSetLabel.setText(f"Set: {slider_val:.2f} kV")
            self.KVReadLabel.setText(f"Read: {kev_stats[1]} kV")

            slider_val = self.filament_amps_slider.value() / 100
            self.AmpLimit.setText(f"Set: {slider_val:.2f} A")
            self.AmpRead.setText(f"Read: {filament_stats[1]} A")
            self.filamentAmpRead = filament_stats[1];


            filgraph = self.load_setting("filament_graph")

            if(filgraph == "1"):
                # Ensure the graph window is open
                if self.graph_window is None:
                    self.graph_window = GraphWindow()
                    self.graph_window.show()

                # Send new data points to graph
                self.graph_window.add_data_point(filament_stats[1], ma_stats[1])            
         
        else:
            print(f"Unknown response: {data}")



    def add_meter(self, label, min_val, max_val, layout, row, col):
        """Adds a meter (dial) with a label to the layout."""
        group_box = QGroupBox(label)
        group_layout = QVBoxLayout()
        group_box.setLayout(group_layout)

        dial = QDial()
        dial.setMinimum(min_val)
        dial.setMaximum(max_val)
        dial.setNotchesVisible(True)
        dial.setValue((min_val + max_val) // 2)  # Set default value
        group_layout.addWidget(dial)

        value_label = QLabel(f"{(min_val + max_val) // 2}")
        value_label.setAlignment(Qt.AlignCenter)
        group_layout.addWidget(value_label)

        layout.addWidget(group_box, row, col)




    def send_command(self, command):
        """Send a command via the serial port."""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.write((command + "\n").encode())
            print(f"Sent: {command}")






































    def capture_video_frame(self):
        """Capture a frame from the camera and update the live feed."""
        try:
            # Capture a frame in 16-bit RAW mode
            frame = self.camera.capture_video_frame()
            frame = self.applyAllCalibrationsOneSpot(frame)

            if self.high_quality_mode:
                # Add frame to the buffer
                self.frame_buffer.append(frame)
                if len(self.frame_buffer) >= 3:
                    # Compute median of the buffered frames
                    median_frame = np.median(np.stack(self.frame_buffer, axis=0), axis=0)
                    median_frame = median_frame.astype(np.uint16);
                    # Normal mode, display every frame
                    if(int(self.load_setting("realtime-deconvolution")) == 1):
                        median_frame = deconvolution_lib.deconvoluteFrame(median_frame)
                    self.update_feed(median_frame)
                    self.frame_buffer.clear()  # Clear the buffer after displaying
            else:
                # Normal mode, display every frame
                if(int(self.load_setting("realtime-deconvolution")) == 1):
                    frame = deconvolution_lib.deconvoluteFrame(frame)
                self.update_feed(frame)
        except Exception as e:
            print(f"Error capturing frame: {e}")




    def start_video_feed(self):
        """Starts the video feed and displays it in the QLabel."""
        if not self.camera:
            QMessageBox.warning(self, "Error", "Camera not initialized.")
            return

        try:
            self.camera.start_video_capture()

            # Start updating the feed with frames from the video stream
            if self.timer:
                self.timer.stop()

            self.timer = QTimer()
            self.timer.timeout.connect(self.capture_video_frame)
            self.timer.start(30)  # Update every 30ms (~33 FPS)

            print("Video feed started.")
            self.start_feed_button.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start video feed: {e}")
            print(f"Failed to start video feed: {e}")

    def stop_video_feed(self):
        """Stops the video feed."""
        if self.timer:
            self.timer.stop()
            self.timer = None

        if self.camera:
            self.camera.stop_video_capture()

        self.start_feed_button.setEnabled(True)
        print("Video feed stopped.")      

    def capture_single_exposure(self):
        """Capture a single still image, stop the video feed, and display it."""
        self.stop_video_feed()  # Stop video feed if running

        try:

            frame = self.camera.capture()  # Capture a single still frame
            frame = self.applyAllCalibrationsOneSpot(frame)

            if(int(self.load_setting("save-deconvolution")) == 1):
                frame = deconvolution_lib.deconvoluteFrame(frame)

            timestamp = datetime.now().strftime("%d-%m-%Y %H-%M")  # Format: dd-mm-YYYY HH:ii
            imsave(f"single_exposure{timestamp}.tiff", frame.astype(np.uint16))
            self.last_frame = frame  # Store the full 12-bit frame
            print("Single exposure captured.")
            self.update_feed(frame)  # Display the captured frame
        except Exception as e:
            print(f"Error capturing single exposure: {e}")





    def update_feed(self, frame):
        """Prepare and display a frame in the QLabel."""
        try:

            self.lastFrame = frame

            # Dynamically determine the actual bit depth range of the data
            frame_min, frame_max = np.min(frame), np.max(frame)
            #print(f"Frame range: min={frame_min}, max={frame_max}")

            # Default to 'contrast' if no method is set
            scaling_method = getattr(self, 'scaling_method', 'direct')

            # Apply the selected scaling method
            if scaling_method == 'direct':
                frame_8bit = self.scale_direct(frame)
            elif scaling_method == 'full range':
                frame_8bit = self.scale_full_range(frame)
            elif scaling_method == 'contrast':
                frame_8bit = self.scale_contrast_stretching(frame)
            elif scaling_method == 'gamma':
                frame_8bit = self.scale_gamma_correction(frame)
            elif scaling_method == 'histogram':
                frame_8bit = self.scale_histogram_equalization(frame)
            elif scaling_method == 'custom_range':
                frame_8bit = self.scale_custom_range(frame)
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}")

            # Convert to QImage
            height, width = frame_8bit.shape
            q_image = QImage(frame_8bit.data, width, height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)

            # Scale the pixmap to fit the QLabel while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.live_feed_label.size(),  # Use the QLabel's current size
                Qt.KeepAspectRatio,          # Maintain the aspect ratio
                Qt.SmoothTransformation      # Smooth scaling for better quality
            )

            # Display the scaled pixmap
            self.live_feed_label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"Error updating feed: {e}")

    def filBeamSpoolUp():
        #takes the current read filament amperage and set amperage and 
        #slowly ramps it up instead of hitting filamnet with huge spike
        setFilamentCurrent = self.filament_amps_slider.value() / 100
        currentFilamentAmps = self.filamentAmpRead
        


    def beam_on(self):
        if(self.BeamOn == 0): 
            self.BeamOn = 1;
            self.StartXray.setText(f"Xray supply ON!")
        else:
            self.BeamOn = 0;
            self.StartXray.setText(f"Xray supply off")

        self.send_command(f"BeamOn:{int(self.BeamOn)}")


















    def take_median_stack_ct(self,framenr,timestamp,stack_frame_count):
        self.stop_video_feed() 
        """
        Captures and processes a median stack of images specific for CT scans. Gets handled from ct_scan_window.py
        """
        
        median_image = self.capture_and_median_stack(stack_frame_count)
        if median_image is not None:
            print("Displaying the median-stacked image...")
            self.update_feed(median_image)  # Display the median-stacked image
            imsave(f"{timestamp}/{framenr}.tiff", median_image.astype(np.uint16))
            return True
        else:
            QMessageBox.warning(self, "Error", "Failed to create the median-stacked image.")
            return False




    def take_median_stack(self):
        self.stop_video_feed() 
        """
        Captures and processes a median stack of images.
        """
        num_exposures = int(self.load_setting("median_stack_frame_count"))  # Example: Set the number of exposures to capture
        median_image = self.capture_and_median_stack(num_exposures)
        if median_image is not None:
            print("Displaying the median-stacked image...")

            if(int(self.load_setting("save-deconvolution")) == 1):
                median_image = deconvolution_lib.deconvoluteFrame(median_image)

            self.update_feed(median_image)  # Display the median-stacked image
      
            timestamp = datetime.now().strftime("%d-%m-%Y %H-%M")  # Format: dd-mm-YYYY HH:ii
            imsave(f"median_stack_{timestamp}.tiff", median_image.astype(np.uint16))
        else:
            QMessageBox.warning(self, "Error", "Failed to create the median-stacked image.")

    def start_feed(self):
        self.start_feed_button.setEnabled(False)
        self.start_video_feed();

    def take_exposure(self):
        """Handle the Take Single Exposure button click."""
        self.capture_single_exposure();
        print("Take single exposure clicked!")
        self.send_command("TakeExposure")

    def capture_and_median_stack(self, num_exposures, use_raw = 0):
        """
        Captures multiple frames and returns the median-stacked image.
        Parameters:
        num_exposures (int): The number of exposures to capture (1 to 5).
        Returns:
        np.ndarray: The median-stacked image.
        """
        
        try:
            # Array to hold all captured frames
            frames = []

            for i in range(num_exposures):
                print(f"Capturing frame {i + 1} of {num_exposures}...")
                frame = self.camera.capture()  # Capture a raw frame
                frame = self.applyAllCalibrationsOneSpot(frame, use_raw)
                frames.append(frame)

            # Stack frames into a 3D array
            stacked_frames = np.stack(frames, axis=0)

            # Calculate the median across the stack
            median_image = np.median(stacked_frames, axis=0)

            print("Median stack completed.")
            return median_image.astype(np.uint16)  # Return as 16-bit array
        except Exception as e:
            print(f"Error in capture_and_median_stack: {e}")
            return None














































    def scale_custom_range(self, frame):
        """
        Scale the input frame based on the darkest and brightest spots.

        Parameters:
        frame (np.ndarray): The input 16-bit frame to scale.

        Returns:
        np.ndarray: The frame scaled to the range [0, 255].
        """
        try:
            # Ensure self.darkest_spot and self.brightest_spot are defined
            if not hasattr(self, 'darkest_spot') or not hasattr(self, 'brightest_spot'):
                raise ValueError("darkest_spot and brightest_spot are not set. Call findCutOfPoint first.")

            # Ensure self.darkest_spot is less than self.brightest_spot
            if self.darkest_spot >= self.brightest_spot:
                raise ValueError("Invalid range: darkest_spot must be less than brightest_spot.")

            # Clip the frame to the range [darkest_spot, brightest_spot]
            clipped_frame = np.clip(frame, self.darkest_spot, self.brightest_spot)

            # Scale the clipped frame to the range [0, 255]
            scaled_frame = (clipped_frame - self.darkest_spot) / (self.brightest_spot - self.darkest_spot) * 255
            return scaled_frame.astype(np.uint8)  # Return as an 8-bit frame

        except Exception as e:
            print(f"Error in scale_custom_range: {e}")
            return frame  # Return the original frame as fallback




    def scale_histogram_equalization(self,frame):
        """Apply histogram equalization to the frame scaled to 8-bit."""
        frame_min, frame_max = np.min(frame), np.max(frame)
        frame_normalized = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
        return cv2.equalizeHist(frame_normalized)

    def scale_gamma_correction(self, frame, gamma=2.2):
        """Apply gamma correction to scale the frame to 8-bit."""
        frame_min, frame_max = np.min(frame), np.max(frame)
        frame_normalized = (frame - frame_min) / (frame_max - frame_min)  # Normalize to [0, 1]
        frame_gamma_corrected = np.power(frame_normalized, 1 / gamma)
        return (frame_gamma_corrected * 255).astype(np.uint8)

    def scale_contrast_stretching(self, frame):
        """Apply contrast stretching to scale the frame to 8-bit."""
        frame_min, frame_max = np.min(frame), np.max(frame)
        if frame_max > frame_min:  # Avoid division by zero
            return (((frame - frame_min) / (frame_max - frame_min)) * 255).astype(np.uint8)
        else:
            return np.zeros_like(frame, dtype=np.uint8)

    def scale_full_range(self, frame):
        """Normalize the frame to the full 8-bit range."""
        return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def scale_direct(self, frame):
        """Scale a frame directly to 8-bit using truncation."""
        return (frame / 256).astype(np.uint8)  # For direct scaling, assumes 16-bit input.


    def findCutOfPoint(self, num_samples=500):
        """
        Determines the darkest and brightest spots in the last frame by sampling a subset of pixels.

        Parameters:
            num_samples (int): Number of pixels to sample across the frame.

        Updates:
            self.darkest_spot: The average of the minimum sampled pixel values.
            self.brightest_spot: The average of the maximum sampled pixel values.
        """
        import traceback

        try:
            # Validate num_samples
            if not isinstance(num_samples, int) or num_samples <= 0:
                print(f"Debug: Invalid num_samples detected: {num_samples}. Resetting to default (500).")
                num_samples = 500

            # Step 1: Validate the lastFrame
            if not hasattr(self, 'lastFrame') or self.lastFrame is None:
                raise ValueError("self.lastFrame is not available.")

            print(f"Debug: lastFrame shape: {self.lastFrame.shape}, dtype: {self.lastFrame.dtype}")

            if self.lastFrame.dtype != np.uint16:
                raise ValueError("self.lastFrame is not a 16-bit frame.")

            # Step 2: Validate dimensions
            height, width = self.lastFrame.shape
            if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
                raise ValueError(f"Invalid frame dimensions: height={height}, width={width}")

            # Step 3: Calculate total pixels and validate
            total_pixels = height * width
            if total_pixels <= 0:
                raise ValueError(f"Invalid total_pixels: {total_pixels}")

            # Step 4: Adjust num_samples
            num_samples = min(num_samples, total_pixels)
            print(f"Debug: Adjusted num_samples: {num_samples}")

            # Step 5: Flatten the frame and sample
            flat_frame = self.lastFrame.flatten()
            sampled_indices = np.random.choice(total_pixels, num_samples, replace=False)
            sampled_pixels = flat_frame[sampled_indices]
            print(f"Debug: Sampled pixels (first 10): {sampled_pixels[:10]}")

            # Step 6: Compute darkest and brightest spots
            darkest_spot = np.mean(np.partition(sampled_pixels, num_samples // 10)[:num_samples // 10])
            brightest_spot = np.mean(np.partition(sampled_pixels, -num_samples // 10)[-num_samples // 10:])
            print(f"Debug: Darkest spot: {darkest_spot}, Brightest spot: {brightest_spot}")

            # Step 7: Save results
            self.darkest_spot = darkest_spot
            self.brightest_spot = brightest_spot

        except Exception as e:
            print("Error in findCutOfPoint:")
            traceback.print_exc()
































    def applyAllCalibrationsOneSpot(self, frame, disable = 0):
        frame = self.undistort_image(frame)
        if(disable != 1):
            frame = self.process_frame_dark_flat(frame)
        return frame



     #this one for the basi barrel distortion fucntion
    def undistort_image(self, image):
        """
        Undistort a 16-bit image using the calibration data from `calibration.json`,
        apply a zoom factor, and apply the secondary crop if defined.

        Parameters:
        - image: Input image (16-bit numpy array).

        Returns:
        - processed_image: The undistorted, zoomed, and cropped 16-bit image.
        """
        calibration_file = "calibration.json"

        try:
            # Load calibration data if not already loaded
            if not hasattr(self, 'camera_matrix') or self.camera_matrix is None or self.dist_coeffs is None:
                print("Loading calibration data from file...")
                with open(calibration_file, 'r') as f:
                    calibration_data = json.load(f)

                self.camera_matrix = np.array(calibration_data["camera_matrix"])
                self.dist_coeffs = np.array(calibration_data["dist_coeffs"])
                self.zoom_factor = calibration_data.get("zoom_factor", 1.0)  # Default zoom factor
                print("Calibration data loaded successfully.")

            # Undistort the image
            h, w = image.shape[:2]
            new_camera_matrix = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), self.zoom_factor
            )[0]
            undistorted_image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)

            # Apply secondary crop
            crop_x = self.load_setting("secondary_crop_x")
            crop_y = self.load_setting("secondary_crop_y")
            crop_width = self.load_setting("secondary_crop_width")
            crop_height = self.load_setting("secondary_crop_height")

            if crop_x and crop_y and crop_width and crop_height:
                start_x = int(crop_x)
                start_y = int(crop_y)
                end_x = start_x + int(crop_width)
                end_y = start_y + int(crop_height)
                undistorted_image = undistorted_image[start_y:end_y, start_x:end_x]

            return undistorted_image.astype(np.uint16)

        except FileNotFoundError:
            print(f"Calibration file {calibration_file} not found. Run calibration first.")
            return image

        except Exception as e:
            print(f"Error during undistortion: {e}")
            return image

    def process_frame_dark_flat(self, raw_frame):
        """
        Processes a raw 16-bit frame using the dark and flat frames.

        Parameters:
        raw_frame (np.ndarray): The raw 16-bit frame to process.

        Returns:
        np.ndarray: The processed frame, ready for display or further processing.
        """
        try:
            # Load dark frame if not already loaded
            if not hasattr(self, 'dark_frame') or self.dark_frame is None:
                print("Dark frame not in memory, loading from file...")
                try:
                    self.dark_frame = np.array(imread('dark_frame.tiff'), dtype=np.float32)
                    print("Dark frame loaded successfully.")
                except Exception as e:
                    raise ValueError(f"Failed to load dark frame: {e}")

            # Load flat frame if not already loaded
            if not hasattr(self, 'flat_frame') or self.flat_frame is None:
                print("Flat frame not in memory, loading from file...")
                try:
                    self.flat_frame = np.array(imread('flat_frame.tiff'), dtype=np.float32)
                    print("Flat frame loaded successfully.")
                except Exception as e:
                    raise ValueError(f"Failed to load flat frame: {e}")

            # Ensure frames have matching dimensions
            if raw_frame.shape != self.dark_frame.shape or raw_frame.shape != self.flat_frame.shape:
                raise ValueError("Frame dimensions do not match the calibration frames.")

            # Convert raw_frame to float for accurate calculations
            raw_frame = raw_frame.astype(np.float32)

            # Subtract dark frame
            corrected_frame = raw_frame - self.dark_frame
            corrected_frame[corrected_frame < 0] = 0  # Clip negative values to zero

            # Normalize the flat frame to have an average value of 1
            flat_frame_normalized = self.flat_frame - self.dark_frame
            flat_frame_normalized[flat_frame_normalized < 0] = 0  # Clip negative values
            mean_flat = np.mean(flat_frame_normalized)
            if mean_flat > 0:
                flat_frame_normalized /= mean_flat
            else:
                raise ValueError("Flat frame normalization failed due to zero mean value.")

            # Avoid division by zero in the flat frame
            flat_frame_normalized[flat_frame_normalized == 0] = 1e-6

            # Apply flat field correction
            corrected_frame = corrected_frame / flat_frame_normalized

            # Clip the corrected frame to the valid range and cast back to uint16
            corrected_frame = np.clip(corrected_frame, 0, 65535).astype(np.uint16)

            return corrected_frame
        except Exception as e:
            print(f"Error processing frame: {e}")
            return raw_frame  # Return raw frame as fallback









































    def take_dark(self):
        self.stop_video_feed() 
        """Captures and saves a median-stacked dark frame."""
        try:
            num_exposures = 30  # Number of frames for the dark frame
            QMessageBox.information(self, "Dark Frame Capture", f"Capturing {num_exposures} dark frames...")
            # Capture and median stack
            self.dark_frame = self.capture_and_median_stack(num_exposures,1)
            
            if self.dark_frame is not None:
                # Save as 16-bit TIFF
                imsave('dark_frame.tiff', self.dark_frame.astype(np.uint16))
                QMessageBox.information(self, "Success", "Dark frame saved as 'dark_frame.tiff'.")
                print("Dark frame saved as 'dark_frame.tiff'.")
            else:
                QMessageBox.warning(self, "Error", "Failed to capture the dark frame.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error capturing dark frame: {e}")
            print(f"Error capturing dark frame: {e}")

    def take_flat(self):
        self.stop_video_feed() 
        """
        Captures and saves a median-stacked flat frame.
        """
        try:
            num_exposures = 30  # Number of frames for the flat frame
            QMessageBox.information(self, "Flat Frame Capture", f"Capturing {num_exposures} flat frames...")
            # Capture and median stack
            self.flat_frame = self.capture_and_median_stack(num_exposures,1)
            if self.flat_frame is not None:
                # Save as 16-bit TIFF
                imsave('flat_frame.tiff', self.flat_frame.astype(np.uint16))
                QMessageBox.information(self, "Success", "Flat frame saved as 'flat_frame.tiff'.")
                print("Flat frame saved as 'flat_frame.tiff'.")
            else:
                QMessageBox.warning(self, "Error", "Failed to capture the flat frame.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error capturing flat frame: {e}")
            print(f"Error capturing flat frame: {e}")




    def check_square_pattern(self, frame, grid_size=(30, 10), square_size=10.0):
        """
        Check for a square dot grid pattern in the given frame and calculate a quality score.

        Parameters:
        - frame: The input frame (8-bit grayscale, undistorted).
        - grid_size: Tuple (columns, rows) indicating the dot grid dimensions.
        - square_size: The size of each square in mm.

        Returns:
        - score: A percentage score (0-100%) indicating the pattern quality.
        """
        try:
            # Find the dot grid pattern
            ret, centers = cv2.findCirclesGrid(
                frame, grid_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID
            )

            if not ret:
                print("Square dot grid pattern not found.")
                return 0.0

            # Refine center detection
            centers = cv2.cornerSubPix(frame, centers, (11, 11), (-1, -1),
                                       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # Calculate distances between horizontal neighbors
            centers = centers.reshape(grid_size[1], grid_size[0], 2)  # Reshape to rows x cols x 2
            horizontal_distances = np.linalg.norm(np.diff(centers, axis=1), axis=2)
            vertical_distances = np.linalg.norm(np.diff(centers, axis=0), axis=2)

            # Calculate mean and standard deviation of all distances
            all_distances = np.concatenate([horizontal_distances.flatten(), vertical_distances.flatten()])
            mean_distance = all_distances.mean()
            std_dev = all_distances.std()

            # Calculate score based on uniformity
            score = max(0.0, min(100.0, 100 - (std_dev / mean_distance * 100)))

            # Visualize the detected pattern for user feedback
            vis_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis_image, grid_size, centers.reshape(-1, 1, 2), ret)
            cv2.imshow("Pattern Detection", vis_image)
            cv2.waitKey(500)

            print(f"Mean deviation: {std_dev:.6f}, Score: {score:.2f}%")
            return score
        except Exception as e:
            print(f"Error in square pattern check: {e}")
            return 0.0




    #THIS ONE SHOULD NOT NEED CALIBRATION, ONLY UNDISTORTION!
    def secondary_crop_after_undistortion(self):
        """
        Interactive function to select the secondary crop region for the undistorted image.
        The preview shown is the undistorted image scaled to 50%.
        """
        print("Starting secondary crop selection. Click two points to define the crop region.")
        print("Press 'r' to reset the selection and 'q' to cancel.")

        crop_points = []
        crop_selected = False

        crop_x = self.load_setting("secondary_crop_x")
        crop_y = self.load_setting("secondary_crop_y")
        crop_width = self.load_setting("secondary_crop_width")
        crop_height = self.load_setting("secondary_crop_height")



        def mouse_callback(event, x, y, flags, param):
            """Mouse event callback to capture crop points."""
            nonlocal crop_points, crop_selected
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(crop_points) < 2:
                    crop_points.append((x, y))
                    print(f"Point {len(crop_points)} selected: {x}, {y}")
                    if len(crop_points) == 2:
                        crop_selected = True

        cv2.namedWindow("Secondary Crop Selection")
        cv2.setMouseCallback("Secondary Crop Selection", mouse_callback)

        try:
            while True:
                # Capture a live frame, apply undistortion
                frame = self.camera.capture()
                frame_undistorted = self.undistort_image(frame)

                # Normalize and scale the frame for better visualization
                frame_8bit = cv2.normalize(frame_undistorted, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                frame_8bit_resized = cv2.resize(frame_8bit, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                vis_frame = cv2.cvtColor(frame_8bit_resized, cv2.COLOR_GRAY2BGR)

                # Draw crop points if selected
                for idx, point in enumerate(crop_points):
                    cv2.circle(vis_frame, point, 5, (0, 255, 0), -1)
                    cv2.putText(vis_frame, f"Point {idx+1}", (point[0] + 10, point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if len(crop_points) == 2:
                    # Draw rectangle around the selected crop region
                    cv2.rectangle(vis_frame, crop_points[0], crop_points[1], (0, 255, 0), 2)

                # Resize the window to match the scaled image dimensions
                h, w = frame_8bit_resized.shape[:2]
                cv2.resizeWindow("Secondary Crop Selection", w, h)

                cv2.imshow("Secondary Crop Selection", vis_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('r'):  # Reset selection
                    print("Resetting crop selection.")
                    crop_points = []
                    crop_selected = False

                elif key == ord('q'):  # Quit crop selection
                    print("Crop selection canceled.")
                    cv2.destroyAllWindows()
                    return None

                elif crop_selected:  # Finish selection
                    cv2.destroyAllWindows()
                    print("Crop selected successfully.")
                    # Scale points back up to the original resolution
                    start_x, start_y = crop_points[0]
                    end_x, end_y = crop_points[1]
                    start_x, start_y = int(start_x * 2), int(start_y * 2)
                    end_x, end_y = int(end_x * 2), int(end_y * 2)

                    width = abs(end_x - start_x)
                    height = abs(end_y - start_y)
                    self.save_or_update_setting("secondary_crop_x", min(start_x, end_x))
                    self.save_or_update_setting("secondary_crop_y", min(start_y, end_y))
                    self.save_or_update_setting("secondary_crop_width", width)
                    self.save_or_update_setting("secondary_crop_height", height)
                    crop = (min(start_x, end_x), min(start_y, end_y), width, height)
                    print(f"Selected crop: {crop}")
                    return crop

        except Exception as e:
            print(f"Error during secondary crop selection: {e}")
            cv2.destroyAllWindows()
            return None




    def calibrate_barrel_distortion(self):
        """
        Calibrate barrel distortion interactively using live images from the camera.
        Allows zooming in and out to adjust the FOV.
        Saves the zoom scale to calibration settings.
        """
        calibration_file = "calibration.json"
        print("Starting barrel distortion calibration...")
        print("Use the sliders to adjust distortion coefficients (k1, k2) and zoom scale.")
        print("Press 'f' to save calibration data and exit.")
        print("Press 'c' to check for square grid pattern and quality score.")
        print("Press 'q' to quit without saving.")

        # Dot grid parameters
        grid_size = (30, 20)  # Number of dots (columns, rows)
        square_size = 10.0  # Spacing between dots in mm

        # Capture an initial frame to determine dimensions
        frame = self.camera.capture()
        h, w = frame.shape[:2]

        # Create a basic camera matrix (assumes no skew, optical center at the image center)
        self.camera_matrix = np.array([
            [w, 0, w / 2],
            [0, w, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        # Initialize distortion coefficients (k1, k2 only)
        self.dist_coeffs = np.zeros(5, dtype=np.float32)

        # Load previous zoom scale if available
        self.zoom_scale = int(self.load_setting("zoom_scale")) or 1.0  # Default is 1.0 (no zoom)

        # Create a window for calibration
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        window_width = int(w * 0.5)
        window_height = int(h * 0.5)
        cv2.resizeWindow("Calibration", window_width, window_height)

        # Callback for trackbar adjustments
        def update_parameters(*args):
            try:
                # Get the slider values and scale them to the range -0.5 to 0.5
                k1 = (cv2.getTrackbarPos('Radial Distortion k1', 'Calibration') - 50) / 100.0
                k2 = (cv2.getTrackbarPos('Radial Distortion k2', 'Calibration') - 50) / 100.0
                self.dist_coeffs[:2] = [k1, k2]

                # Get zoom scale (ranges from 0.5 to 2.0, mapped from 0 to 150)
                zoom_slider_value = cv2.getTrackbarPos('Zoom Scale', 'Calibration')
                self.zoom_scale = -1 + (zoom_slider_value / 100.0)  # Map 0-150 to 0.5-2.0
            except cv2.error as e:
                print(f"Error updating parameters: {e}")

        # Create trackbars for distortion coefficients and zoom scale
        cv2.createTrackbar('Radial Distortion k1', "Calibration", 50, 100, update_parameters)
        cv2.createTrackbar('Radial Distortion k2', "Calibration", 50, 100, update_parameters)
        cv2.createTrackbar('Zoom Scale', "Calibration", int((self.zoom_scale - 0.5) * 100), 150, update_parameters)

        while True:
            # Capture live frames
            frame = self.camera.capture()
            frame_8bit = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Generate the new camera matrix with the current zoom scale
            self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), self.zoom_scale, (w, h)
            )

            # Apply the current distortion coefficients with the zoom scale
            undistorted = cv2.undistort(frame_8bit, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)

            # Display the undistorted frame
            cv2.imshow("Calibration", undistorted)

            key = cv2.waitKey(250)  # Refresh every 250ms

            if key == ord('f'):  # Save and exit
                print("Saving calibration data...")
                calibration_data = {
                    "camera_matrix": self.camera_matrix.tolist(),
                    "dist_coeffs": self.dist_coeffs.tolist(),
                    "zoom_scale": self.zoom_scale  # Save zoom scale
                }
                with open(calibration_file, 'w') as f:
                    json.dump(calibration_data, f)
                self.save_or_update_setting("zoom_scale", self.zoom_scale)
                print(f"Calibration data saved to {calibration_file}.")
                break
            elif key == ord('c'):  # Check for grid pattern and quality score
                score = self.check_square_pattern(undistorted, grid_size=grid_size, square_size=square_size)
                print(f"Square dot grid quality score: {score:.2f}%")
            elif key == ord('q'):  # Quit without saving
                print("Calibration canceled.")
                break

        cv2.destroyAllWindows()


    def select_camera_roi(self):
        """
        Interactive function to select the ROI (Region of Interest) for the camera.
        The preview is displayed at half the original size.

        Returns:
        - roi: Tuple (start_x, start_y, width, height) defining the ROI.
        """
        print("Starting ROI selection. Click two points to define the ROI (top-left and bottom-right).")
        print("Press 'r' to reset the selection and 'q' to cancel.")
        
        roi_points = []
        roi_selected = False

        # Load saved ROI settings if available
        roi_x = self.load_setting("roi_x")
        roi_y = self.load_setting("roi_y")
        roi_width = self.load_setting("roi_width")
        roi_height = self.load_setting("roi_height")

        if roi_x:
            roi = (int(roi_x), int(roi_y), int(roi_width), int(roi_height))
            print(roi)
            return roi

        def mouse_callback(event, x, y, flags, param):
            """Mouse event callback to capture ROI points."""
            nonlocal roi_points, roi_selected
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(roi_points) < 2:
                    roi_points.append((x * 2, y * 2))  # Scale back to full resolution
                    print(f"Point {len(roi_points)} selected: {roi_points[-1]}")
                    if len(roi_points) == 2:
                        roi_selected = True

        cv2.namedWindow("ROI Selection")
        cv2.setMouseCallback("ROI Selection", mouse_callback)

        try:
            while True:
                # Capture a live frame from the camera
                frame = self.camera.capture()
                frame = self.undistort_image(frame)
                frame_8bit = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Resize the frame to half its original size for display
                vis_frame = cv2.resize(frame_8bit, (frame_8bit.shape[1] // 2, frame_8bit.shape[0] // 2))
                vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)

                # Draw ROI points if selected
                for idx, point in enumerate(roi_points):
                    scaled_point = (point[0] // 2, point[1] // 2)  # Scale down for display
                    cv2.circle(vis_frame, scaled_point, 5, (0, 255, 0), -1)
                    cv2.putText(vis_frame, f"Point {idx+1}", (scaled_point[0] + 10, scaled_point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if len(roi_points) == 2:
                    # Draw rectangle around the selected ROI
                    scaled_start = (roi_points[0][0] // 2, roi_points[0][1] // 2)
                    scaled_end = (roi_points[1][0] // 2, roi_points[1][1] // 2)
                    cv2.rectangle(vis_frame, scaled_start, scaled_end, (0, 255, 0), 2)

                cv2.imshow("ROI Selection", vis_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('r'):  # Reset selection
                    print("Resetting ROI selection.")
                    roi_points = []
                    roi_selected = False

                elif key == ord('q'):  # Quit ROI selection
                    print("ROI selection canceled.")
                    cv2.destroyAllWindows()
                    return None

                elif roi_selected:  # Finish selection
                    cv2.destroyAllWindows()
                    print("ROI selected successfully.")
                    start_x, start_y = roi_points[0]
                    end_x, end_y = roi_points[1]
                    width = abs(end_x - start_x)
                    height = abs(end_y - start_y)
                    self.save_or_update_setting("roi_x", min(start_x, end_x))
                    self.save_or_update_setting("roi_y", min(start_y, end_y))
                    self.save_or_update_setting("roi_width", width)
                    self.save_or_update_setting("roi_height", height)
                    roi = (min(start_x, end_x), min(start_y, end_y), width, height)
                    print(f"Selected ROI: {roi}")
                    return roi

        except Exception as e:
            print(f"Error during ROI selection: {e}")
            cv2.destroyAllWindows()
            return None
















    def closeEvent(self, event):
        """Handle cleanup when the window is closed."""
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
