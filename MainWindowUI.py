from PyQt6.QtWidgets import QMainWindow, QSpinBox, QDoubleSpinBox, QLabel, QGridLayout, QHBoxLayout, QVBoxLayout
from PyQt6 import uic
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from pubsub import pub
import logging
import cv2
import numpy as np

from message_types import Topics

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)

class MainWindowUI(QMainWindow):
    def __init__(self):
        super(MainWindowUI, self).__init__()
        self.ui = uic.loadUi("mainwindow.ui", self)
        self.ui.showMaximized()
        self.ui.setWindowTitle("Face Recognition and Detection")
        
        # Initialize variables
        self.current_image = None
        self.current_tab = "recognition"  # Default tab
        
        # Setup connections and subscriptions
        self.setup_connections()
        self.setup_subscriptions()

    def setup_connections(self):
        # Tab change connection
        self.tabWidget.currentChanged.connect(self.on_tab_changed)
        
        # Recognition tab connections
        self.btnUploadRecognition.clicked.connect(self.handle_recognition_upload)
        self.btnROCRecognition.clicked.connect(self.handle_recognition_rocgnition)
        
        # Detection tab connections
        self.btnUploadDetection.clicked.connect(self.handle_detection_upload)
        self.btnROCDetection.clicked.connect(self.handle_detection_roc)

    def setup_subscriptions(self):
        pub.subscribe(self.update_recognition_display, Topics.UPDATE_RECOGNITION_DISPLAY)
        pub.subscribe(self.update_detection_display, Topics.UPDATE_DETECTION_DISPLAY)
        pub.subscribe(self.update_status_bar, Topics.UPDATE_STATUS_BAR)

    def on_tab_changed(self, index):
        tab_name = self.tabWidget.tabText(index).lower()
        self.current_tab = tab_name
        pub.sendMessage(Topics.TAB_CHANGED, tab=tab_name)
        logging.info(f"Tab changed to {tab_name}")

    def handle_recognition_upload(self):
        file_path = self.get_image_file_path()
        if file_path:
            image = self.load_image(file_path)
            if image is not None:
                self.current_image = image
                pub.sendMessage(Topics.UPLOAD_IMAGE_RECOGNITION, 
                              image_path=file_path,
                              image_data=image)
                pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                              message="Image loaded for recognition",
                              timeout=3000)

    def handle_detection_upload(self):
        file_path = self.get_image_file_path()
        if file_path:
            image = self.load_image(file_path)
            if image is not None:
                self.current_image = image
                pub.sendMessage(Topics.UPLOAD_IMAGE_DETECTION,
                              image_path=file_path,
                              image_data=image)
                pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                              message="Image loaded for detection",
                              timeout=3000)

    def get_image_file_path(self):
        return QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )[0]

    def load_image(self, file_path):
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Failed to load image")
            logging.info(f"Successfully loaded image: {file_path}")
            return image
        except Exception as e:
            logging.error(f"Error loading image {file_path}: {str(e)}")
            pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                          message="Error loading image",
                          timeout=3000)
            return None

    def display_image(self, label, image):
        """Display an image on a QLabel"""
        if image is None:
            return
            
        # Convert OpenCV image (BGR) to RGB for Qt
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        height, width = rgb_image.shape[:2]
        
        # Create QImage from numpy array
        q_img = QImage(rgb_image.data, width, height, width * 3, QImage.Format.Format_RGB888)
        
        # Create and set pixmap
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale pixmap to fit the label while maintaining aspect ratio
        pixmap = pixmap.scaled(label.width(), label.height(), 
                              Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        
        # Set pixmap to label
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def update_recognition_display(self, image_data):
        self.display_image(self.labelRecognitionImage, image_data)

    def update_detection_display(self, image_data):
        self.display_image(self.labelDetectionImage, image_data)

    def update_status_bar(self, message, timeout=3000):
        self.statusbar.showMessage(message, timeout)

    def handle_recognition_rocgnition(self):
        if self.current_image is not None:
            pub.sendMessage(Topics.SHOW_ROC_RECOGNITION)
            logging.info("Generating ROC curve for recognition")

    def handle_detection_roc(self):
        if self.current_image is not None:
            pub.sendMessage(Topics.SHOW_ROC_DETECTION)
            logging.info("Generating ROC curve for detection")