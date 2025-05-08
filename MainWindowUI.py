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
        pub.subscribe(self.display_detection_results, Topics.FACE_DETECTION_COMPLETED)
        pub.subscribe(self.display_recognition_results, Topics.FACE_RECOGNITION_COMPLETED)


    def on_tab_changed(self, index):
        tab_name = self.tabWidget.tabText(index).lower()
        self.current_tab = tab_name
        pub.sendMessage(Topics.TAB_CHANGED, tab=tab_name)
        logging.info(f"Tab changed to {tab_name}")


    def handle_recognition_upload(self):
        try:
            file_path = self.get_image_file_path()
            if file_path:
                logging.info(f"Attempting to upload image from: {file_path}")
                image = self.load_image(file_path)
                if image is not None:
                    self.current_image = image
                    self.display_recognition_image(self.current_image)
                    pub.sendMessage(Topics.UPLOAD_IMAGE_RECOGNITION, 
                                  image_path=file_path,
                                  image_data=image)
                    pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                                  message="Image loaded for recognition",
                                  timeout=3000)
                    
                else:
                    pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                                  message="Failed to load image",
                                  timeout=3000)
        except Exception as e:
            logging.error(f"Error in handle_recognition_upload: {str(e)}")
            pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                          message="Error during image upload",
                          timeout=3000)


    def handle_detection_upload(self):
        try:
            file_path = self.get_image_file_path()
            if file_path:
                logging.info(f"Attempting to upload image from: {file_path}")
                image = self.load_image(file_path)
                if image is not None:
                    self.current_image = image
                    self.display_detection_image(self.current_image)
                    pub.sendMessage(Topics.UPLOAD_IMAGE_DETECTION,
                                  image_path=file_path,
                                  image_data=image)
                    pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                                  message="Image loaded for detection",
                                  timeout=3000)
                    pub.sendMessage(Topics.APPLY_DETECTION, image=image)
                    logging.info("Face detection applied")
                else:
                    pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                                  message="Failed to load image",
                                  timeout=3000)
        except Exception as e:
            logging.error(f"Error in handle_detection_upload: {str(e)}")
            pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                          message="Error during image upload",
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
        """
        Display an image on a QLabel with fixed size of 400x400
        
        Args:
            label (QLabel): The label widget to display the image on
            image (numpy.ndarray): The image to display
        """
        if image is None:
            label.clear()
            label.setText("No image to display")
            return
            
        try:
            # Convert image format
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
            # Set fixed size for label
            FIXED_SIZE = 600
            label.setFixedSize(FIXED_SIZE, FIXED_SIZE)
            
            # Create QImage
            height, width = rgb_image.shape[:2]
            q_img = QImage(rgb_image.data, width, height, width * 3, QImage.Format.Format_RGB888)
            
            # Create pixmap and scale it to fixed size
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(FIXED_SIZE, FIXED_SIZE, 
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            
            # Calculate padding to center the image
            x_padding = (FIXED_SIZE - pixmap.width()) // 2
            y_padding = (FIXED_SIZE - pixmap.height()) // 2
            
            # Create a new label with background
            label.setStyleSheet("QLabel { background-color: #252536; border: 1px solid #3d3d5c; }")
            
            # Set pixmap to label with padding
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            logging.info(f"Image displayed in fixed size {FIXED_SIZE}x{FIXED_SIZE}")
                        
        except Exception as e:
            logging.error(f"Error in display_image: {str(e)}")
            label.clear()
            label.setText("Error displaying image")

    def update_recognition_display(self, image_data):
        self.display_recognition_image(image_data)


    def update_detection_display(self, image_data):
        self.display_detection_image(image_data)

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

    
    
    def display_recognition_image(self, image_data=None):

        if image_data is None:
            image_data = self.current_image
            
        if image_data is None:
            self.imageDisplayRecognition.clear()
            self.imageDisplayRecognition.setText("No image loaded")
            return
            
        try:
            self.display_image(self.imageDisplayRecognition, image_data)
            logging.info("Recognition image displayed successfully")
        except Exception as e:
            logging.error(f"Error displaying recognition image: {str(e)}")
            self.imageDisplayRecognition.clear()
            self.imageDisplayRecognition.setText("Error displaying image")



    def display_detection_image(self, image_data=None):
        if image_data is None:
            image_data = self.current_image
            
        if image_data is None:
            self.imageDisplayDetection.clear()
            self.imageDisplayDetection.setText("No image loaded")
            return
            
        try:
            self.display_image(self.imageDisplayDetection, image_data)
            logging.info("Detection image displayed successfully")
        except Exception as e:
            logging.error(f"Error displaying detection image: {str(e)}")
            self.imageDisplayDetection.clear()
            self.imageDisplayDetection.setText("Error displaying image")


    def display_detection_results(self, image_data):
        try:
            if image_data is not None:
                self.display_image(self.imageDisplayDetection, image_data)
                logging.info("Detection results displayed successfully")
                pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                              message="Detection results updated",
                              timeout=3000)
            else:
                logging.warning("No detection results to display")
                pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                              message="No detection results available",
                              timeout=3000)
        except Exception as e:
            logging.error(f"Error displaying detection results: {str(e)}")
            pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                          message="Error displaying detection results",
                          timeout=3000)
    

    def display_recognition_results(self, image_data):
        try:
            if image_data is not None:
                self.display_image(self.labelRecognitionResults, image_data)
                logging.info("Recognition results displayed successfully")
                pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                                message="Recognition results updated",
                                timeout=3000)
            else:
                logging.warning("No recognition results to display")
                pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                                message="No recognition results available",
                                timeout=3000)
        except Exception as e:
            logging.error(f"Error displaying recognition results: {str(e)}")
            pub.sendMessage(Topics.UPDATE_STATUS_BAR,
                            message="Error displaying recognition results",
                            timeout=3000)
    