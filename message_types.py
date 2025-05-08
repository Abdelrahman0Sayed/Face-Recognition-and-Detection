class Topics:
    """
    Defines all PubSub topics used for communication within the Face Recognition and Detection application.
    
    This class serves as a central registry of all the event topics that components
    can publish to or subscribe to, providing a structured communication framework.
    """
    
    # Image Upload Events
    UPLOAD_IMAGE_RECOGNITION = "upload_image_recognition"
    """
    Published when a new image is uploaded for face recognition.
    
    Args:
        image_path (str): Path to the uploaded image file
        image_data (numpy.ndarray): The loaded image as an OpenCV matrix
    """
    
    UPLOAD_IMAGE_DETECTION = "upload_image_detection"
    """
    Published when a new image is uploaded for face detection.
    
    Args:
        image_path (str): Path to the uploaded image file
        image_data (numpy.ndarray): The loaded image as an OpenCV matrix
    """
    
    # Processing Results
    FACE_RECOGNITION_COMPLETED = "face_recognition_completed"
    """
    Published when face recognition processing is complete.
    
    Args:
        result_image (numpy.ndarray): The processed image with recognized faces marked
        faces (List[RecognitionResult]): List of recognized faces with confidence scores
    """
    
    FACE_DETECTION_COMPLETED = "face_detection_completed"
    """
    Published when face detection processing is complete.
    
    Args:
        result_image (numpy.ndarray): The processed image with detected faces marked
        faces (List[DetectionResult]): List of detected face locations
    """
    
    # ROC Curve Events
    SHOW_ROC_RECOGNITION = "show_roc_recognition"
    """
    Published when the ROC curve for face recognition should be displayed.
    
    Args:
        fpr (numpy.ndarray): False positive rates
        tpr (numpy.ndarray): True positive rates
        auc (float): Area under curve value
    """
    
    SHOW_ROC_DETECTION = "show_roc_detection"
    """
    Published when the ROC curve for face detection should be displayed.
    
    Args:
        fpr (numpy.ndarray): False positive rates
        tpr (numpy.ndarray): True positive rates
        auc (float): Area under curve value
    """
    
    # UI Updates
    UPDATE_RECOGNITION_DISPLAY = "update_recognition_display"
    """
    Published to update the recognition tab's image display.
    
    Args:
        image_data (numpy.ndarray): The image to display
        message (str): Optional status message to show
    """
    
    UPDATE_DETECTION_DISPLAY = "update_detection_display"
    """
    Published to update the detection tab's image display.
    
    Args:
        image_data (numpy.ndarray): The image to display
        message (str): Optional status message to show
    """
    
    UPDATE_STATUS_BAR = "update_status_bar"
    """
    Published to update the application's status bar.
    
    Args:
        message (str): The message to display
        timeout (int, optional): Duration to show the message in milliseconds
    """
    
    # File Operations
    SAVE_RESULTS = "save_results"
    """
    Published when results should be saved to file.
    
    Args:
        data (dict): Results data to save
        file_path (str): Path where to save the results
    """
    
    # Error Events
    PROCESSING_ERROR = "processing_error"
    """
    Published when an error occurs during processing.
    
    Args:
        error_type (str): Type of error that occurred
        error_message (str): Detailed error message
        source (str): Component where the error originated
    """
    
    # Tab Events
    TAB_CHANGED = "tab_changed"
    """
    Published when switching between recognition and detection tabs.
    
    Args:
        tab_name (str): Name of the selected tab ("recognition" or "detection")
    """

    APPLY_DETECTION = "apply_detection"
    """
    Published when a detection operation is requested.
    
    Args:
        image (numpy.ndarray): The image to process for detection
    """

    APPLY_RECOGNITION = "apply_recognition"
    """
    Published when a recognition operation is requested.

    Args:
        image (numpy.ndarray): The image to process for recognition
    """