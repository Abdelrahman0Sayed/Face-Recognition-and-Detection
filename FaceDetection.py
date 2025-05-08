from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time
import cv2
import numpy as np
import logging

class FaceDetection:
    def __init__(self):
        self.setup_subscriptions()

    def setup_subscriptions(self):
        pub.subscribe(self.on_apply_detection, Topics.APPLY_DETECTION)

    def on_apply_detection(self, image):
        logging.info(f"Applying Face Detection on the image")
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_in_executor(executor, self.apply_face_detection, image)

    def apply_face_detection(self, image):
        print("Applying face detection...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the image in a window
        cv2.imshow('Face Detection', image)
        cv2.waitKey(1)  # Update window with small delay
        
        pub.sendMessage(Topics.FACE_DETECTION_COMPLETED, image_data=image)