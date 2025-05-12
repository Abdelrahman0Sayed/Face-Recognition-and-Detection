from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time
import cv2
import numpy as np
import logging
from FaceDetection import FaceDetection
from model import predict_multiple_identities, load_model
class FaceRecognition:
    def __init__(self):
        self.setup_subscriptions()
        self.model = load_model()

    def setup_subscriptions(self):
        pub.subscribe(self.on_apply_recognition, Topics.APPLY_RECOGNITION)

    def on_apply_detection(self, image):
        logging.info(f"Applying Face Recognition on the image")
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_in_executor(executor, self.on_apply_recognition, image)

    def on_apply_recognition(self, image):
        results , image_with_faces=  predict_multiple_identities(image, self.model)
        pub.sendMessage(Topics.FACE_RECOGNITION_COMPLETED, image_data=image_with_faces)

