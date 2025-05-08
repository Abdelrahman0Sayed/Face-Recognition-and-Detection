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
        pass