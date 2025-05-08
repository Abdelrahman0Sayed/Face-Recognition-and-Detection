import sys
from PyQt6.QtWidgets import QApplication
from MainWindowUI import MainWindowUI
import asyncio
from qasync import QEventLoop
from FaceRecognition import FaceRecognition
from FaceDetection import FaceDetection

async def main():
    # Initialize face recognition and detection classes
    face_recognition = FaceRecognition()
    face_detection = FaceDetection()
    
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    # Create and show main window
    window = MainWindowUI()
    window.show()
    
    # Run event loop
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)