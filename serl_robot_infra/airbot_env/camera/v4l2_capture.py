import cv2


class V4L2Capture:
    def __init__(self, name, index, dim=(640, 480), fps=30, exposure=None):
        self.name = name
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: /dev/video{index}")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, dim[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dim[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Set exposure if specified
        if exposure is not None:
            # Disable auto exposure first
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

        # Store actual settings
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def read(self):
        ret, frame = self.cap.read()
        if ret and frame is not None:
            return True, frame
        else:
            return False, None

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.close()
