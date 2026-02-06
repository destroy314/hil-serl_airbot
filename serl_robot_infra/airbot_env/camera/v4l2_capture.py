import cv2
import numpy as np
import time

class V4L2Capture:
    def __init__(self, name, index, fake=False, dim=(640, 480), fps=30, exposure=None):
        self.name = name
        if fake:
            self.fake = True
            self.cap = None
            self.fake_frame = np.zeros((dim[1], dim[0], 3), dtype=np.uint8)
            self.fake_frame = self.tile_text(self.fake_frame, index)
            return
        self.fake = False
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
    
    def tile_text(self, frame, index):
        h, w = frame.shape[:2]

        text = f"FAKECAM{index} "

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 1
        color = (255, 255, 255)

        size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_w = size[0]
        text_h = size[1]

        step_x = text_w + 10
        step_y = text_h + 10

        y = 0
        while y < h + text_h:
            x = 0
            while x < w + text_w:
                cv2.putText(
                    frame,
                    text,
                    (x, y + text_h),
                    font,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
                x += step_x
            y += step_y
        return frame

    def read(self):
        if self.fake:
            time.sleep(0.03)
            return True, self.fake_frame
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
