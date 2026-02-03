# camera.py - Fixed version with better error handling
import os
import cv2


class Camera:
    """
    Simple camera wrapper using OpenCV VideoCapture.
    Supports Intel RealSense and other USB cameras.
    """

    def __init__(self, device_path: str = '/dev/video4', rotation: int = 0,
                 width: int = 640):
        self.device_path = device_path

        # Open camera with V4L2 backend for better control
        self.video_capture = cv2.VideoCapture(device_path, cv2.CAP_V4L2)

        if not self.video_capture.isOpened():
            # Fallback to default backend
            self.video_capture = cv2.VideoCapture(device_path)

        if not self.video_capture.isOpened():
            raise RuntimeError(f"Failed to open camera at {device_path}")

        # Try YUYV (raw) first for faster capture, fall back to MJPG for higher resolutions
        fourcc_yuyv = cv2.VideoWriter_fourcc(*'YUYV')
        fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_capture.set(cv2.CAP_PROP_FOURCC, fourcc_yuyv)

        # Get native resolution to calculate aspect ratio
        native_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = native_height / native_width if native_width > 0 else 9/16

        # Calculate height from width, keeping aspect ratio
        height = int(width * aspect_ratio)
        self.width = width
        self.height = height

        # Set resolution
        print(f"Requesting resolution: {width}x{height} (native: {native_width}x{native_height})")
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Request higher FPS (60fps if supported)
        self.video_capture.set(cv2.CAP_PROP_FPS, 60)

        # Set buffer size to 1 to minimize latency
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Get actual resolution and FPS (camera may not support requested)
        self.actual_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        print(f"Actual resolution: {self.actual_width}x{self.actual_height} @ {actual_fps:.0f}fps")

        # Try to set rotation via v4l2-ctl
        if rotation != 0:
            try:
                os.system(f'v4l2-ctl -d {device_path} --set-ctrl=rotate={rotation} 2>/dev/null')
            except Exception:
                pass

        # Discard first few frames (camera initialization)
        print(f"Warming up camera at {device_path}...")
        for _ in range(10):
            self.video_capture.read()
        print("Camera ready!")

    def __del__(self):
        try:
            self.video_capture.release()
        except Exception:
            pass

    def read_image(self):
        ret, frame = self.video_capture.read()
        if not ret:
            return None
        return frame

    def read_latest(self):
        """Read the latest frame - with BUFFERSIZE=1, this is already the latest"""
        ret, frame = self.video_capture.read()
        if not ret:
            return None
        return frame

    def flush_buffer(self):
        """Flush camera buffer by grabbing frames without decoding"""
        for _ in range(3):
            self.video_capture.grab()


class MonochromeCamera(Camera):
    """
    Camera that requests monochrome color effects via v4l2.
    """

    def __init__(self, device_path: str = '/dev/video3', rotation: int = 0, width: int = 640):
        super().__init__(device_path, rotation, width)
        try:
            os.system(f'v4l2-ctl -d {device_path} --set-ctrl=color_effects=1 2>/dev/null')
        except Exception:
            pass