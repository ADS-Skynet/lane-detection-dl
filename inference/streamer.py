# ===================================================================
# streamer.py - Web Streaming Handler
# ===================================================================

import time
from threading import Thread, Lock


class LaneDetectionStreamer:
    """Handles video streaming and lane detection in separate thread"""

    def __init__(self, camera, detector):
        self.camera = camera
        self.detector = detector
        self.frame_lock = Lock()
        self.current_frame = None
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.num_lanes = 0
        self.start_time = None
        self.dropped_frames = 0
        self.detection_enabled = True  # Toggle for detection mode

    def start(self):
        """Start the detection thread"""
        self.running = True
        self.start_time = time.time()
        thread = Thread(target=self._detection_loop, daemon=True)
        thread.start()
        print("✓ Detection thread started")

    def stop(self):
        """Stop the detection thread"""
        self.running = False

    def _detection_loop(self):
        """Main detection loop running in background thread"""
        last_lanes = []
        last_mask = None
        inference_skip = 4  # Process every Nth frame, reuse mask for others

        # Timing stats
        cam_time_sum = 0
        infer_time_sum = 0
        timing_frames = 0

        while self.running:
            try:
                # Time camera read
                t0 = time.time()
                frame = self.camera.read_latest()
                cam_time = time.time() - t0

                if frame is None:
                    time.sleep(0.01)
                    continue

                # Time inference
                t1 = time.time()

                # Check if detection is enabled
                if self.detection_enabled:
                    # Run inference only on selected frames
                    if self.frame_count % inference_skip == 0:
                        lanes, vis_img, mask = self.detector.detect(frame)
                        last_lanes = lanes
                        last_mask = mask
                    else:
                        # Skip heavy inference, reuse last mask for consistent visualization
                        if last_mask is not None:
                            vis_img = self.detector._visualize_mask(frame, last_mask)
                        else:
                            vis_img = frame.copy()
                else:
                    # Raw mode - no detection, just pass through the frame
                    vis_img = frame
                    last_lanes = []

                infer_time = time.time() - t1

                # Accumulate timing stats
                cam_time_sum += cam_time
                infer_time_sum += infer_time
                timing_frames += 1

                # Calculate FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                self.num_lanes = len(last_lanes)

                # Update current frame (thread-safe) - REPLACE, don't queue
                with self.frame_lock:
                    if self.current_frame is not None:
                        self.dropped_frames += 1
                    self.current_frame = vis_img

                # Print status with timing breakdown
                if self.frame_count % 60 == 0:
                    avg_cam = (cam_time_sum / timing_frames) * 1000 if timing_frames > 0 else 0
                    avg_infer = (infer_time_sum / timing_frames) * 1000 if timing_frames > 0 else 0
                    print(f"Frame {self.frame_count}: FPS: {self.fps:.1f} | Cam: {avg_cam:.1f}ms | Infer: {avg_infer:.1f}ms")
                    cam_time_sum = 0
                    infer_time_sum = 0
                    timing_frames = 0
                    self.dropped_frames = 0

            except Exception as e:
                print(f"Error in detection loop: {e}")
                time.sleep(0.1)

    def get_frame(self):
        """Get current frame for streaming (always returns latest, never queues)"""
        with self.frame_lock:
            if self.current_frame is not None:
                frame = self.current_frame.copy()
                # Clear the frame so we know if a new one arrives
                self.current_frame = None
                return frame
            return None

    def set_detection_enabled(self, enabled: bool):
        """Toggle detection on/off"""
        self.detection_enabled = enabled
        print(f"Detection {'enabled' if enabled else 'disabled'}")