"""
Vehicle Controller Module
JetRacer controller for Waveshare JetRacer
"""

import time

DEBUG = False  # Set to False to reduce logging


def debug_print(msg):
    if DEBUG:
        print(f"[JetRacer DEBUG] {msg}")


class JetRacerController:
    """
    Controller for Waveshare JetRacer using NvidiaRacecar
    Based on working implementation from vehicle module
    """

    def __init__(self, throttle_gain: float = 0.3, steering_gain: float = 1.0):
        """
        Initialize JetRacer controller

        Args:
            throttle_gain: Max throttle value (default 0.3 for safety)
            steering_gain: Multiplier for steering (default 1.0)
        """
        self.throttle_gain = throttle_gain
        self.steering_gain = steering_gain
        self.car = None

        debug_print(f"Initializing with throttle_gain={throttle_gain}, steering_gain={steering_gain}")

        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            debug_print("Importing NvidiaRacecar...")
            print("Initializing JetRacer (NvidiaRacecar)...")
            self.car = NvidiaRacecar()
            debug_print(f"NvidiaRacecar object created: {self.car}")
            debug_print(f"Setting initial steering=0.0, throttle=0.0")
            self.car.steering = 0.0
            self.car.throttle = 0.0
            debug_print(f"Initial values set. steering={self.car.steering}, throttle={self.car.throttle}")
            print("✓ JetRacer initialized")
        except Exception as e:
            print(f"⚠ JetRacer initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.car = None

    def forward(self, speed=50):
        """Move forward at specified speed (0-100)"""
        debug_print(f"forward() called with speed={speed}")

        if not self.car:
            print(f"[NO CAR] Forward {speed}%")
            return

        throttle = (speed / 100.0) * self.throttle_gain
        debug_print(f"Calculated throttle={throttle}, will set car.throttle={-throttle}")

        try:
            self.car.throttle = -throttle  # Negative for forward
            self.car.steering = 0.0
            debug_print(f"SET: car.throttle={self.car.throttle}, car.steering={self.car.steering}")
            print(f"Forward {speed}% (throttle={-throttle:.2f})")
        except Exception as e:
            print(f"⚠ Error setting throttle: {e}")
            import traceback
            traceback.print_exc()

    def backward(self, speed=50):
        """Move backward at specified speed (0-100)"""
        debug_print(f"backward() called with speed={speed}")

        if not self.car:
            print(f"[NO CAR] Backward {speed}%")
            return

        throttle = (speed / 100.0) * self.throttle_gain
        debug_print(f"Calculated throttle={throttle}, will set car.throttle={throttle}")

        try:
            self.car.throttle = throttle  # Positive for backward
            self.car.steering = 0.0
            debug_print(f"SET: car.throttle={self.car.throttle}, car.steering={self.car.steering}")
            print(f"Backward {speed}% (throttle={throttle:.2f})")
        except Exception as e:
            print(f"⚠ Error setting throttle: {e}")
            import traceback
            traceback.print_exc()

    def left(self, speed=50):
        """Turn left while moving forward"""
        debug_print(f"left() called with speed={speed}")

        if not self.car:
            print(f"[NO CAR] Left {speed}%")
            return

        throttle = (speed / 100.0) * self.throttle_gain
        debug_print(f"Calculated throttle={throttle}, steering={-self.steering_gain}")

        try:
            self.car.throttle = -throttle
            self.car.steering = self.steering_gain  # Negative for left
            debug_print(f"SET: car.throttle={self.car.throttle}, car.steering={self.car.steering}")
            print(f"Left {speed}% (steering={-self.steering_gain:.2f})")
        except Exception as e:
            print(f"⚠ Error setting left: {e}")
            import traceback
            traceback.print_exc()

    def right(self, speed=50):
        """Turn right while moving forward"""
        debug_print(f"right() called with speed={speed}")

        if not self.car:
            print(f"[NO CAR] Right {speed}%")
            return

        throttle = (speed / 100.0) * self.throttle_gain
        debug_print(f"Calculated throttle={throttle}, steering={self.steering_gain}")

        try:
            self.car.throttle = -throttle
            self.car.steering = -self.steering_gain  # Positive for right
            debug_print(f"SET: car.throttle={self.car.throttle}, car.steering={self.car.steering}")
            print(f"Right {speed}% (steering={self.steering_gain:.2f})")
        except Exception as e:
            print(f"⚠ Error setting right: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """Stop the vehicle"""
        debug_print("stop() called")

        if not self.car:
            debug_print("stop() - no car object")
            return

        try:
            self.car.throttle = 0.0
            self.car.steering = 0.0
            debug_print(f"SET: car.throttle={self.car.throttle}, car.steering={self.car.steering}")
            print("Stop")
        except Exception as e:
            print(f"⚠ Error stopping: {e}")
            import traceback
            traceback.print_exc()

    def cleanup(self):
        """Cleanup - stop vehicle"""
        debug_print("cleanup() called")

        if self.car:
            try:
                self.car.throttle = 0.0
                self.car.steering = 0.0
                debug_print(f"SET: car.throttle={self.car.throttle}, car.steering={self.car.steering}")
            except Exception as e:
                print(f"⚠ Error in cleanup: {e}")
                import traceback
                traceback.print_exc()
        else:
            debug_print("cleanup() - no car object")
        print("✓ JetRacer cleanup")


def create_controller(controller_type='auto', **kwargs):
    """Create JetRacer controller"""
    if controller_type == 'none':
        return None
    return JetRacerController(**kwargs)
