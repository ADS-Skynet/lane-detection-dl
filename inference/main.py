# ===================================================================
# main.py - Main entry point for web streaming
# ===================================================================

import argparse
import torch
import socket
from camera import Camera
from detector import BiSeNetDetector
from streamer import LaneDetectionStreamer
from web_app import WebApp
from vehicle_controller import create_controller


def parse_args():
    parser = argparse.ArgumentParser(description='BiSeNet Lane Detection Web Streamer')
    parser.add_argument('--model', '-m', type=str, default='bise.pth',
                        help='Path to model checkpoint (default: bise.pth)')
    parser.add_argument('--n_classes', '-n', type=int, default=2,
                        help='Number of classes: 2 for binary, 5 for multi-class (default: 5)')
    parser.add_argument('--device', '-d', type=str, default='cuda:0',
                        help='Device to run on (default: cuda:0)')
    parser.add_argument('--camera', '-c', type=str, default='/dev/video4',
                        help='Camera device path (default: /dev/video4)')
    parser.add_argument('--cam_width', '-cw', type=int, default=640,
                        help='Camera capture width, height auto-calculated (default: 640)')
    parser.add_argument('--model_width', '-mw', type=int, default=640,
                        help='Model input width, height auto-calculated (default: 640)')
    parser.add_argument('--port', '-p', type=int, default=5000,
                        help='Web server port (default: 5000)')
    parser.add_argument('--no-controller', action='store_true',
                        help='Disable vehicle controller')
    parser.add_argument('--throttle', type=float, default=0.7,
                        help='Max throttle for JetRacer (0.0-1.0, default: 0.7)')
    parser.add_argument('--steering', type=float, default=1.0,
                        help='Steering gain for JetRacer (default: 1.0)')
    return parser.parse_args()


def main():
    """Main function - starts web server and detection"""
    args = parse_args()

    print("=" * 60)
    print("BiSeNet Lane Detection Web Streamer")
    print("=" * 60)
    print(f"Model: {args.model} ({args.n_classes} classes)")

    # Initialize camera (height auto-calculated from native aspect ratio)
    print("\n[1/4] Initializing camera...")
    camera = Camera(device_path=args.camera, rotation=0, width=args.cam_width)

    # Calculate model input size from model_width using camera's aspect ratio
    cam_ratio = camera.actual_height / camera.actual_width
    model_width = args.model_width
    model_height = int(model_width * cam_ratio)
    # Ensure height is divisible by 32 for model compatibility
    model_height = (model_height // 32) * 32

    # Initialize detector
    print("\n[2/4] Loading BiSeNet lane detection model...")
    detector = BiSeNetDetector(
        model_path=args.model,
        n_classes=args.n_classes,
        image_size=(model_height, model_width),  # (height, width)
        device=args.device
    )

    # Create vehicle controller
    print("\n[3/4] Initializing vehicle controller...")
    if args.no_controller:
        controller = None
        print("✓ Vehicle controller disabled")
    else:
        controller = create_controller(
            throttle_gain=args.throttle,
            steering_gain=args.steering
        )

    # Create streamer
    print("\n[4/4] Starting video streamer...")
    streamer = LaneDetectionStreamer(camera, detector)
    streamer.start()
    print(f"  Camera: {camera.actual_width}x{camera.actual_height}")
    print(f"  Model input: {model_width}x{model_height}")

    # Create web app with controller
    web_app = WebApp()
    web_app.set_streamer(streamer)
    web_app.set_controller(controller)

    # Get IP address
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    print("\n" + "=" * 60)
    print("✓ Web server ready!")
    print("=" * 60)
    print(f"\nAccess the stream at:")
    print(f"  • Local:   http://localhost:{args.port}")
    print(f"  • Network: http://{ip_address}:{args.port}")
    print(f"\nOpen this URL in your browser to view the lane detection")
    print("Press Ctrl+C to stop\n")

    try:
        # Run Flask server
        web_app.run(host='0.0.0.0', port=args.port)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        # Stop vehicle first for safety
        if controller:
            print("\nStopping vehicle...")
            controller.stop()
            controller.cleanup()

        # Stop streamer
        streamer.stop()

        # Final stats
        if torch.cuda.is_available():
            print(f"\nGPU Memory Summary:")
            print(f"  Max allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
            print(f"  Max reserved: {torch.cuda.max_memory_reserved(0) / 1e9:.2f} GB")

        print(f"Processed {streamer.frame_count} frames")
        print(f"Average FPS: {streamer.fps:.1f}")
        print("Done!")


if __name__ == '__main__':
    main()