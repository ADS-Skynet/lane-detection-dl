#!/usr/bin/env python3
"""
WebSocket-based video streaming module
Handles efficient frame streaming with frame skipping
"""

import cv2
import base64
import time
from threading import Thread, Lock
from flask import Flask, Response, render_template_string
from flask_socketio import SocketIO


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Jetracer Video Stream</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #fff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
            background-color: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        #videoStream {
            max-width: 100%;
            height: auto;
            border: 2px solid #4CAF50;
            border-radius: 5px;
        }
        .info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
            padding: 20px;
            background-color: #2a2a2a;
            border-radius: 10px;
        }
        .stat {
            background-color: #333;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-label {
            color: #888;
            font-size: 14px;
            margin-bottom: 5px;
        }
        .stat-value {
            color: #4CAF50;
            font-size: 24px;
            font-weight: bold;
        }
        .key-hints {
            margin-top: 20px;
            padding: 20px;
            background-color: #2a2a2a;
            border-radius: 10px;
        }
        .key-hints h3 {
            margin-top: 0;
            color: #4CAF50;
        }
        .key-row {
            display: flex;
            gap: 20px;
            margin: 10px 0;
            flex-wrap: wrap;
        }
        .key {
            background-color: #555;
            padding: 8px 15px;
            border-radius: 5px;
            font-family: monospace;
            min-width: 150px;
        }
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            margin: 5px;
        }
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        let socket;

        window.onload = function() {
            socket = io.connect('http://' + document.domain + ':' + location.port);

            socket.on('connect', function() {
                console.log('WebSocket connected');
                updateStatus('Connected', '#4CAF50');
            });

            socket.on('disconnect', function() {
                console.log('WebSocket disconnected');
                updateStatus('Disconnected', '#f44336');
            });

            // Receive video frames via WebSocket
            socket.on('video_frame', function(data) {
                document.getElementById('videoStream').src = 'data:image/jpeg;base64,' + data.frame;
            });

            // Receive stats
            socket.on('stats', function(data) {
                updateStats(data);
            });

            // Keyboard controls
            document.addEventListener('keydown', function(e) {
                if(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' ', 'm', 'q'].includes(e.key)) {
                    e.preventDefault();
                    socket.emit('key_press', {key: e.key});
                }
            });
        };

        function updateStats(stats) {
            if (stats.collected !== undefined) {
                document.getElementById('collected').textContent = stats.collected + '/' + stats.target;
                const progress = (stats.collected / stats.target * 100).toFixed(1);
                document.getElementById('progress').textContent = progress + '%';
            }
            if (stats.throttle !== undefined) {
                document.getElementById('throttle').textContent = stats.throttle.toFixed(2);
            }
            if (stats.steering !== undefined) {
                document.getElementById('steering').textContent = stats.steering.toFixed(2);
            }
            if (stats.fps !== undefined) {
                document.getElementById('fps').textContent = stats.fps.toFixed(1);
            }
        }

        function updateStatus(text, color) {
            const status = document.getElementById('status');
            status.textContent = 'Status: ' + text;
            status.style.backgroundColor = color;
        }
    </script>
</head>
<body>
    <div class="container">
        <h1 id="title">🚗 Jetracer Stream</h1>

        <div class="video-container">
            <img id="videoStream" src="" alt="Waiting for stream...">
        </div>

        <div style="text-align: center;">
            <span class="status" id="status" style="background-color: #FF9800;">Status: Connecting...</span>
        </div>

        <div class="info">
            <div class="stat">
                <div class="stat-label">Images Collected</div>
                <div class="stat-value" id="collected">0/0</div>
            </div>
            <div class="stat">
                <div class="stat-label">Progress</div>
                <div class="stat-value" id="progress">0%</div>
            </div>
            <div class="stat">
                <div class="stat-label">Throttle</div>
                <div class="stat-value" id="throttle">0.00</div>
            </div>
            <div class="stat">
                <div class="stat-label">Steering</div>
                <div class="stat-value" id="steering">0.00</div>
            </div>
            <div class="stat">
                <div class="stat-label">Stream FPS</div>
                <div class="stat-value" id="fps">0.0</div>
            </div>
        </div>

        <div class="key-hints">
            <h3>Keyboard Controls</h3>
            <div class="key-row">
                <div class="key">↑ Up: Forward</div>
                <div class="key">↓ Down: Backward</div>
            </div>
            <div class="key-row">
                <div class="key">← Left: Turn Left</div>
                <div class="key">→ Right: Turn Right</div>
            </div>
            <div class="key-row">
                <div class="key">Space: Stop</div>
                <div class="key">M: Manual Capture</div>
                <div class="key">Q: Quit</div>
            </div>
        </div>
    </div>
</body>
</html>
"""


class WebStreamer:
    """WebSocket-based video streamer with frame skipping"""

    def __init__(self, title="Video Stream", port=5000):
        self.title = title
        self.port = port
        self.frame_lock = Lock()
        self.current_frame = None
        self.running = False
        self.stats = {}
        self.key_callback = None

        # Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'jetracer-stream'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')

        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes and SocketIO events"""

        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)

        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            if not self.running:
                self.running = True
                self.socketio.start_background_task(self._stream_loop)

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client disconnected')

        @self.socketio.on('key_press')
        def handle_key(data):
            if self.key_callback:
                self.key_callback(data.get('key'))

    def update_frame(self, frame):
        """Update the current frame (called by external source)"""
        with self.frame_lock:
            self.current_frame = frame.copy() if frame is not None else None

    def update_stats(self, stats_dict):
        """Update stats to display (called by external source)"""
        self.stats = stats_dict

    def set_key_callback(self, callback):
        """Set callback function for keyboard events"""
        self.key_callback = callback

    def _stream_loop(self):
        """Background task to stream frames via WebSocket with frame skipping"""
        print("Started WebSocket streaming")

        target_fps = 20  # Target streaming FPS
        frame_interval = 1.0 / target_fps
        last_emit_time = time.time()
        frame_count = 0
        fps_start_time = time.time()

        while self.running:
            current_time = time.time()

            # Rate limiting - only send frames at target FPS
            time_since_last = current_time - last_emit_time
            if time_since_last < frame_interval:
                time.sleep(0.01)
                continue

            # Get latest frame (skip old frames)
            frame = None
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = self.current_frame
                    self.current_frame = None  # Clear to skip old frames

            if frame is None:
                time.sleep(0.01)
                continue

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ret:
                continue

            # Convert to base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Emit via WebSocket
            self.socketio.emit('video_frame', {'frame': frame_base64})

            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start_time
                fps = 30 / elapsed if elapsed > 0 else 0
                self.stats['fps'] = fps
                fps_start_time = time.time()

            # Send stats
            if self.stats:
                self.socketio.emit('stats', self.stats)

            last_emit_time = current_time

    def start(self, blocking=True):
        """Start the web server"""
        import socket
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)

        print("\n" + "="*60)
        print("WebSocket Video Streamer")
        print("="*60)
        print(f"\nAccess stream at:")
        print(f"  • Local:   http://localhost:{self.port}")
        print(f"  • Network: http://{ip_address}:{self.port}")
        print(f"\nOpen in browser and use keyboard to control")
        print("="*60 + "\n")

        if blocking:
            try:
                self.socketio.run(self.app, host='0.0.0.0', port=self.port,
                                debug=False, allow_unsafe_werkzeug=True)
            except KeyboardInterrupt:
                print("\nStopping...")
            finally:
                self.stop()
        else:
            # Non-blocking - run in thread
            thread = Thread(target=lambda: self.socketio.run(
                self.app, host='0.0.0.0', port=self.port,
                debug=False, allow_unsafe_werkzeug=True
            ), daemon=True)
            thread.start()

    def stop(self):
        """Stop streaming"""
        self.running = False


if __name__ == '__main__':
    # Test the streamer
    from camera import Camera

    camera = Camera(device_path='/dev/video4', rotation=0)

    # Warmup
    for i in range(10):
        camera.read_image()

    streamer = WebStreamer(title="Test Stream", port=5000)

    # Background thread to feed frames
    def update_frames():
        while True:
            frame = camera.read_image()
            if frame is not None:
                streamer.update_frame(frame)
            time.sleep(0.03)

    Thread(target=update_frames, daemon=True).start()

    # Start server (blocking)
    streamer.start()