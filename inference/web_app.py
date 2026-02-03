# ===================================================================
# web_app.py - Flask-SocketIO Web Application
# ===================================================================

import cv2
import base64
from flask import Flask, render_template_string
from flask_socketio import SocketIO
import time

# HTML template with WebSocket
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Lane Detection - Live Stream</title>
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
            max-width: 1600px;
            margin: 0 auto;
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
            background-color: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            position: relative;
            display: inline-block;
            width: 100%;
        }
        .video-wrapper {
            position: relative;
            display: inline-block;
        }
        #videoStream {
            max-width: 100%;
            height: auto;
            border: 2px solid #4CAF50;
            border-radius: 5px;
            display: block;
        }
        .video-overlay {
            position: absolute;
            top: 10px;
            left: 10px;
            background-color: rgba(0, 0, 0, 0.6);
            color: #4CAF50;
            padding: 8px 12px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            font-weight: bold;
            pointer-events: none;
        }
        .video-overlay span {
            margin-right: 15px;
        }
        .mode-toggle {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: rgba(0, 0, 0, 0.6);
            padding: 8px 12px;
            border-radius: 5px;
        }
        .mode-toggle button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            font-size: 12px;
        }
        .mode-toggle button.raw {
            background-color: #FF9800;
        }
        .mode-toggle button:hover {
            opacity: 0.8;
        }
        .info {
            text-align: center;
            padding: 20px;
            background-color: #2a2a2a;
            border-radius: 10px;
            margin-top: 20px;
        }
        .info p {
            margin: 10px 0;
            font-size: 16px;
        }
        .badge {
            display: inline-block;
            padding: 5px 15px;
            background-color: #4CAF50;
            border-radius: 20px;
            margin: 5px;
        }
        .status {
            display: inline-block;
            padding: 5px 15px;
            background-color: #FF9800;
            border-radius: 20px;
            margin: 5px;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #888;
            font-size: 14px;
        }
        .control-panel {
            background-color: #2a2a2a;
            padding: 30px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .control-panel h2 {
            text-align: center;
            color: #4CAF50;
            margin-bottom: 20px;
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            max-width: 400px;
            margin: 0 auto;
        }
        .control-btn {
            padding: 20px;
            font-size: 18px;
            font-weight: bold;
            border: 2px solid #4CAF50;
            background-color: #3a3a3a;
            color: #fff;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s;
            user-select: none;
        }
        .control-btn:hover {
            background-color: #4CAF50;
            transform: scale(1.05);
        }
        .control-btn:active {
            background-color: #45a049;
            transform: scale(0.95);
        }
        .control-btn.stop {
            background-color: #f44336;
            border-color: #f44336;
        }
        .control-btn.stop:hover {
            background-color: #da190b;
        }
        .speed-control {
            margin-top: 20px;
            text-align: center;
        }
        .speed-slider {
            width: 80%;
            max-width: 400px;
            margin: 10px auto;
        }
        input[type="range"] {
            width: 100%;
            height: 8px;
            border-radius: 5px;
            background: #4CAF50;
            outline: none;
            opacity: 0.7;
            transition: opacity .2s;
        }
        input[type="range"]:hover {
            opacity: 1;
        }
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        let socket;
        let isConnected = false;

        window.onload = function() {
            // Connect to WebSocket
            socket = io.connect('http://' + document.domain + ':' + location.port);

            socket.on('connect', function() {
                console.log('WebSocket connected');
                isConnected = true;
                updateStatus('Connected', '#4CAF50');
            });

            socket.on('disconnect', function() {
                console.log('WebSocket disconnected');
                isConnected = false;
                updateStatus('Disconnected', '#f44336');
            });

            // Receive video frames
            socket.on('video_frame', function(data) {
                const img = document.getElementById('videoStream');
                img.src = 'data:image/jpeg;base64,' + data.frame;

                // Update stats overlay
                if (data.stats) {
                    document.getElementById('statFrames').textContent = 'F:' + data.stats.frames;
                    document.getElementById('statLanes').textContent = 'L:' + data.stats.lanes;
                    document.getElementById('statFps').textContent = 'FPS:' + data.stats.fps;

                    // Sync detection mode button with server state
                    const btn = document.getElementById('modeBtn');
                    if (data.stats.detection !== window.detectionEnabled) {
                        window.detectionEnabled = data.stats.detection;
                        btn.textContent = window.detectionEnabled ? 'Detection ON' : 'RAW MODE';
                        btn.classList.toggle('raw', !window.detectionEnabled);
                    }
                }
            });

            // Update status
            function updateStatus(text, color) {
                const status = document.getElementById('status');
                status.textContent = 'Status: ' + text;
                status.style.backgroundColor = color;
            }

            // Vehicle control functions
            window.sendCommand = function(command) {
                if (!isConnected) {
                    console.log('Not connected to server');
                    return;
                }
                const speed = document.getElementById('speedSlider').value;
                socket.emit('vehicle_control', { command: command, speed: parseInt(speed) });
                console.log('Sent command:', command, 'speed:', speed);
            };

            // Keyboard controls
            document.addEventListener('keydown', function(e) {
                if (!isConnected) return;

                switch(e.key) {
                    case 'ArrowUp':
                    case 'w':
                    case 'W':
                        e.preventDefault();
                        sendCommand('forward');
                        break;
                    case 'ArrowDown':
                    case 's':
                    case 'S':
                        e.preventDefault();
                        sendCommand('backward');
                        break;
                    case 'ArrowLeft':
                    case 'a':
                    case 'A':
                        e.preventDefault();
                        sendCommand('left');
                        break;
                    case 'ArrowRight':
                    case 'd':
                    case 'D':
                        e.preventDefault();
                        sendCommand('right');
                        break;
                    case ' ':
                        e.preventDefault();
                        sendCommand('stop');
                        break;
                }
            });

            // Auto-stop when releasing keys
            document.addEventListener('keyup', function(e) {
                if (!isConnected) return;
                if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'w', 'a', 's', 'd', 'W', 'A', 'S', 'D'].includes(e.key)) {
                    sendCommand('stop');
                }
            });

            // Update speed display
            document.getElementById('speedSlider')?.addEventListener('input', function(e) {
                document.getElementById('speedValue').textContent = e.target.value + '%';
            });

            // Detection mode toggle
            window.detectionEnabled = true;
            window.toggleDetection = function() {
                if (!isConnected) return;
                window.detectionEnabled = !window.detectionEnabled;
                socket.emit('toggle_detection', { enabled: window.detectionEnabled });

                const btn = document.getElementById('modeBtn');
                if (window.detectionEnabled) {
                    btn.textContent = 'Detection ON';
                    btn.classList.remove('raw');
                } else {
                    btn.textContent = 'RAW MODE';
                    btn.classList.add('raw');
                }
            };
        };
    </script>
</head>
<body>
    <div class="container">
        <h1>Inference Live Stream</h1>

        <div class="video-container">
            <div class="video-wrapper">
                <img id="videoStream" src="" alt="Waiting for stream...">
                <div class="video-overlay">
                    <span id="statFrames">F:0</span>
                    <span id="statLanes">L:0</span>
                    <span id="statFps">FPS:0</span>
                </div>
                <div class="mode-toggle">
                    <button id="modeBtn" onclick="toggleDetection()">Detection ON</button>
                </div>
            </div>
        </div>

        <div class="info">
            <p><span class="status" id="status">Status: Connecting...</span></p>
            <p>Real-time lane detection via WebSocket</p>
            <p><strong>Low latency streaming from Jetson Orin</strong></p>
        </div>

        <div class="control-panel">
            <h2>🎮 Vehicle Control</h2>

            <div class="controls">
                <div></div>
                <button class="control-btn" onclick="sendCommand('forward')" title="Forward (↑ or W)">
                    ⬆️ Forward
                </button>
                <div></div>

                <button class="control-btn" onclick="sendCommand('left')" title="Left (← or A)">
                    ⬅️ Left
                </button>
                <button class="control-btn stop" onclick="sendCommand('stop')" title="Stop (Space)">
                    ⏹️ STOP
                </button>
                <button class="control-btn" onclick="sendCommand('right')" title="Right (→ or D)">
                    ➡️ Right
                </button>

                <div></div>
                <button class="control-btn" onclick="sendCommand('backward')" title="Backward (↓ or S)">
                    ⬇️ Backward
                </button>
                <div></div>
            </div>

            <div class="speed-control">
                <h3>Speed: <span id="speedValue">50%</span></h3>
                <div class="speed-slider">
                    <input type="range" id="speedSlider" min="0" max="100" value="50" step="5">
                </div>
                <p style="font-size: 12px; color: #888; margin-top: 10px;">
                    Use arrow keys or WASD to control | Space to stop
                </p>
            </div>
        </div>

        <div class="footer">
            <p>Powered by BiSeNet V2 | Running on GPU | WebSocket Streaming</p>
        </div>
    </div>
</body>
</html>
"""


class WebApp:
    """Flask-SocketIO web application for streaming"""

    def __init__(self, controller=None):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'lane-detection-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        self.streamer = None
        self.controller = controller
        self.is_streaming = False
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes and SocketIO events"""

        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)

        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            if not self.is_streaming:
                self.is_streaming = True
                self.socketio.start_background_task(self._stream_frames)

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client disconnected')
            # Stop vehicle when client disconnects for safety
            if self.controller:
                self.controller.stop()

        @self.socketio.on('vehicle_control')
        def handle_vehicle_control(data):
            """Handle vehicle control commands from web interface"""

            if not self.controller:
                print('⚠ No controller configured')
                return

            command = data.get('command', 'stop')
            speed = data.get('speed', 50)


            if command == 'forward':
                self.controller.forward(speed)
            elif command == 'backward':
                self.controller.backward(speed)
            elif command == 'left':
                self.controller.left(speed)
            elif command == 'right':
                self.controller.right(speed)
            elif command == 'stop':
                self.controller.stop()
            else:
                print(f'⚠ Unknown command: {command}')


        @self.socketio.on('toggle_detection')
        def handle_toggle_detection(data):
            """Toggle detection on/off for raw video mode"""
            if self.streamer:
                enabled = data.get('enabled', True)
                self.streamer.set_detection_enabled(enabled)

    def _stream_frames(self):
        """Background task to stream frames via WebSocket"""
        print("Started WebSocket streaming")
        last_emit_time = 0
        target_fps = 30  # Target streaming FPS
        frame_interval = 1.0 / target_fps

        while self.is_streaming:
            if self.streamer is None:
                time.sleep(0.1)
                continue

            # Rate limiting first - sleep until next frame time
            current_time = time.time()
            time_until_next = frame_interval - (current_time - last_emit_time)
            if time_until_next > 0:
                time.sleep(time_until_next)

            # Get the latest frame
            frame = self.streamer.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            # Encode frame to JPEG (lower quality = faster encoding + smaller payload)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            if not ret:
                continue

            # Convert to base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Get stats from streamer
            stats = {
                'frames': self.streamer.frame_count,
                'lanes': self.streamer.num_lanes,
                'fps': round(self.streamer.fps),
                'detection': self.streamer.detection_enabled
            }

            # Emit to all connected clients
            self.socketio.emit('video_frame', {'frame': frame_base64, 'stats': stats})
            last_emit_time = time.time()

    def set_streamer(self, streamer):
        """Set the video streamer"""
        self.streamer = streamer

    def set_controller(self, controller):
        """Set the vehicle controller"""
        self.controller = controller

    def run(self, host='0.0.0.0', port=5000):
        """Run the Flask-SocketIO server"""
        self.socketio.run(self.app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)