#!/usr/bin/env python3
"""
Web-based lane annotation tool
Much faster and more responsive than OpenCV-based tool
"""

import json
import base64
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO
import cv2
import numpy as np


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Lane Annotation Tool</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1a1a1a;
            color: #fff;
            overflow: hidden;
        }
        .container {
            display: flex;
            height: 100vh;
        }
        .main-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
        }
        .sidebar {
            width: 300px;
            background-color: #2a2a2a;
            padding: 20px;
            overflow-y: auto;
        }
        h1 {
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .canvas-container {
            flex: 1;
            background-color: #000;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            position: relative;
            overflow: hidden;
            cursor: crosshair;
        }
        #canvas {
            position: absolute;
            top: 0;
            left: 0;
        }
        .controls {
            margin-top: 15px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.2s;
        }
        .btn-primary {
            background-color: #4CAF50;
            color: white;
        }
        .btn-primary:hover {
            background-color: #45a049;
        }
        .btn-secondary {
            background-color: #555;
            color: white;
        }
        .btn-secondary:hover {
            background-color: #666;
        }
        .btn-danger {
            background-color: #f44336;
            color: white;
        }
        .btn-danger:hover {
            background-color: #da190b;
        }
        .info-box {
            background-color: #333;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .info-item {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid #444;
        }
        .info-label {
            color: #888;
        }
        .info-value {
            color: #4CAF50;
            font-weight: bold;
        }
        .lane-list {
            margin-top: 15px;
        }
        .lane-item {
            background-color: #333;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: move;
            transition: all 0.2s;
        }
        .lane-item:hover {
            background-color: #3a3a3a;
        }
        .lane-item.dragging {
            opacity: 0.5;
            cursor: grabbing;
        }
        .lane-item.drag-over {
            border-top: 3px solid #4CAF50;
        }
        .lane-info {
            display: flex;
            align-items: center;
        }
        .lane-color {
            width: 20px;
            height: 20px;
            border-radius: 3px;
            display: inline-block;
            margin-right: 10px;
        }
        .lane-actions {
            display: flex;
            gap: 5px;
        }
        .help-section {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #444;
        }
        .help-section h3 {
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .shortcut {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            font-size: 13px;
        }
        .key {
            background-color: #555;
            padding: 2px 8px;
            border-radius: 3px;
            font-family: monospace;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #333;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.3s;
        }
        .drag-handle {
            cursor: grab;
            padding: 5px;
            margin-right: 5px;
            color: #888;
        }
        .drag-handle:active {
            cursor: grabbing;
        }
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="main-area">
            <h1>Lane Annotation Tool</h1>

            <div class="canvas-container" id="canvasContainer">
                <canvas id="canvas"></canvas>
            </div>

            <div class="controls">
                <button class="btn-primary" onclick="finishLane()">Finish Lane (Space)</button>
                <button class="btn-secondary" onclick="undoPoint()">Undo Point (U)</button>
                <button class="btn-secondary" onclick="resetImage()">Reset Image (R)</button>
                <button class="btn-primary" onclick="nextImage()">Next Image (Enter/→)</button>
                <button class="btn-secondary" onclick="prevImage()">Previous (←)</button>
                <button class="btn-danger" onclick="skipImage()">Skip Image (S)</button>
            </div>
        </div>

        <div class="sidebar">
            <div class="info-box">
                <h3>Progress</h3>
                <div class="info-item">
                    <span class="info-label">Image:</span>
                    <span class="info-value" id="imageNum">0/0</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Filename:</span>
                    <span class="info-value" id="imageName" style="font-size: 12px; word-break: break-all;">-</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressBar" style="width: 0%"></div>
                </div>
                <div class="info-item">
                    <span class="info-label">Status:</span>
                    <span class="info-value" id="annotationStatus">New</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Current Lane:</span>
                    <span class="info-value" id="currentLane">Lane 1</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Points:</span>
                    <span class="info-value" id="pointCount">0</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Lanes Saved:</span>
                    <span class="info-value" id="laneCount">0</span>
                </div>
            </div>

            <div class="lane-list" id="laneList">
                <h3>Completed Lanes (Drag to Reorder)</h3>
            </div>

            <div class="help-section">
                <h3>Keyboard Shortcuts</h3>
                <div class="shortcut">
                    <span>Click image</span>
                    <span class="key">Add point</span>
                </div>
                <div class="shortcut">
                    <span>Space</span>
                    <span class="key">Finish lane</span>
                </div>
                <div class="shortcut">
                    <span>Enter / →</span>
                    <span class="key">Next image</span>
                </div>
                <div class="shortcut">
                    <span>←</span>
                    <span class="key">Previous image</span>
                </div>
                <div class="shortcut">
                    <span>U</span>
                    <span class="key">Undo point</span>
                </div>
                <div class="shortcut">
                    <span>R</span>
                    <span class="key">Reset image</span>
                </div>
                <div class="shortcut">
                    <span>S</span>
                    <span class="key">Skip image</span>
                </div>
                <div class="shortcut">
                    <span>Drag lane</span>
                    <span class="key">Reorder</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const container = document.getElementById('canvasContainer');

        let socket;
        let currentImage = null;
        let currentLane = [];
        let completedLanes = [];
        let imageData = null;
        let draggedLaneIdx = null;

        const colors = [
            '#FF0000', // Red
            '#00FF00', // Green
            '#0000FF', // Blue
            '#FFFF00', // Yellow
            '#FF00FF', // Magenta
            '#00FFFF'  // Cyan
        ];

        // Initialize
        window.onload = function() {
            socket = io.connect('http://' + document.domain + ':' + location.port);

            socket.on('connect', function() {
                console.log('Connected');
                socket.emit('request_image');
            });

            socket.on('image_data', function(data) {
                imageData = data;
                loadImage(data);

                // Load existing annotations if available
                if (data.existing_lanes && data.existing_lanes.length > 0) {
                    completedLanes = data.existing_lanes;
                    currentLane = [];
                    console.log('Loaded existing annotation with ' + completedLanes.length + ' lanes');
                } else {
                    completedLanes = [];
                    currentLane = [];
                }

                updateInfo();
                updateLaneList();
            });

            socket.on('annotation_saved', function(data) {
                console.log('Annotation saved');
            });

            // Canvas click handler
            canvas.addEventListener('click', function(e) {
                const rect = canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) * (canvas.width / rect.width);
                const y = (e.clientY - rect.top) * (canvas.height / rect.height);

                addPoint(x, y);
            });

            // Keyboard shortcuts
            document.addEventListener('keydown', function(e) {
                // Space - finish lane
                if (e.key === ' ') {
                    e.preventDefault();
                    finishLane();
                }
                // Enter or Right Arrow - next image
                else if (e.key === 'Enter' || e.key === 'ArrowRight') {
                    e.preventDefault();
                    nextImage();
                }
                // Left Arrow - previous image
                else if (e.key === 'ArrowLeft') {
                    e.preventDefault();
                    prevImage();
                }
                // U - undo
                else if (e.key === 'u' || e.key === 'U') {
                    undoPoint();
                }
                // R - reset
                else if (e.key === 'r' || e.key === 'R') {
                    resetImage();
                }
                // S - skip
                else if (e.key === 's' || e.key === 'S') {
                    skipImage();
                }
            });

            // Resize canvas to fit container
            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);
        };

        function resizeCanvas() {
            const rect = container.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            redraw();
        }

        function loadImage(data) {
            const img = new Image();
            img.onload = function() {
                currentImage = img;
                resizeCanvas();
                redraw();
            };
            img.src = 'data:image/jpeg;base64,' + data.image;
        }

        function addPoint(x, y) {
            // Scale to original image size
            const scaleX = imageData.width / canvas.width;
            const scaleY = imageData.height / canvas.height;

            currentLane.push([Math.round(x * scaleX), Math.round(y * scaleY)]);
            updateInfo();
            redraw();
        }

        function finishLane() {
            if (currentLane.length < 5) {
                alert('Need at least 5 points for a lane');
                return;
            }

            completedLanes.push([...currentLane]);
            currentLane = [];
            updateInfo();
            updateLaneList();
            redraw();
        }

        function undoPoint() {
            if (currentLane.length > 0) {
                currentLane.pop();
            } else if (completedLanes.length > 0) {
                currentLane = completedLanes.pop();
                updateLaneList();
            }
            updateInfo();
            redraw();
        }

        function resetImage() {
            currentLane = [];
            completedLanes = [];
            updateInfo();
            updateLaneList();
            redraw();
        }

        function nextImage() {
            if (currentLane.length > 0) {
                if (!confirm('You have an unfinished lane. Continue anyway?')) {
                    return;
                }
            }

            saveAndNext();
        }

        function prevImage() {
            socket.emit('prev_image');
        }

        function skipImage() {
            socket.emit('skip_image');
        }

        function saveAndNext() {
            const annotation = {
                lanes: completedLanes
            };

            socket.emit('save_annotation', annotation);
            socket.emit('next_image');
        }

        function redraw() {
            if (!currentImage) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw image scaled to canvas
            ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);

            const scaleX = canvas.width / imageData.width;
            const scaleY = canvas.height / imageData.height;

            // Draw completed lanes
            completedLanes.forEach((lane, idx) => {
                ctx.strokeStyle = colors[idx % colors.length];
                ctx.fillStyle = colors[idx % colors.length];
                ctx.lineWidth = 3;

                ctx.beginPath();
                lane.forEach((point, i) => {
                    const x = point[0] * scaleX;
                    const y = point[1] * scaleY;

                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }

                    // Draw points
                    ctx.fillRect(x - 3, y - 3, 6, 6);
                });
                ctx.stroke();
            });

            // Draw current lane
            if (currentLane.length > 0) {
                const color = colors[completedLanes.length % colors.length];
                ctx.strokeStyle = color;
                ctx.fillStyle = color;
                ctx.lineWidth = 3;

                ctx.beginPath();
                currentLane.forEach((point, i) => {
                    const x = point[0] * scaleX;
                    const y = point[1] * scaleY;

                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }

                    // Draw points larger for current lane
                    ctx.fillRect(x - 4, y - 4, 8, 8);
                });
                ctx.stroke();
            }
        }

        function updateInfo() {
            if (!imageData) return;

            document.getElementById('imageNum').textContent =
                imageData.current + '/' + imageData.total;

            document.getElementById('imageName').textContent = imageData.filename || '-';

            const progress = (imageData.current / imageData.total * 100).toFixed(1);
            document.getElementById('progressBar').style.width = progress + '%';

            // Update status
            const statusEl = document.getElementById('annotationStatus');
            if (imageData.existing_lanes && imageData.existing_lanes.length > 0) {
                statusEl.textContent = 'Editing';
                statusEl.style.color = '#FF9800'; // Orange
            } else {
                statusEl.textContent = 'New';
                statusEl.style.color = '#4CAF50'; // Green
            }

            document.getElementById('currentLane').textContent =
                'Lane ' + (completedLanes.length + 1);

            document.getElementById('pointCount').textContent = currentLane.length;
            document.getElementById('laneCount').textContent = completedLanes.length;
        }

        function updateLaneList() {
            const list = document.getElementById('laneList');
            list.innerHTML = '<h3>Completed Lanes (Drag to Reorder)</h3>';

            completedLanes.forEach((lane, idx) => {
                const div = document.createElement('div');
                div.className = 'lane-item';
                div.draggable = true;
                div.dataset.index = idx;

                div.innerHTML = `
                    <div class="lane-info">
                        <span class="drag-handle">☰</span>
                        <span class="lane-color" style="background-color: ${colors[idx % colors.length]}"></span>
                        Lane ${idx + 1} (${lane.length} points)
                    </div>
                    <div class="lane-actions">
                        <button onclick="deleteLane(${idx})" style="padding: 5px 10px; background: #f44336; border: none; border-radius: 3px; color: white; cursor: pointer;">
                            Delete
                        </button>
                    </div>
                `;

                // Drag events
                div.addEventListener('dragstart', handleDragStart);
                div.addEventListener('dragend', handleDragEnd);
                div.addEventListener('dragover', handleDragOver);
                div.addEventListener('drop', handleDrop);
                div.addEventListener('dragleave', handleDragLeave);

                list.appendChild(div);
            });
        }

        function handleDragStart(e) {
            draggedLaneIdx = parseInt(e.target.dataset.index);
            e.target.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
        }

        function handleDragEnd(e) {
            e.target.classList.remove('dragging');
            document.querySelectorAll('.lane-item').forEach(item => {
                item.classList.remove('drag-over');
            });
        }

        function handleDragOver(e) {
            if (e.preventDefault) {
                e.preventDefault();
            }
            e.dataTransfer.dropEffect = 'move';

            const target = e.target.closest('.lane-item');
            if (target && draggedLaneIdx !== null) {
                target.classList.add('drag-over');
            }

            return false;
        }

        function handleDragLeave(e) {
            const target = e.target.closest('.lane-item');
            if (target) {
                target.classList.remove('drag-over');
            }
        }

        function handleDrop(e) {
            if (e.stopPropagation) {
                e.stopPropagation();
            }

            const target = e.target.closest('.lane-item');
            if (!target || draggedLaneIdx === null) return false;

            const dropIdx = parseInt(target.dataset.index);

            if (draggedLaneIdx !== dropIdx) {
                // Reorder the lanes array
                const draggedLane = completedLanes[draggedLaneIdx];
                completedLanes.splice(draggedLaneIdx, 1);
                completedLanes.splice(dropIdx, 0, draggedLane);

                updateInfo();
                updateLaneList();
                redraw();
            }

            return false;
        }

        function deleteLane(idx) {
            if (confirm('Delete lane ' + (idx + 1) + '?')) {
                completedLanes.splice(idx, 1);
                updateInfo();
                updateLaneList();
                redraw();
            }
        }
    </script>
</body>
</html>
"""


class WebAnnotator:
    """Web-based annotation tool"""

    def __init__(self, image_dir='training_data/images',
                 annotation_dir='training_data/annotations',
                 port=5001):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        self.port = port

        # Load images
        self.images = sorted(list(self.image_dir.glob('*.jpg')))
        if not self.images:
            raise ValueError(f"No images found in {self.image_dir}")

        print(f"Found {len(self.images)} images")

        # Find first unannotated image
        self.current_idx = 0
        self.load_progress()

        # Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'lane-annotator'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        self.setup_routes()

    def load_progress(self):
        """Find first unannotated image (but allow editing of annotated ones)"""
        annotated = {f.stem for f in self.annotation_dir.glob('*.json')}

        # Start from first unannotated image
        for idx, img_path in enumerate(self.images):
            if img_path.stem not in annotated:
                self.current_idx = idx
                print(f"Starting from first unannotated image: {idx + 1}")
                return

        # All annotated - start from beginning for review/editing
        if annotated:
            print(f"All {len(annotated)} images already annotated!")
            print("You can review and edit them")
            self.current_idx = 0

    def get_image_data(self):
        """Get current image as base64 with existing annotations if available"""
        if self.current_idx >= len(self.images):
            return None

        img_path = self.images[self.current_idx]
        img = cv2.imread(str(img_path))

        if img is None:
            return None

        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Check for existing annotation
        annotation_path = self.annotation_dir / f"{img_path.stem}.json"
        existing_lanes = []

        if annotation_path.exists():
            try:
                with open(annotation_path, 'r') as f:
                    ann = json.load(f)
                    existing_lanes = ann.get('lanes', [])
                    print(f"Loaded existing annotation: {annotation_path.name} ({len(existing_lanes)} lanes)")
            except Exception as e:
                print(f"Error loading annotation: {e}")

        return {
            'image': img_base64,
            'filename': img_path.name,
            'width': img.shape[1],
            'height': img.shape[0],
            'current': self.current_idx + 1,
            'total': len(self.images),
            'existing_lanes': existing_lanes
        }

    def save_annotation(self, lanes):
        """Save annotation to JSON"""
        img_path = self.images[self.current_idx]
        img = cv2.imread(str(img_path))

        annotation = {
            'image': img_path.name,
            'width': img.shape[1],
            'height': img.shape[0],
            'lanes': lanes
        }

        output_path = self.annotation_dir / f"{img_path.stem}.json"
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)

        print(f"✓ Saved {output_path.name} ({len(lanes)} lanes)")

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)

        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')

        @self.socketio.on('request_image')
        def handle_request():
            data = self.get_image_data()
            if data:
                self.socketio.emit('image_data', data)

        @self.socketio.on('save_annotation')
        def handle_save(data):
            lanes = data.get('lanes', [])
            if lanes:
                self.save_annotation(lanes)
            self.socketio.emit('annotation_saved', {})

        @self.socketio.on('next_image')
        def handle_next():
            self.current_idx += 1
            if self.current_idx >= len(self.images):
                self.socketio.emit('complete', {})
                print("\n✓ All images annotated!")
            else:
                data = self.get_image_data()
                if data:
                    self.socketio.emit('image_data', data)

        @self.socketio.on('prev_image')
        def handle_prev():
            if self.current_idx > 0:
                self.current_idx -= 1
                data = self.get_image_data()
                if data:
                    self.socketio.emit('image_data', data)

        @self.socketio.on('skip_image')
        def handle_skip():
            print(f"Skipped {self.images[self.current_idx].name}")
            self.current_idx += 1
            if self.current_idx >= len(self.images):
                self.socketio.emit('complete', {})
            else:
                data = self.get_image_data()
                if data:
                    self.socketio.emit('image_data', data)

    def run(self):
        """Start web server"""
        import socket
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)

        print("\n" + "="*60)
        print("Web-Based Lane Annotation Tool")
        print("="*60)
        print(f"\nTotal images: {len(self.images)}")
        print(f"Starting from: {self.current_idx + 1}")
        print(f"\nAccess at:")
        print(f"  • Local:   http://localhost:{self.port}")
        print(f"  • Network: http://{ip_address}:{self.port}")
        print(f"\nOpen in your browser and start annotating!")
        print("="*60 + "\n")

        try:
            self.socketio.run(self.app, host='0.0.0.0', port=self.port,
                            debug=False, allow_unsafe_werkzeug=True)
        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            print(f"\n✓ Annotations saved to: {self.annotation_dir}")


if __name__ == '__main__':
    annotator = WebAnnotator(
        image_dir='training_data/images',
        annotation_dir='training_data/annotations',
        port=5001
    )
    annotator.run()