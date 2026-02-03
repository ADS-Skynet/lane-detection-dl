import cv2
import torch
import numpy as np
from pathlib import Path
import sys


class BiSeNetDetector:
    """Lane detector using BiSeNet V2 for semantic segmentation"""

    def __init__(self, model_path: str, n_classes: int = 2,
                 image_size: tuple = (512, 1024), device: str = 'cuda:0'):
        """
        Initialize BiSeNet detector

        Args:
            model_path: Path to trained BiSeNet checkpoint
            n_classes: Number of classes (default: 2 for binary lane/background)
            image_size: (height, width) for model input
            device: Device to run on
        """
        # Add lib to path for BiSeNet import
        lib_path = Path(__file__).parent.parent / 'lib'
        if str(lib_path) not in sys.path:
            sys.path.insert(0, str(lib_path))

        from models.bisenetv2 import BiSeNetV2

        if 'cuda' in device and torch.cuda.is_available():
            self.device = torch.device(device)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        self.n_classes = n_classes
        self.image_size = image_size  # (height, width)

        # Load model
        print(f"Loading BiSeNet model from: {model_path}")
        self.model = BiSeNetV2(n_classes=n_classes, aux_mode='eval')

        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        print(f"BiSeNet model loaded successfully on {self.device}")
        print(f"Input size: {image_size[0]}x{image_size[1]}, Classes: {n_classes}")

        # Define colors for visualization (BGR format)
        # 5 classes: background + 4 lane types
        self.colors = np.array([
            [0, 0, 0],       # Class 0: Background - black
            [255, 0, 0],     # Class 1: Lane type 1 - blue
            [0, 255, 0],     # Class 2: Lane type 2 - green
            [0, 0, 255],     # Class 3: Lane type 3 - red
            [0, 255, 255],   # Class 4: Lane type 4 - yellow
        ], dtype=np.uint8)

        # Class names for debugging
        self.class_names = ['background', 'lane_1', 'lane_2', 'lane_3', 'lane_4']

        self.frame_count = 0
        self._logged_resize_info = False

    def detect(self, img):
        """
        Detect lanes in image

        Args:
            img: Input image (BGR format)

        Returns:
            lanes: List of lane point arrays (for compatibility)
            vis_img: Visualization image with overlay
            mask: Segmentation mask
        """
        self.frame_count += 1
        orig_h, orig_w = img.shape[:2]

        # Check if input matches model size (skip resize if same)
        needs_resize = (orig_h != self.image_size[0]) or (orig_w != self.image_size[1])

        # Log resize info once on first frame
        if not self._logged_resize_info:
            if needs_resize:
                print(f"  Input {orig_w}x{orig_h} -> Model {self.image_size[1]}x{self.image_size[0]} (resizing enabled)")
            else:
                print(f"  Input {orig_w}x{orig_h} matches model size (resizing disabled)")
            self._logged_resize_info = True

        # Preprocess
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if needs_resize:
            img_input = cv2.resize(img_rgb, (self.image_size[1], self.image_size[0]))
        else:
            img_input = img_rgb

        # Normalize (ImageNet normalization)
        img_norm = img_input.astype(np.float32) / 255.0
        img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            logits = outputs[0]

        # Postprocess
        pred = logits.argmax(dim=1)[0].cpu().numpy()

        # Resize mask to original size (skip if same)
        if needs_resize:
            mask = cv2.resize(
                pred.astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask = pred.astype(np.uint8)

        # Create visualization
        vis_img = self._visualize_mask(img, mask)

        # Extract lane points (for compatibility with streamer)
        lanes = self._extract_lane_points(mask)

        # Debug info every 30 frames
        if self.frame_count % 30 == 0:
            total_pixels = mask.size
            print(f"--- BiSeNet Frame {self.frame_count} ---")
            # Show per-class statistics for multi-class model
            for cls in range(self.n_classes):
                cls_pixels = np.sum(mask == cls)
                cls_percentage = (cls_pixels / total_pixels) * 100
                cls_name = self.class_names[cls] if cls < len(self.class_names) else f'class_{cls}'
                if cls > 0 and cls_pixels > 0:  # Only show non-background classes with pixels
                    print(f"  {cls_name}: {cls_pixels} pixels ({cls_percentage:.2f}%)")
            print(f"  Detected lane contours: {len(lanes)}")

        # Return mask for streamer to cache
        return lanes, vis_img, mask

    def _visualize_mask(self, img, mask, alpha=0.4):
        """
        Create colored overlay from segmentation mask
        Uses same approach as diagnostic script - fills lane areas with solid color
        """
        # Create colored mask (same approach as diagnostic script)
        colored_mask = np.zeros_like(img)

        for cls in range(min(self.n_classes, len(self.colors))):
            colored_mask[mask == cls] = self.colors[cls]

        # Blend colored mask with original image (60% image, 40% colored mask)
        # Same as diagnostic script: cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)
        overlay = cv2.addWeighted(img, 1 - alpha, colored_mask, alpha, 0)

        return overlay

    def _extract_lane_points(self, mask):
        """
        Extract lane centerlines from segmentation mask
        Returns list of point arrays for compatibility with existing code

        For multi-class models, extracts lanes from each non-background class
        """
        lanes = []

        # Extract lanes from each class (skip background class 0)
        for cls in range(1, self.n_classes):
            # Create binary mask for this class
            class_mask = (mask == cls).astype(np.uint8) * 255

            # Find contours of lane regions for this class
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) > 10:  # Filter small contours
                    # Simplify contour
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    points = approx.reshape(-1, 2)
                    lanes.append(points)

        return lanes
