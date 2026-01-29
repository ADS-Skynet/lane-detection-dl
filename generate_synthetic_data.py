#!/usr/bin/env python3
"""
Generate synthetic training data through augmentation
"""

import cv2
import numpy as np
import json
from pathlib import Path
import shutil
from tqdm import tqdm

class SyntheticDataGenerator:
    """Generate augmented versions of existing data"""

    def __init__(self, source_img_dir, source_ann_dir,
                 output_img_dir, output_ann_dir):
        self.source_img_dir = Path(source_img_dir)
        self.source_ann_dir = Path(source_ann_dir)
        self.output_img_dir = Path(output_img_dir)
        self.output_ann_dir = Path(output_ann_dir)

        # Create output directories
        self.output_img_dir.mkdir(parents=True, exist_ok=True)
        self.output_ann_dir.mkdir(parents=True, exist_ok=True)

    def augment_brightness_contrast(self, img, alpha_range=(0.6, 1.4),
                                    beta_range=(-40, 40)):
        """Random brightness and contrast adjustment"""
        alpha = np.random.uniform(*alpha_range)
        beta = np.random.uniform(*beta_range)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def augment_blur(self, img):
        """Random blur"""
        blur_type = np.random.choice(['gaussian', 'median', 'none'])
        if blur_type == 'gaussian':
            ksize = np.random.choice([3, 5, 7])
            return cv2.GaussianBlur(img, (ksize, ksize), 0)
        elif blur_type == 'median':
            ksize = np.random.choice([3, 5])
            return cv2.medianBlur(img, ksize)
        return img

    def augment_noise(self, img, noise_level=15):
        """Add random Gaussian noise"""
        noise = np.random.randn(*img.shape) * noise_level
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def augment_shadow(self, img):
        """Add random shadow overlay"""
        h, w = img.shape[:2]

        # Random shadow polygon
        top_y = int(h * np.random.uniform(0, 0.3))
        bottom_y = int(h * np.random.uniform(0.7, 1.0))

        # Create shadow mask
        mask = np.zeros((h, w), dtype=np.float32)
        pts = np.array([
            [0, top_y],
            [w, top_y],
            [w, bottom_y],
            [0, bottom_y]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1.0)

        # Apply shadow
        shadow_intensity = np.random.uniform(0.3, 0.7)
        mask = mask * shadow_intensity

        result = img.copy().astype(np.float32)
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - mask) + result[:, :, c] * mask * 0.5

        return np.clip(result, 0, 255).astype(np.uint8)

    def augment_hue_saturation(self, img):
        """Random hue and saturation adjustment"""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Adjust hue
        hue_shift = np.random.uniform(-10, 10)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

        # Adjust saturation
        sat_scale = np.random.uniform(0.7, 1.3)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def augment_perspective(self, img, lanes, max_warp=0.05):
        """Slight perspective warp"""
        h, w = img.shape[:2]

        # Random perspective transform points
        src_pts = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])

        dst_pts = src_pts + np.random.uniform(-max_warp, max_warp,
                                              src_pts.shape) * np.array([w, h])
        dst_pts = np.float32(dst_pts)

        # Get transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Warp image
        warped_img = cv2.warpPerspective(img, M, (w, h))

        # Warp lanes
        warped_lanes = []
        for lane in lanes:
            if not lane:
                warped_lanes.append([])
                continue

            lane_pts = np.array(lane, dtype=np.float32).reshape(-1, 1, 2)
            warped_pts = cv2.perspectiveTransform(lane_pts, M)
            warped_lane = warped_pts.reshape(-1, 2).tolist()
            warped_lanes.append(warped_lane)

        return warped_img, warped_lanes

    def flip_horizontal(self, img, lanes):
        """Horizontal flip"""
        h, w = img.shape[:2]
        flipped_img = cv2.flip(img, 1)

        flipped_lanes = []
        for lane in lanes:
            if not lane:
                flipped_lanes.append([])
                continue
            flipped_lane = [[w - x, y] for x, y in lane]
            flipped_lanes.append(flipped_lane)

        return flipped_img, flipped_lanes

    def generate_augmentations(self, img, lanes, num_augmentations=5):
        """Generate multiple augmented versions"""
        h, w = img.shape[:2]
        augmented_samples = []

        for i in range(num_augmentations):
            aug_img = img.copy()
            aug_lanes = [lane[:] for lane in lanes]  # Deep copy

            # Apply random combination of augmentations

            # 1. Brightness/Contrast (80% chance)
            if np.random.rand() > 0.2:
                aug_img = self.augment_brightness_contrast(aug_img)

            # 2. Hue/Saturation (60% chance)
            if np.random.rand() > 0.4:
                aug_img = self.augment_hue_saturation(aug_img)

            # 3. Blur (40% chance)
            if np.random.rand() > 0.6:
                aug_img = self.augment_blur(aug_img)

            # 4. Noise (50% chance)
            if np.random.rand() > 0.5:
                aug_img = self.augment_noise(aug_img)

            # 5. Shadow (30% chance)
            if np.random.rand() > 0.7:
                aug_img = self.augment_shadow(aug_img)

            # 6. Perspective warp (40% chance)
            if np.random.rand() > 0.6:
                aug_img, aug_lanes = self.augment_perspective(aug_img, aug_lanes)

            # 7. Horizontal flip (50% chance)
            if np.random.rand() > 0.5:
                aug_img, aug_lanes = self.flip_horizontal(aug_img, aug_lanes)

            augmented_samples.append((aug_img, aug_lanes))

        return augmented_samples

    def process_dataset(self, augmentations_per_image=5):
        """Process all images and generate synthetic data"""

        # First, copy original data
        print("Copying original data...")
        for img_file in tqdm(list(self.source_img_dir.glob('*.png')) +
                            list(self.source_img_dir.glob('*.jpg'))):
            shutil.copy(img_file, self.output_img_dir / img_file.name)

        for ann_file in self.source_ann_dir.glob('*.json'):
            shutil.copy(ann_file, self.output_ann_dir / ann_file.name)

        # Load all annotations
        annotations = []
        for json_file in sorted(self.source_ann_dir.glob('*.json')):
            try:
                with open(json_file, encoding='utf-8') as f:
                    annotations.append((json_file.stem, json.load(f)))
            except:
                try:
                    with open(json_file, encoding='latin-1') as f:
                        annotations.append((json_file.stem, json.load(f)))
                except Exception as e:
                    print(f"Warning: Could not load {json_file}: {e}")
                    continue

        print(f"\nGenerating {augmentations_per_image} augmentations per image...")
        print(f"Original images: {len(annotations)}")
        print(f"Total images after augmentation: {len(annotations) * (1 + augmentations_per_image)}")

        # Generate augmentations
        for file_stem, ann in tqdm(annotations):
            # Load original image
            img_path = self.source_img_dir / ann['image']
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Generate augmented versions
            augmented_samples = self.generate_augmentations(
                img, ann['lanes'], augmentations_per_image
            )

            # Save augmented images and annotations
            for idx, (aug_img, aug_lanes) in enumerate(augmented_samples):
                # Save image
                aug_img_name = f"{file_stem}_aug{idx}.png"
                aug_img_path = self.output_img_dir / aug_img_name
                cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

                # Save annotation
                aug_ann = {
                    'image': aug_img_name,
                    'lanes': aug_lanes
                }
                aug_ann_path = self.output_ann_dir / f"{file_stem}_aug{idx}.json"
                with open(aug_ann_path, 'w') as f:
                    json.dump(aug_ann, f, indent=2)

        # Count final dataset
        final_img_count = len(list(self.output_img_dir.glob('*.png'))) + \
                         len(list(self.output_img_dir.glob('*.jpg')))
        final_ann_count = len(list(self.output_ann_dir.glob('*.json')))

        print(f"\n✓ Synthetic data generation complete!")
        print(f"  Output images: {final_img_count}")
        print(f"  Output annotations: {final_ann_count}")
        print(f"  Augmentation ratio: {final_img_count / len(annotations):.1f}x")


def main():
    # Configuration
    source_img_dir = 'training_data/images'
    source_ann_dir = 'training_data/annotations'
    output_img_dir = 'training_data_augmented/images'
    output_ann_dir = 'training_data_augmented/annotations'

    # Number of augmented versions per original image
    augmentations_per_image = 5  # Will create 6x dataset (1 original + 5 augmented)

    print("="*60)
    print("SYNTHETIC DATA GENERATOR")
    print("="*60)
    print(f"Source: {source_img_dir}")
    print(f"Output: {output_img_dir}")
    print(f"Augmentations per image: {augmentations_per_image}")
    print("="*60 + "\n")

    # Generate synthetic data
    generator = SyntheticDataGenerator(
        source_img_dir, source_ann_dir,
        output_img_dir, output_ann_dir
    )

    generator.process_dataset(augmentations_per_image)

    print(f"\n✓ Done! Use '{output_img_dir}' and '{output_ann_dir}' for training")


if __name__ == '__main__':
    main()