"""
Download crop disease images for training using bing_image_downloader.
Downloads ~50 images per disease class for 6 crops.
"""

import os
import shutil
from bing_image_downloader import downloader

# Define crops and their diseases
CROP_DISEASES = {
    "Sugarcane": ["Sugarcane Red Rot Disease", "Sugarcane Smut Disease", "Sugarcane Rust Disease", "Healthy Sugarcane Leaf"],
    "Pulses": ["Pulses Anthracnose Disease", "Pulses Powdery Mildew Disease", "Pulses Rust Disease", "Healthy Pulses Leaf"],
    "Maize": ["Maize Northern Leaf Blight", "Maize Common Rust Disease", "Maize Gray Leaf Spot", "Healthy Maize Leaf"],
    "Wheat": ["Wheat Leaf Rust Disease", "Wheat Septoria Disease", "Wheat Yellow Rust Disease", "Healthy Wheat Leaf"],
    "Paddy": ["Paddy Rice Blast Disease", "Paddy Brown Spot Disease", "Paddy Leaf Scald Disease", "Healthy Paddy Rice Leaf"],
    "Mustard": ["Mustard White Rust Disease", "Mustard Alternaria Blight", "Mustard Downy Mildew Disease", "Healthy Mustard Leaf"],
}

# Shorter class labels for model training
CLASS_LABELS = {
    "Sugarcane Red Rot Disease": "Sugarcane_Red_Rot",
    "Sugarcane Smut Disease": "Sugarcane_Smut",
    "Sugarcane Rust Disease": "Sugarcane_Rust",
    "Healthy Sugarcane Leaf": "Sugarcane_Healthy",
    "Pulses Anthracnose Disease": "Pulses_Anthracnose",
    "Pulses Powdery Mildew Disease": "Pulses_Powdery_Mildew",
    "Pulses Rust Disease": "Pulses_Rust",
    "Healthy Pulses Leaf": "Pulses_Healthy",
    "Maize Northern Leaf Blight": "Maize_Northern_Leaf_Blight",
    "Maize Common Rust Disease": "Maize_Common_Rust",
    "Maize Gray Leaf Spot": "Maize_Gray_Leaf_Spot",
    "Healthy Maize Leaf": "Maize_Healthy",
    "Wheat Leaf Rust Disease": "Wheat_Leaf_Rust",
    "Wheat Septoria Disease": "Wheat_Septoria",
    "Wheat Yellow Rust Disease": "Wheat_Yellow_Rust",
    "Healthy Wheat Leaf": "Wheat_Healthy",
    "Paddy Rice Blast Disease": "Paddy_Blast",
    "Paddy Brown Spot Disease": "Paddy_Brown_Spot",
    "Paddy Leaf Scald Disease": "Paddy_Leaf_Scald",
    "Healthy Paddy Rice Leaf": "Paddy_Healthy",
    "Mustard White Rust Disease": "Mustard_White_Rust",
    "Mustard Alternaria Blight": "Mustard_Alternaria_Blight",
    "Mustard Downy Mildew Disease": "Mustard_Downy_Mildew",
    "Healthy Mustard Leaf": "Mustard_Healthy",
}

IMAGES_PER_CLASS = 50
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")


def download_all_images():
    """Download images for all crop diseases."""
    print("=" * 60)
    print("  CROP DISEASE IMAGE DOWNLOADER")
    print("=" * 60)

    os.makedirs(DATASET_DIR, exist_ok=True)

    total_classes = sum(len(diseases) for diseases in CROP_DISEASES.values())
    current = 0

    for crop, diseases in CROP_DISEASES.items():
        print(f"\n{'─' * 40}")
        print(f"  Crop: {crop}")
        print(f"{'─' * 40}")

        for search_query in diseases:
            current += 1
            label = CLASS_LABELS[search_query]
            target_dir = os.path.join(DATASET_DIR, label)

            if os.path.exists(target_dir) and len(os.listdir(target_dir)) >= 10:
                print(f"  [{current}/{total_classes}] SKIP (already exists): {label}")
                continue

            print(f"  [{current}/{total_classes}] Downloading: {search_query} → {label}")

            try:
                downloader.download(
                    search_query,
                    limit=IMAGES_PER_CLASS,
                    output_dir=DATASET_DIR,
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=60,
                    verbose=False,
                )

                # Rename downloaded folder to our class label
                downloaded_dir = os.path.join(DATASET_DIR, search_query)
                if os.path.exists(downloaded_dir) and downloaded_dir != target_dir:
                    if os.path.exists(target_dir):
                        # Merge into existing directory
                        for f in os.listdir(downloaded_dir):
                            src = os.path.join(downloaded_dir, f)
                            dst = os.path.join(target_dir, f)
                            if not os.path.exists(dst):
                                shutil.move(src, dst)
                        shutil.rmtree(downloaded_dir)
                    else:
                        os.rename(downloaded_dir, target_dir)

                if os.path.exists(target_dir):
                    count = len(os.listdir(target_dir))
                    print(f"    ✓ Downloaded {count} images")
                else:
                    print(f"    ✗ No images downloaded")

            except Exception as e:
                print(f"    ✗ Error: {e}")

    print(f"\n{'=' * 60}")
    print("  Download complete!")
    print(f"  Dataset directory: {DATASET_DIR}")
    print(f"{'=' * 60}")

    # Print summary
    print("\nDataset Summary:")
    if os.path.exists(DATASET_DIR):
        for class_dir in sorted(os.listdir(DATASET_DIR)):
            class_path = os.path.join(DATASET_DIR, class_dir)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
                print(f"  {class_dir}: {count} images")


if __name__ == "__main__":
    download_all_images()
