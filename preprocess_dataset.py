import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import urllib.request


def download_cascade_file():
    """Download the Haar cascade file if it doesn't exist."""
    cascade_file = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_file):
        print("Downloading face detection model...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_file)
        print("Download completed.")
    return cascade_file


def create_directory_structure():
    """Create the necessary directory structure for the processed dataset."""
    base_dir = "processed_dataset"
    splits = ["train", "val", "test"]
    classes = ["male", "female"]

    for split in splits:
        for cls in classes:
            Path(f"{base_dir}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

    return base_dir


def extract_faces(image_path):
    """Extract faces from an image using OpenCV."""
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return []

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the face cascade
    cascade_file = download_cascade_file()
    face_cascade = cv2.CascadeClassifier(cascade_file)

    if face_cascade.empty():
        raise ValueError("Error loading face cascade classifier")

    # Detect faces with adjusted parameters for better detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    face_images = []
    for x, y, w, h in faces:
        # Add margin around the face
        margin = int(0.2 * w)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        # Extract and resize face
        face = image[y : y + h, x : x + w]
        face = cv2.resize(face, (224, 224))
        face_images.append(face)

    return face_images


def process_dataset(source_dir, output_dir):
    """Process the entire dataset of group photos."""
    # Get all image files recursively
    image_files = []
    for ext in [".jpg", ".jpeg", ".png"]:
        image_files.extend(Path(source_dir).rglob(f"*{ext}"))

    total_faces = 0
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            faces = extract_faces(img_path)
            if not faces:
                continue

            # Save each detected face
            for idx, face in enumerate(faces):
                # Generate unique filename
                face_filename = f"{img_path.stem}_face_{idx}.jpg"

                # Save to temporary directory for manual sorting
                temp_dir = Path(output_dir) / "temp_faces"
                temp_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(temp_dir / face_filename), face)
                total_faces += 1
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
            continue

    print(f"\nTotal faces extracted: {total_faces}")
    print("\nFaces have been saved to the temp_faces directory.")
    print("Please manually sort these faces into male/female categories.")
    print(
        "After sorting, run the split_dataset function to create train/val/test splits."
    )


def split_dataset(temp_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """Split the manually sorted dataset into train/val/test sets."""
    for gender in ["male", "female"]:
        source_dir = Path(temp_dir) / gender
        if not source_dir.exists():
            print(f"Directory not found: {source_dir}")
            continue

        files = list(source_dir.glob("*.jpg"))
        np.random.shuffle(files)

        # Calculate split indices
        n_files = len(files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)

        # Split files
        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]

        # Copy files to respective directories
        for split, split_files in [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ]:
            target_dir = Path(output_dir) / split / gender
            target_dir.mkdir(parents=True, exist_ok=True)

            for f in split_files:
                shutil.copy2(f, target_dir / f.name)

        print(f"\n{gender.capitalize()} split statistics:")
        print(f"Train: {len(train_files)}")
        print(f"Validation: {len(val_files)}")
        print(f"Test: {len(test_files)}")


if __name__ == "__main__":
    try:
        # Create directory structure
        output_dir = create_directory_structure()

        # Process the dataset
        source_dir = "/Users/achal/Downloads/gender-classification/Dataset/new-dataset"
        process_dataset(source_dir, output_dir)

        print("\nNext steps:")
        print("1. Check the 'temp_faces' directory")
        print("2. Create 'male' and 'female' subdirectories in 'temp_faces'")
        print("3. Manually sort the faces into these directories")
        print("4. Run the split_dataset function with:")
        print(
            '   python3 -c \'from preprocess_dataset import split_dataset; split_dataset("processed_dataset/temp_faces", "processed_dataset")\''
        )
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
