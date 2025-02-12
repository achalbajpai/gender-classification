import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn


class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        # Use ResNet50 for better feature extraction
        self.base_model = models.resnet50(pretrained=True)

        # Modify the final layers
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.base_model(x)


def load_model(model_path="best_model_new.pth"):
    model = GenderClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def transform_image(image):
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def detect_and_classify_faces(image_path, model, min_confidence=0):
    # Load cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")

    # Convert to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    results = []
    for x, y, w, h in faces:
        # Extract face with margin
        margin = int(0.2 * w)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)

        face = img_rgb[y1:y2, x1:x2]
        face_pil = Image.fromarray(face)

        # Predict gender
        face_tensor = transform_image(face_pil)
        with torch.no_grad():
            output = model(face_tensor)
            probability = torch.sigmoid(output).item()

            # Use threshold of 0.6 for male classification
            threshold = 0.6
            prediction = "Male" if probability > threshold else "Female"
            confidence = probability if probability > threshold else 1 - probability

            if confidence * 100 >= min_confidence:
                results.append(
                    {
                        "gender": prediction,
                        "confidence": confidence * 100,
                        "bbox": (x, y, w, h),
                    }
                )

    return {
        "total_faces": len(results),
        "males": sum(1 for r in results if r["gender"] == "Male"),
        "females": sum(1 for r in results if r["gender"] == "Female"),
        "results": results,
        "image": img_rgb,
    }


def create_visualization(image_path, output_path="visualization.png"):
    # Load model and process image
    model = load_model()
    results = detect_and_classify_faces(image_path, model)

    # Create figure
    plt.figure(figsize=(20, 10))

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(results["image"])
    plt.title("Original Image")
    plt.axis("off")

    # Create visualization with squares
    viz_img = results["image"].copy()

    # Define colors
    male_color = (0, 128, 255)  # Orange for male
    female_color = (255, 128, 0)  # Blue for female

    # Draw boxes and labels
    for result in results["results"]:
        x, y, w, h = result["bbox"]
        gender = result["gender"]
        confidence = result["confidence"]

        # Set color based on gender
        color = male_color if gender == "Male" else female_color

        # Draw rectangle
        cv2.rectangle(viz_img, (x, y), (x + w, y + h), color, 3)

        # Create label with gender and confidence
        label = f"{gender}: {confidence:.1f}%"

        # Calculate text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Draw background rectangle for text
        cv2.rectangle(
            viz_img, (x, y - text_height - 10), (x + text_width + 10, y), color, -1
        )

        # Add text
        cv2.putText(
            viz_img, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness
        )

    # Plot visualization
    plt.subplot(1, 2, 2)
    plt.imshow(viz_img)
    plt.title("Gender Classification Results")
    plt.axis("off")

    # Add summary text
    summary_text = f"Total Faces: {results['total_faces']}\n"
    summary_text += f"Males: {results['males']}\n"
    summary_text += f"Females: {results['females']}"
    plt.figtext(
        0.02, 0.02, summary_text, fontsize=12, bbox=dict(facecolor="white", alpha=0.8)
    )

    # Save the visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved as {output_path}")
    return results


if __name__ == "__main__":
    # Test the visualization
    image_path = "trial100.png"
    results = create_visualization(image_path, "worst_visualization.png")

    # Print detailed results
    print("\nDetailed Results:")
    print(f"Total faces detected: {results['total_faces']}")
    print(f"Males detected: {results['males']}")
    print(f"Females detected: {results['females']}")

    print("\nIndividual face results:")
    for i, result in enumerate(results["results"], 1):
        print(f"\nFace {i}:")
        print(f"Gender: {result['gender']}")
        print(f"Confidence: {result['confidence']:.1f}%")
