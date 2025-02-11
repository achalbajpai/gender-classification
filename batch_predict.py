import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 1))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def transform_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()
        prediction = "Male" if probability > 0.5 else "Female"
        confidence = probability if probability > 0.5 else 1 - probability
        return prediction, confidence * 100


def process_folder(
    model, folder_path, output_file="predictions.csv", create_visualization=True
):
    results = []
    folder = Path(folder_path)
    image_files = (
        list(folder.glob("**/*.jpg"))
        + list(folder.glob("**/*.jpeg"))
        + list(folder.glob("**/*.png"))
    )

    print(f"\nProcessing {len(image_files)} images...")

    for img_path in tqdm(image_files):
        image_tensor = transform_image(img_path)
        if image_tensor is not None:
            try:
                prediction, confidence = predict(model, image_tensor)
                results.append(
                    {
                        "image_path": str(img_path),
                        "predicted_gender": prediction,
                        "confidence": confidence,
                        "filename": img_path.name,
                    }
                )
            except Exception as e:
                print(f"\nError predicting {img_path}: {str(e)}")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Print summary statistics
    print("\nPrediction Summary:")
    print(df["predicted_gender"].value_counts())
    print("\nConfidence Statistics:")
    print(df["confidence"].describe())

    if create_visualization and not df.empty:
        # Create visualizations
        plt.figure(figsize=(15, 5))

        # Gender distribution
        plt.subplot(1, 2, 1)
        sns.countplot(data=df, x="predicted_gender")
        plt.title("Gender Distribution")
        plt.ylabel("Count")

        # Confidence distribution
        plt.subplot(1, 2, 2)
        sns.histplot(data=df, x="confidence", bins=20)
        plt.title("Confidence Distribution")
        plt.xlabel("Confidence (%)")
        plt.ylabel("Count")

        plt.tight_layout()
        plt.savefig("batch_predictions_visualization.png")
        print("\nVisualization saved as 'batch_predictions_visualization.png'")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Predict gender for multiple images in a folder"
    )
    parser.add_argument(
        "folder_path", type=str, help="Path to the folder containing images"
    )
    parser.add_argument(
        "--model", type=str, default="best_model.pth", help="Path to the model file"
    )
    parser.add_argument(
        "--output", type=str, default="predictions.csv", help="Output CSV file path"
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Disable visualization generation"
    )
    args = parser.parse_args()

    try:
        # Load model
        print("Loading model...")
        model = load_model(args.model)

        # Process folder
        process_folder(model, args.folder_path, args.output, not args.no_viz)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
