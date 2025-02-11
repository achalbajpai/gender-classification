import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse


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

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()
        prediction = "Male" if probability > 0.5 else "Female"
        confidence = probability if probability > 0.5 else 1 - probability
        return prediction, confidence * 100


def main():
    parser = argparse.ArgumentParser(description="Predict gender from facial image")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument(
        "--model", type=str, default="best_model.pth", help="Path to the model file"
    )
    args = parser.parse_args()

    try:
        # Load model
        model = load_model(args.model)

        # Transform image
        image_tensor = transform_image(args.image_path)

        # Make prediction
        prediction, confidence = predict(model, image_tensor)

        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
