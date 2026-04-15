import argparse
import torch
from torchvision import transforms
from PIL import Image
from model.resnet import ResNet18

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

def predict(image_path, model_path="results/best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)

    label = CLASSES[predicted.item()]
    score = confidence.item() * 100

    print(f"{label} ({score:.1f}%)")
    return label, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="results/best_model.pt")
    args = parser.parse_args()

    predict(args.image, args.model)
