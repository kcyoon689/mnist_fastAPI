import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import model
import random
from utils import get_dataloader

# parser = argparse.ArgumentParser(description='Predict MNIST digits from an image file.')
# parser.add_argument('--image_path', type=str, default='./sample_image.png', required=True, help='Path to the input image')
# args = parser.parse_args()

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def predict(img: Image, model_path="./weights/model_10epochs_240414_164948.pth"):
    # print(img_path)
    # image = Image.open(img_path)
    image = transform(img).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance = model.MnistModel().to(device)
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.eval()

    # prediction
    with torch.no_grad():
        image = image.to(device)
        outputs = model_instance(image)
        _, predicted = torch.max(outputs, 1)
        confidence = (
            torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        )  # 클래스별 확률

    return predicted.item(), confidence


if __name__ == "__main__":
    image_path = "./samples/sample_image.png"
    image = Image.open(image_path)
    prediction, confidence = predict(image)
    print(f"Predicted digit: {prediction}")
    print(f"Confidence: {confidence[prediction]}%")
