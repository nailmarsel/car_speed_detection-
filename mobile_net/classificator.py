import cv2
import torch


from model import load_mobilenetv2_model
from image_utils import preprocess_image


def classify_image(image) -> str:
    model = load_mobilenetv2_model()
    input_batch = preprocess_image(image)
    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    class_idx = torch.argmax(probabilities).item()

    return class_idx


def main():
    image_path = "path/to/your/image.jpg"  # Замените путь на путь к вашему изображению
    image = cv2.imread(image_path)
    class_idx = classify_image(image)
    print(f"The image belongs to class: {class_idx}")

if __name__ == '__main__':
    main()