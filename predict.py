import sys
import os
import torch
from torchvision import transforms
from PIL import Image
from model import CrowdCounterModel

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py <filename> <model>")
        print("<filename>: Path to the image file to evaluate")
        print("<model>: 'dense' or 'sparse'")
        sys.exit(1)

    filename = sys.argv[1]
    model_type = sys.argv[2]

    # Determine model weights based on argument
    model_weights = "1model_best.pth.tar" if model_type == "dense" else "2model_best.pth.tar"

    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist.")
        sys.exit(1)

    if not os.path.exists(model_weights):
        print(f"Error: Model weights {model_weights} do not exist.")
        sys.exit(1)

    # Load the model
    model = CrowdCounterModel()
    model = model.cuda()
    checkpoint = torch.load(model_weights, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])

    # Process the image
    img = transform(Image.open(filename).convert('RGB')).cuda()
    output = model(img.unsqueeze(0))
    predicted_count = int(output.detach().cpu().sum().numpy())

    print(predicted_count)

if __name__ == "__main__":
    main()
