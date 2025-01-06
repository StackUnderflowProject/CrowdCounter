import os
import glob
import random
import h5py
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from model import CrowdCounterModel

# Define the paths
ROOT = 'ShanghaiTech'
PART_A_TEST = os.path.join(ROOT, 'part_A', 'test_data', 'images')
PART_B_TEST = os.path.join(ROOT, 'part_B', 'test_data', 'images')

# Transformation pipeline for images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load the model
def load_model(weights_path):
    model = CrowdCounterModel()
    model = model.cuda()  # Use GPU if available
    checkpoint = torch.load(weights_path, map_location='cuda', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model


def compute_mae(img_paths, model, dataset_name):
    """
    Computes MAE for a set of image paths using the given model.
    """
    mae = 0
    selected_images = random.sample(img_paths, min(100, len(img_paths)))  # Select 100 images randomly
    for img_path in tqdm(selected_images):
        # Load and preprocess the image
        img = transform(Image.open(img_path).convert('RGB')).cuda()

        # Load the ground truth density map
        gt_file = h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground-truth'), 'r')
        groundtruth = np.asarray(gt_file['density'])

        # Predict the count using the model
        output = model(img.unsqueeze(0))
        predicted_count = output.detach().cpu().sum().item()

        # Compute MAE
        mae += abs(predicted_count - np.sum(groundtruth))
    mae = mae / len(selected_images)
    return mae


def main():
    # Get image paths for part A and part B test datasets
    img_paths_a = glob.glob(os.path.join(PART_A_TEST, '*.jpg'))
    img_paths_b = glob.glob(os.path.join(PART_B_TEST, '*.jpg'))

    # Load models for Part A and Part B
    model_a = load_model('1model_best.pth.tar')
    model_b = load_model('2model_best.pth.tar')

    # Compute MAE for part A and part B
    mae_a = compute_mae(img_paths_a, model_a, "Part A Test Set")
    mae_b = compute_mae(img_paths_b, model_b, "Part B Test Set")

    # Print results
    print(f"MAE for Part A Test Set: {mae_a:.3f}")
    print(f"MAE for Part B Test Set: {mae_b:.3f}")


if __name__ == '__main__':
    main()
