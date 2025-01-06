import h5py
import numpy as np
import torch
import shutil
import os
import json
import random
import cv2
from PIL import Image

def create_image_paths_json(directory, output_file="image_paths.json", limit=None):
    """
    Creates a JSON document with all the image paths in a specified directory.

    Args:
        directory (str): Path to the directory (relative to the project root).
        output_file (str): Name of the output JSON file. Defaults to 'image_paths.json'.

    Returns:
        None
    """
    # Define supported image file extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

    # List to store image paths
    image_paths = []

    # Walk through the directory and collect image paths
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                # Add the relative path to the image_paths list
                image_paths.append(os.path.join(root, file))

    random.shuffle(image_paths)

    # Apply the limit if specified
    if limit is not None:
        image_paths = image_paths[:limit]

    # Write the image paths to a JSON file
    with open(output_file, "w") as json_file:
        json.dump(image_paths, json_file, indent=4)

    print(f"JSON document with image paths saved to {output_file}")

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)

def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth.tar'):
    torch.save(state, task_id + filename)
    if is_best:
        shutil.copyfile(task_id + filename, task_id + 'model_best.pth.tar')

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground-truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    target = cv2.resize(target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64

    return img,target
