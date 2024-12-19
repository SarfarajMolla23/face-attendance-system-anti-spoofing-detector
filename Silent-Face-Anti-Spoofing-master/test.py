# -*- coding: utf-8 -*-
# @Time : 20-6-9 3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

SAMPLE_IMAGE_PATH = ""


def adjust_image_aspect_ratio(image, target_ratio=(4, 3)):
    """
    Adjust the aspect ratio of the image to the target ratio by resizing and cropping.
    """
    height, width, _ = image.shape
    target_width, target_height = target_ratio

    if width / height > target_width / target_height:
        # Image is too wide, crop width
        new_width = int(height * (target_width / target_height))
        offset = (width - new_width) // 2
        image = image[:, offset:offset + new_width]
    else:
        # Image is too tall, crop height
        new_height = int(width * (target_height / target_width))
        offset = (height - new_height) // 2
        image = image[offset:offset + new_height, :]

    return image


def test(image=None, image_path=None, model_dir="./resources/anti_spoof_models", device_id=0):
    """
    Perform anti-spoofing detection on an image or an image path.

    Args:
        image: np.array, The image array.
        image_path: str, The path to the image.
        model_dir: str, Directory where the models are stored.
        device_id: int, The GPU device ID.

    Returns:
        label: int, The label of the prediction (0: fake, 1: real).
    """
    if image is None and image_path is None:
        raise ValueError("Either 'image' or 'image_path' must be provided.")

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    # Load image from the path if no image array is provided
    if image is None:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Adjust the image to a 4:3 aspect ratio
    image = adjust_image_aspect_ratio(image, target_ratio=(4, 3))

    # Get the bounding box
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0

    # Iterate through the models and make predictions
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start

    # Draw the result of the prediction
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    print(f"Prediction Label: {label}, Value: {value}")
    print(f"Average Test Speed: {test_speed / len(os.listdir(model_dir))}s")

    return label


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test"
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default=r"C:\Users\sarfa\Documents\PythonProject\Silent-Face-Anti-Spoofing-master\Silent-Face-Anti-Spoofing-master\images\sample\image_F1.jpg",
        help="image used to test"
    )
    args = parser.parse_args()
    test(image_path=args.image_name, model_dir=args.model_dir, device_id=args.device_id)
