import cv2
import os
import numpy as np

def resize_images(input_dir, output_dir, img_size=(145, 145)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            resized_img = cv2.resize(img, img_size)
            cv2.imwrite(os.path.join(output_dir, filename), resized_img)
    print(f"Images resized and saved to {output_dir}")
