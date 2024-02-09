# get the images from the input directory, corresponding labels, and visualize the images with the labels and bouding box
import os
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def visualize_images(input_img_dir, input_label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_img_dir):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(input_img_dir, filename)
            label_path = os.path.join(input_label_dir, filename.replace('.jpg', '.json'))
            with open(label_path, 'r') as file:
                labels = json.load(file)
            img = cv2.imread(img_path)
            # color = (0, 255, 0) if label == 0 else (255, 0, 0)
            for label in labels:
                bounds = label['bounds']
                target = label['label']
                if target == 0:
                    color = (0, 255, 0)
                else: 
                    color = (255, 0, 0)
                x1, y1, x2, y2 = bounds
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)
            print(f"Visualized {filename} and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize images with labels')
    parser.add_argument('input_img_dir', type=str, help='Directory containing images')
    parser.add_argument('input_label_dir', type=str, help='Directory containing labels')
    parser.add_argument('output_dir', type=str, help='Output directory for visualized images')
    args = parser.parse_args()

    visualize_images(args.input_img_dir, args.input_label_dir, args.output_dir)

