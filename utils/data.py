import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class YOLOv8Dataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.images = [x for x in os.listdir(image_dir) if x.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Load annotations and adjust box format from normalized to absolute [x1, y1, x2, y2]
        annot_path = os.path.join(self.annotation_dir, self.images[idx].replace(".jpg", ".txt"))
        boxes = []
        labels = []
        area = []
        with open(annot_path, 'r') as file:
            for line in file.readlines():
                class_label, x_center, y_center, width_rel, height_rel = [float(x) for x in line.strip().split()]
                # Convert from relative to absolute coordinates
                x1 = (x_center - width_rel / 2) * width
                y1 = (y_center - height_rel / 2) * height
                x2 = (x_center + width_rel / 2) * width
                y2 = (y_center + height_rel / 2) * height
                boxes.append([x1, y1, x2, y2])
                labels.append(int(class_label))
                area.append((x2 - x1) * (y2 - y1))

        # Convert to Torch tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        area = torch.tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)  # Assuming no crowd instances

        # Image ID
        image_id = torch.tensor([idx])

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd': iscrowd}

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        return image, target

