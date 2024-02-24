import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pprint import pprint

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


class SiameseDataset(Dataset):
    def __init__(self, post_dir, pre_dirs, annotation_dir,  transform=None):
        self.post_dir = post_dir
        self.pre_dirs = pre_dirs
        self.transform = transform
        self.annotation_dir = annotation_dir
        self.post_images = [os.path.join(post_dir, x) for x in os.listdir(post_dir) if x.endswith('.jpg')]
        def strip_extension(filename):
            filename = filename.split('.')[0][:-4]
            return filename
        def get_pre_images(filename, pre_dirs):
            for pre_dir in pre_dirs:
                pre_images = [x.split('.')[0] for x in os.listdir(pre_dir) if x.endswith('.jpg')]
                print(pre_images)
                print(filename)
                exit()
                if filename in pre_images:
                    return os.path.join(pre_dir, filename)
        print(self.post_images[0], strip_extension(self.post_images[0]))
        exit()
        self.pre_images = [get_pre_images(strip_extension(x), pre_dirs) for x in self.post_images]
        pprint(self.pre_images)
        # pprint(self.post_images)
    
    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        # Load image
        pre_img_path = self.pre_images[idx]
        post_img_path = self.post_images[idx]
        pre_image = Image.open(pre_img_path).convert("RGB")
        post_image = Image.open(post_img_path).convert("RGB")
        width, height = post_image.size

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
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)


        return pre_image, post_image, target
