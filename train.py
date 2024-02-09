import torch
import torchvision
from torchvision.models import ResNet18_Weights, resnet18, ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torch.utils.data import DataLoader

import argparse
import os

from utils.data import YOLOv8Dataset

# Define the model
def get_lightweight_rcnn_model(num_classes):
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.DEFAULT)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

# Define transformations
def get_transform():
    custom_transforms = []
    custom_transforms.append(T.ToTensor())
    return T.Compose(custom_transforms)



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a Faster R-CNN model')
    arg_parser.add_argument('--data', type=str, help='Path to the dataset')
    arg_parser.add_argument('--output', type=str, help='Path to save the trained model')
    arg_parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')

    data_dir = arg_parser.parse_args().data
    output_dir = arg_parser.parse_args().output
    num_workers = arg_parser.parse_args().num_workers
    if not os.path.exists(data_dir):
        raise ValueError(f"Dataset not found at {data_dir}")
    
    train_dir = os.path.join(data_dir, 'train')
    # val_dir = os.path.join(data_dir, 'valid')
    val_dir = os.path.join(data_dir, 'train')
    train_dataset = YOLOv8Dataset(os.path.join(train_dir, 'images'), os.path.join(train_dir, 'labels'), transform=get_transform())
    val_dataset = YOLOv8Dataset(os.path.join(val_dir, 'images'),os.path.join(val_dir, 'labels'))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=lambda x: tuple(zip(*x)))
    
    num_classes = 5
    model = get_lightweight_rcnn_model(num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    # Parameters and optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        total_loss = []
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            # print(images[0].shape)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # print(targets[0])
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss.append(losses.item())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            # exit(1)
        print(f"Epoch: {epoch+1}, Loss: {sum(total_loss)/len(total_loss)}")

    # Save the model with a unique name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path = os.path.join(output_dir, 'model.pth')  
    torch.save(model.state_dict(), model_path)