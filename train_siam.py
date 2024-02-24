import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import logging
import argparse
import os

from utils.data import SiameseDataset
from utils.models import get_model

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
    logging.basicConfig(level=logging.DEBUG)
    logging.info('parsing arguments')

    data_dir = arg_parser.parse_args().data
    output_dir = arg_parser.parse_args().output
    num_workers = arg_parser.parse_args().num_workers
    if not os.path.exists(data_dir):
        raise ValueError(f"Dataset not found at {data_dir}")
    
    train_dir = os.path.join(data_dir, 'train')
    # val_dir = os.path.join(data_dir, 'valid')
    val_dir = os.path.join(data_dir, 'train')
    pre_images = [
        '/home/projects/storm-damage/data/intermediate/Post_Event_Grids_In_JPEG_all_buildings', 
        '/home/projects/storm-damage/data/intermediate/Post_Event_Grids_In_JPEG'
        ]
    logging.debug('loading datasets')
    train_dataset = SiameseDataset(os.path.join(train_dir, 'images'), pre_images, os.path.join(train_dir, 'labels'), transform=get_transform())
    val_dataset = SiameseDataset(os.path.join(val_dir, 'images'), pre_images, os.path.join(val_dir, 'labels'))
    logging.debug('loading dataloaders')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=num_workers, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=lambda x: tuple(zip(*x)))
    
    num_classes = 5
    model = get_model()

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')    
    model.to(device)
    model.train()

    # Parameters and optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Training loop
    num_epochs = 1000
    logging.debug('entering the training loop')
    for epoch in range(num_epochs):
        model.train()
        total_loss = []
        for pre_images, post_images, targets in train_loader:
            pre_images = list(image.to(device) for image in pre_images)
            post_images = list(image.to(device) for image in post_images)
            logging.debug('have the images and targets on gpu')
            # print(images[0].shape)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # print(targets[0])
            loss_dict = model(pre_images, post_images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss.append(losses.item())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch: {epoch+1}, Loss: {sum(total_loss)/len(total_loss)}")
        # save the model every 5 epochs
        if (epoch+1) % 5 == 0:
            model_path = os.path.join(output_dir, f'siam_model_{epoch+1}.pth')  
            torch.save(model.state_dict(), model_path)
