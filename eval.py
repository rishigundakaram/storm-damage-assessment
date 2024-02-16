import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models import ResNet18_Weights, resnet18, ResNet50_Weights
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import os
import sys
import argparse

def load_model(checkpoint_path, num_classes):
    # Load the trained model from file
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.DEFAULT)
    model = FasterRCNN(backbone, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def predict(model, device, img_path):
    # Load the image and convert to tensor
    img = Image.open(img_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Extract predictions
    pred_boxes = predictions[0]['boxes']
    pred_scores = predictions[0]['scores']
    pred_labels = predictions[0]['labels']
    
    return pred_boxes, pred_scores, pred_labels


def draw_boxes(img, prediction):
    classes = {
        0: "background",
        1: "damagedcommercialbuilding", 
        2: "damagedresidentialbuilding",
        3: "undamagedcommercialbuilding",
        4: "undamagedresidentialbuilding",
        
    }
    # Ensure the image is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    draw = ImageDraw.Draw(img)
    predictions_text = []

    for element in range(len(prediction['boxes'])):
        boxes = prediction['boxes'][element].cpu().numpy()
        label = prediction['labels'][element].item()
        score = prediction['scores'][element].item()

        # Format the predictions text
        box_text = f"{classes[label]} {score:.2f}"
        box_text = f"{classes[label]} {score} {boxes[0]} {boxes[1]} {boxes[2]} {boxes[3]}"
        predictions_text.append(box_text)

        draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="red", width=3)
        draw.text((boxes[0], boxes[1]), box_text, fill="red")
    
    return img, predictions_text

def save_predictions_to_txt(predictions_text, img_file, output_dir):
    txt_filename = os.path.splitext(img_file)[0] + '.txt'
    txt_path = os.path.join(output_dir, txt_filename)
    with open(txt_path, 'w') as f:
        for line in predictions_text:
            f.write("%s\n" % line)


from torchvision.ops import nms

def apply_nms(boxes, scores, labels, iou_threshold=0.5):
    keep = nms(boxes, scores, iou_threshold)
    nms_boxes = boxes[keep]
    nms_scores = scores[keep]
    nms_labels = labels[keep]
    return nms_boxes, nms_scores, nms_labels

def main(input_dir, output_dir, checkpoint_path, nms=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, 5).to(device)
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_file in os.listdir(input_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_file)
            boxes, scores, labels = predict(model, device, img_path)
            
            # Apply NMS
            if nms: 
                boxes, scores, labels = apply_nms(boxes, scores, labels)

            # Convert to format expected by draw_boxes and save_predictions_to_txt
            predictions_nms = {'boxes': boxes, 'labels': labels, 'scores': scores}
            
            img = Image.open(img_path).convert("RGB")
            img_with_boxes, predictions_text = draw_boxes(img, predictions_nms)
            img_with_boxes.save(os.path.join(output_dir, img_file))
            save_predictions_to_txt(predictions_text, img_file, output_dir)
    # zip all of the txt files
    os.system(f"zip -j {output_dir}/predictions.zip {output_dir}/*.txt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model on a directory of images')
    parser.add_argument('--input_dir', type=str, help='Directory of input images')
    parser.add_argument('--output_dir', type=str, help='Directory to save output images')
    parser.add_argument('--chkpt', type=str, help='Path to model checkpoint')
    parser.add_argument('--nms', type=bool, help='Non-maximum suppression threshold', default=False)
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.chkpt, args.nms)
