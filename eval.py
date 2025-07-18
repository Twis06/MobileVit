import os
import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from seg_model import SegMobileViT_DeepLabV3
import numpy as np

def compute_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()

def compute_dice(pred, target, threshold=0.5, smooth=1.):
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def main():
    images_dir = 'dataset/images'  # 修改为验证集路径
    masks_dir = 'dataset/masks'    # 修改为验证集路径
    model_path = 'best_model.pth'
    image_size = (256, 256)
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SegmentationDataset(images_dir, masks_dir, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SegMobileViT_DeepLabV3(backbone_name='mobilevit_xs', num_classes=1, image_size=image_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    iou_scores = []
    dice_scores = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            for i in range(images.size(0)):
                iou = compute_iou(preds[i,0], masks[i,0])
                dice = compute_dice(preds[i,0], masks[i,0])
                iou_scores.append(iou)
                dice_scores.append(dice)
    print(f'Mean IoU: {np.mean(iou_scores):.4f}')
    print(f'Mean Dice: {np.mean(dice_scores):.4f}')

if __name__ == '__main__':
    main() 