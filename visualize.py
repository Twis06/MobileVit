import os
import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from seg_model import SegMobileViT_DeepLabV3
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_batch(images_dir, masks_dir, model_path, backbone='mobilevit_xs', image_size=(256,256), num_samples=5, save_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SegmentationDataset(images_dir, masks_dir, image_size=image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = SegMobileViT_DeepLabV3(backbone_name=backbone, num_classes=1, image_size=image_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    shown = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()[0,0]
            img = images.cpu().numpy()[0].transpose(1,2,0)
            img = (img * 0.5 + 0.5).clip(0,1)  # 反归一化
            mask = masks.cpu().numpy()[0,0]
            fig, axs = plt.subplots(1,3,figsize=(12,4))
            axs[0].imshow(img)
            axs[0].set_title('Image')
            axs[1].imshow(mask, cmap='gray')
            axs[1].set_title('Mask')
            axs[2].imshow(preds > 0.5, cmap='gray')
            axs[2].set_title('Prediction')
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'vis_{shown}.png'))
            else:
                plt.show()
            shown += 1
            if shown >= num_samples:
                break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--masks_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='mobilevit_xs')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256,256])
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    visualize_batch(args.images_dir, args.masks_dir, args.model_path, args.backbone, tuple(args.image_size), args.num_samples, args.save_dir) 