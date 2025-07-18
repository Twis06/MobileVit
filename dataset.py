import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=(256, 256), is_train=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        if is_train:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # 你可以加更多增强
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                ToTensorV2()
            ])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB').resize(self.image_size, Image.BILINEAR))
        mask = np.array(Image.open(mask_path).convert('L').resize(self.image_size, Image.NEAREST))
        augmented = self.aug(image=image, mask=mask)
        image = augmented['image']
        mask = (augmented['mask'] > 0.5).float().unsqueeze(0)  # [1, H, W]
        return image, mask 