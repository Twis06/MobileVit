import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=(256, 256), transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.transform = transform
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)
        # 数据增强
        if self.transform:
            image = self.transform(image)
        else:
            aug = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = aug(image)
            image = T.ToTensor()(image)
            image = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image)
        mask = T.ToTensor()(mask)
        mask = (mask > 0.5).float()  # 保证为0/1
        return image, mask 