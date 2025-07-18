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
            # 更强的数据增强
            self.aug = A.Compose([
                # 几何变换
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
                
                # 颜色变换
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
                
                # 噪声和模糊
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.MotionBlur(blur_limit=7, p=0.3),
                
                # 弹性变换
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                
                # 随机裁剪和填充
                A.RandomCrop(height=image_size[0], width=image_size[1], p=0.3),
                
                # 归一化
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet标准化
                ToTensorV2()
            ])
        else:
            # 验证集只做归一化
            self.aug = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        # 读取图像和掩码
        image = np.array(Image.open(img_path).convert('RGB').resize(self.image_size, Image.BILINEAR))
        mask = np.array(Image.open(mask_path).convert('L').resize(self.image_size, Image.NEAREST))
        
        # 应用数据增强
        augmented = self.aug(image=image, mask=mask)
        image = augmented['image']
        mask = (augmented['mask'] > 0.5).float().unsqueeze(0)  # [1, H, W]
        
        return image, mask

class SegmentationDatasetMixup(Dataset):
    """支持Mixup的数据集类"""
    def __init__(self, images_dir, masks_dir, image_size=(256, 256), is_train=True, mixup_alpha=0.2):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.is_train = is_train
        self.mixup_alpha = mixup_alpha
        
        if is_train:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
        
        # 应用数据增强
        augmented = self.aug(image=image, mask=mask)
        image = augmented['image']
        mask = (augmented['mask'] > 0.5).float().unsqueeze(0)
        
        # 训练时应用Mixup
        if self.is_train and np.random.random() < 0.5:
            # 随机选择另一个样本
            idx2 = np.random.randint(0, len(self.images))
            img_name2 = self.images[idx2]
            img_path2 = os.path.join(self.images_dir, img_name2)
            mask_path2 = os.path.join(self.masks_dir, img_name2)
            
            image2 = np.array(Image.open(img_path2).convert('RGB').resize(self.image_size, Image.BILINEAR))
            mask2 = np.array(Image.open(mask_path2).convert('L').resize(self.image_size, Image.NEAREST))
            
            augmented2 = self.aug(image=image2, mask=mask2)
            image2 = augmented2['image']
            mask2 = (augmented2['mask'] > 0.5).float().unsqueeze(0)
            
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            image = lam * image + (1 - lam) * image2
            mask = lam * mask + (1 - lam) * mask2
        
        return image, mask 