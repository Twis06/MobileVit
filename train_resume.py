import os
import logging
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from seg_model import SegMobileViT_DeepLabV3
from losses import TverskyLoss
from eval import compute_iou, compute_dice
import torch.optim as optim

logging.getLogger("PIL").setLevel(logging.WARNING)

# 配置
images_dir = 'Dataset/train/images'
masks_dir = 'Dataset/train/masks'
val_images_dir = 'Dataset/test/images'
val_masks_dir = 'Dataset/test/masks'
image_size = (256, 256)
batch_size = 64
epochs = 100
lr = 1e-4
num_workers = 16
save_path = 'best_model.pth'
checkpoint_path = 'checkpoint.pth'
start_epoch = 1
best_val_iou = 0.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 数据集
train_dataset = SegmentationDataset(images_dir, masks_dir, image_size=image_size, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataset = SegmentationDataset(val_images_dir, val_masks_dir, image_size=image_size, is_train=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 模型
model = SegMobileViT_DeepLabV3(backbone_name='mobilevit_xs', num_classes=1, image_size=image_size)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_iou = checkpoint.get('best_val_iou', 0.0)
    print(f"Resumed from {checkpoint_path}, epoch {start_epoch}")
elif os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Loaded weights from {save_path}, optimizer/scheduler reset, training from epoch 1.")

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {param_count}")

criterion = TverskyLoss(alpha=0.9, beta=0.1)
scaler = torch.cuda.amp.GradScaler()
for epoch in range(start_epoch, epochs+1):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    # 验证集评估
    model.eval()
    val_loss = 0.0
    iou_scores = []
    dice_scores = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs)
            for i in range(images.size(0)):
                iou = compute_iou(preds[i,0], masks[i,0])
                dice = compute_dice(preds[i,0], masks[i,0])
                iou_scores.append(iou)
                dice_scores.append(dice)
    val_loss = val_loss / len(val_loader.dataset)
    val_iou = sum(iou_scores) / len(iou_scores)
    val_dice = sum(dice_scores) / len(dice_scores)
    print(f'Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}')
    scheduler.step(val_loss)
    # 保存最佳模型
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), save_path)
        print(f'Best model saved at epoch {epoch} with Val IoU {best_val_iou:.4f}')
    # 保存断点
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_iou': best_val_iou
    }, checkpoint_path) 