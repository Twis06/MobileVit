import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optimddsssssssssssssssssssssssssssssssssssssssssssss
from dataset import SegmentationDataset  
from seg_model import SegMobileViT_DeepLabV3
from losses import DiceLoss, FocalLoss, TverskyLoss
from eval import compute_iou, compute_dice
import numpy as np
import logging

# Enable wandb debugging
# os.environ['WANDB_DEBUG'] = 'true'
# os.environ['WANDB_CORE_DEBUG'] = 'true'

# Set up logging
logging.getLogger("PIL").setLevel(logging.WARNING)
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# 配置
images_dir = 'Dataset/train/images'
masks_dir = 'Dataset/train/masks'
val_images_dir = 'Dataset/test/images'  # 验证集路径
val_masks_dir = 'Dataset/test/masks'    # 验证集路径
image_size = (256, 256)
batch_size = 64
epochs = 5000
lr = 1e-5
num_workers = 16
save_path = 'best_model.pth'
checkpoint_path = 'checkpoint.pth'
start_epoch = 1
best_val_iou = 0.0

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 数据集
train_dataset = SegmentationDataset(images_dir, masks_dir, image_size=image_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataset = SegmentationDataset(val_images_dir, val_masks_dir, image_size=image_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 模型
model = SegMobileViT_DeepLabV3(backbone_name='mobilevit_xs', num_classes=1, image_size=image_size)
model = model.to(device)  # Move model to device immediately
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Move optimizer state to device
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_iou = checkpoint.get('best_val_iou', 0.0)
    print(f"Resumed from {checkpoint_path}, epoch {start_epoch}")
elif os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Loaded weights from {save_path}, optimizer/scheduler reset, training from epoch 1.")


# 损失和优化器
criterion = TverskyLoss(alpha=0.98, beta=0.02)
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
    epoch_loss = running_loss / len(train_loader.dataset)
    # 验证集评估
    model.eval()
    val_loss = 0.0
    iou_scores = []
    dice_scores = []
    vis_images = []
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
                # Log a few images to wandb
                if epoch % 5 == 0 and i < 2:
                    img = images[i].detach().cpu().numpy().transpose(1,2,0)
                    img = (img * 0.5 + 0.5).clip(0,1)
                    mask_np = masks[i,0].detach().cpu().numpy()
                    pred_np = (preds[i,0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                    
    val_loss = val_loss / len(val_loader.dataset)
    val_iou = sum(iou_scores) / len(iou_scores)
    val_dice = sum(dice_scores) / len(dice_scores)
    lr_current = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}')
    scheduler.step(val_loss)
    # wandb log
    # wandb.log({
    #     "epoch": epoch,
    #     "train_loss": epoch_loss,
    #     "val_loss": val_loss,
    #     "val_iou": val_iou,
    #     "val_dice": val_dice,
    #     "lr": lr_current,
    #     "images": vis_images if vis_images else None
    # })
    # 保存最佳模型
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), save_path)
        print(f'Best model saved at epoch {epoch} with Val IoU {best_val_iou:.4f}')
        # wandb.run.summary["best_val_iou"] = best_val_iou
    # 保存断点
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_iou': best_val_iou
    }, checkpoint_path) 