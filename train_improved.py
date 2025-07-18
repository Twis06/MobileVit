import os
import logging
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from seg_model import SegMobileViT_DeepLabV3
from losses import TverskyLoss, DiceLoss, FocalLoss
from eval import compute_iou, compute_dice
import torch.optim as optim
import matplotlib.pyplot as plt

logging.getLogger("PIL").setLevel(logging.WARNING)

# 配置
images_dir = 'Dataset/train/images'
masks_dir = 'Dataset/train/masks'
val_images_dir = 'Dataset/test/images'
val_masks_dir = 'Dataset/test/masks'
image_size = (256, 256)
batch_size = 32  # 减小batch size以增加更新频率
epochs = 100
lr = 5e-5  # 降低学习率
num_workers = 16
save_path = 'best_model_improved.pth'
checkpoint_path = 'checkpoint_improved.pth'
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
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # 使用AdamW和权重衰减

# 基于IoU的学习率调度器
class IoUScheduler:
    def __init__(self, optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-7):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_score = None
        self.counter = 0
        
    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'max' and score > self.best_score) or \
             (self.mode == 'min' and score < self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0
                
    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * self.factor, self.min_lr)
        print(f"Learning rate reduced to {self.optimizer.param_groups[0]['lr']:.2e}")

scheduler = IoUScheduler(optimizer, mode='max', factor=0.7, patience=8)

# 组合损失函数
class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tversky = TverskyLoss(alpha=0.7, beta=0.3)
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=0.8, gamma=2)
        
    def forward(self, inputs, targets):
        tversky_loss = self.tversky(inputs, targets)
        dice_loss = self.dice(torch.sigmoid(inputs), targets)
        focal_loss = self.focal(inputs, targets)
        return self.alpha * tversky_loss + self.beta * dice_loss + self.gamma * focal_loss

criterion = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)

# 动态阈值优化
def find_optimal_threshold(preds, targets, thresholds=np.arange(0.1, 0.9, 0.05)):
    best_iou = 0
    best_threshold = 0.5
    for threshold in thresholds:
        iou_scores = []
        for i in range(preds.size(0)):
            iou = compute_iou(preds[i,0], targets[i,0], threshold=threshold)
            iou_scores.append(iou)
        avg_iou = np.mean(iou_scores)
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_threshold = threshold
    return best_threshold, best_iou

# 加载检查点
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_iou = checkpoint.get('best_val_iou', 0.0)
    print(f"Resumed from {checkpoint_path}, epoch {start_epoch}")

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {param_count}")

# 训练历史记录
train_losses = []
val_losses = []
val_ious = []
val_dices = []
optimal_thresholds = []

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
    train_losses.append(epoch_loss)
    
    # 验证集评估
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())
    
    val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    
    # 合并所有预测和标签
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # 寻找最优阈值
    optimal_threshold, _ = find_optimal_threshold(all_preds, all_masks)
    optimal_thresholds.append(optimal_threshold)
    
    # 使用最优阈值计算IoU和Dice
    iou_scores = []
    dice_scores = []
    for i in range(all_preds.size(0)):
        iou = compute_iou(all_preds[i,0], all_masks[i,0], threshold=optimal_threshold)
        dice = compute_dice(all_preds[i,0], all_masks[i,0], threshold=optimal_threshold)
        iou_scores.append(iou)
        dice_scores.append(dice)
    
    val_iou = np.mean(iou_scores)
    val_dice = np.mean(dice_scores)
    val_ious.append(val_iou)
    val_dices.append(val_dice)
    
    print(f'Epoch {epoch}/{epochs}')
    print(f'Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}')
    print(f'Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f} | Optimal Threshold: {optimal_threshold:.3f}')
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
    print('-' * 60)
    
    # 基于IoU调整学习率
    scheduler.step(val_iou)
    
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
        'best_val_iou': best_val_iou,
        'optimal_threshold': optimal_threshold
    }, checkpoint_path)
    
    # 每10个epoch绘制训练曲线
    if epoch % 10 == 0:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(val_ious, label='Val IoU')
        plt.plot(val_dices, label='Val Dice')
        plt.title('Metrics')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(optimal_thresholds, label='Optimal Threshold')
        plt.title('Optimal Threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'training_curves_epoch_{epoch}.png')
        plt.close()

print(f"Training completed. Best Val IoU: {best_val_iou:.4f}")
print(f"Final optimal threshold: {optimal_thresholds[-1]:.3f}") 