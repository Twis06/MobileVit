import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import SegmentationDataset
from seg_model import SegMobileViT
from losses import DiceLoss

# 配置
images_dir = 'Dataset/train/images'
masks_dir = 'Dataset/train/masks'
image_size = (256, 256)
batch_size = 8
epochs = 30
lr = 1e-3
num_workers = 4
save_path = 'best_model.pth'

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 数据集
train_dataset = SegmentationDataset(images_dir, masks_dir, image_size=image_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# 模型
model = SegMobileViT(backbone_name='mobilevit_xxs', num_classes=1, image_size=image_size)
model = model.to(device)

# 损失和优化器
criterion_bce = nn.BCELoss()
criterion_dice = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

best_loss = float('inf')
for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # [B, 1, H, W]
        loss_bce = criterion_bce(outputs, masks)
        loss_dice = criterion_dice(outputs, masks)
        loss = 0.5 * loss_bce + 0.5 * loss_dice
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}')
    # 保存最优模型
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), save_path)
        print(f'Best model saved at epoch {epoch} with loss {best_loss:.4f}') 