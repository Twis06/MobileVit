import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from seg_model import SegMobileViT_DeepLabV3
from eval import compute_iou, compute_dice
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_predictions(model_path, images_dir, masks_dir, num_samples=20):
    """分析模型预测结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = SegMobileViT_DeepLabV3(backbone_name='mobilevit_xs', num_classes=1, image_size=(256, 256))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 加载数据
    dataset = SegmentationDataset(images_dir, masks_dir, image_size=(256, 256), is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 收集预测结果
    all_ious = []
    all_dices = []
    all_thresholds = np.arange(0.1, 0.9, 0.05)
    threshold_ious = {t: [] for t in all_thresholds}
    
    print("分析预测结果...")
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            # 计算不同阈值下的IoU
            for threshold in all_thresholds:
                iou = compute_iou(preds[0,0], masks[0,0], threshold=threshold)
                threshold_ious[threshold].append(iou)
            
            # 使用0.5阈值计算基础指标
            iou = compute_iou(preds[0,0], masks[0,0], threshold=0.5)
            dice = compute_dice(preds[0,0], masks[0,0], threshold=0.5)
            all_ious.append(iou)
            all_dices.append(dice)
            
            print(f"Sample {i+1}: IoU={iou:.4f}, Dice={dice:.4f}")
    
    # 分析阈值影响
    mean_ious_by_threshold = {t: np.mean(threshold_ious[t]) for t in all_thresholds}
    best_threshold = max(mean_ious_by_threshold, key=mean_ious_by_threshold.get)
    best_iou = mean_ious_by_threshold[best_threshold]
    
    print(f"\n阈值分析:")
    print(f"最佳阈值: {best_threshold:.3f}")
    print(f"最佳IoU: {best_iou:.4f}")
    print(f"0.5阈值IoU: {np.mean(all_ious):.4f}")
    
    # 绘制阈值-IoU曲线
    plt.figure(figsize=(10, 6))
    thresholds = list(mean_ious_by_threshold.keys())
    ious = list(mean_ious_by_threshold.values())
    plt.plot(thresholds, ious, 'b-o')
    plt.axvline(x=0.5, color='r', linestyle='--', label='Default threshold (0.5)')
    plt.axvline(x=best_threshold, color='g', linestyle='--', label=f'Best threshold ({best_threshold:.3f})')
    plt.xlabel('Threshold')
    plt.ylabel('Mean IoU')
    plt.title('IoU vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig('threshold_analysis.png')
    plt.close()
    
    return best_threshold, best_iou

def analyze_class_balance(images_dir, masks_dir):
    """分析类别平衡"""
    dataset = SegmentationDataset(images_dir, masks_dir, image_size=(256, 256), is_train=False)
    
    foreground_pixels = 0
    background_pixels = 0
    total_samples = 0
    
    print("分析类别平衡...")
    for i in range(min(100, len(dataset))):  # 分析前100个样本
        _, mask = dataset[i]
        foreground_pixels += mask.sum().item()
        background_pixels += (1 - mask).sum().item()
        total_samples += 1
    
    total_pixels = foreground_pixels + background_pixels
    foreground_ratio = foreground_pixels / total_pixels
    background_ratio = background_pixels / total_pixels
    
    print(f"前景像素比例: {foreground_ratio:.4f}")
    print(f"背景像素比例: {background_ratio:.4f}")
    print(f"类别不平衡比例: {background_ratio/foreground_ratio:.2f}:1")
    
    return foreground_ratio, background_ratio

def analyze_loss_landscape(model_path, images_dir, masks_dir):
    """分析损失景观"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型和数据
    model = SegMobileViT_DeepLabV3(backbone_name='mobilevit_xs', num_classes=1, image_size=(256, 256))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    dataset = SegmentationDataset(images_dir, masks_dir, image_size=(256, 256), is_train=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # 获取一些样本
    images, masks = next(iter(loader))
    images = images.to(device)
    masks = masks.to(device)
    
    # 计算不同阈值下的损失
    from losses import TverskyLoss, DiceLoss, FocalLoss
    tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
    dice_loss = DiceLoss()
    focal_loss = FocalLoss()
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    losses = {'tversky': [], 'dice': [], 'focal': []}
    
    print("分析损失景观...")
    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        
        for threshold in thresholds:
            # 二值化预测
            binary_preds = (preds > threshold).float()
            
            # 计算各种损失
            t_loss = tversky_loss(torch.logit(binary_preds + 1e-8), masks).item()
            d_loss = dice_loss(binary_preds, masks).item()
            f_loss = focal_loss(torch.logit(binary_preds + 1e-8), masks).item()
            
            losses['tversky'].append(t_loss)
            losses['dice'].append(d_loss)
            losses['focal'].append(f_loss)
    
    # 绘制损失景观
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(thresholds, losses['tversky'])
    plt.title('Tversky Loss vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(thresholds, losses['dice'])
    plt.title('Dice Loss vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(thresholds, losses['focal'])
    plt.title('Focal Loss vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('loss_landscape.png')
    plt.close()

def main():
    model_path = 'best_model.pth'  # 或 'best_model_improved.pth'
    images_dir = 'Dataset/test/images'
    masks_dir = 'Dataset/test/masks'
    
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在")
        return
    
    print("=== 训练问题诊断分析 ===\n")
    
    # 1. 分析预测结果和阈值
    print("1. 预测结果分析")
    best_threshold, best_iou = analyze_predictions(model_path, images_dir, masks_dir)
    
    # 2. 分析类别平衡
    print("\n2. 类别平衡分析")
    fg_ratio, bg_ratio = analyze_class_balance(images_dir, masks_dir)
    
    # 3. 分析损失景观
    print("\n3. 损失景观分析")
    analyze_loss_landscape(model_path, images_dir, masks_dir)
    
    # 4. 给出建议
    print("\n=== 诊断结果和建议 ===")
    
    if best_threshold != 0.5:
        print(f"✓ 发现最佳阈值 ({best_threshold:.3f}) 与默认阈值 (0.5) 不同")
        print("  建议: 使用动态阈值优化或调整损失函数")
    
    if bg_ratio / fg_ratio > 10:
        print(f"✓ 发现严重的类别不平衡 ({bg_ratio/fg_ratio:.1f}:1)")
        print("  建议: 使用Focal Loss或调整Tversky Loss参数")
    
    if best_iou < 0.5:
        print(f"✓ IoU较低 ({best_iou:.4f})")
        print("  建议: 增加数据增强、调整学习率、尝试更强的backbone")
    
    print("\n建议的改进措施:")
    print("1. 使用改进的训练脚本 (train_improved.py)")
    print("2. 使用更强的数据增强 (dataset_improved.py)")
    print("3. 尝试组合损失函数")
    print("4. 基于IoU调整学习率")
    print("5. 使用动态阈值优化")

if __name__ == '__main__':
    main() 