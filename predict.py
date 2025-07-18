import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import sys
import os
from seg_model import SegMobileViT_DeepLabV3

# 用法: python predict.py input.jpg best_model.pth output_mask.png
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str, help='输入图片路径')
parser.add_argument('model_path', type=str, help='模型权重路径')
parser.add_argument('output_path', type=str, help='输出mask路径')
parser.add_argument('--backbone', type=str, default='mobilevit_xs')
parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256])
parser.add_argument('--threshold', type=float, default=0.5)
args = parser.parse_args()

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = SegMobileViT_DeepLabV3(backbone_name=args.backbone, num_classes=1, image_size=tuple(args.image_size))
model.load_state_dict(torch.load(args.model_path, map_location=device))
model = model.to(device)
model.eval()

# 预处理
transform = T.Compose([
    T.Resize(tuple(args.image_size)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 读取图片
image = Image.open(args.image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# 推理
with torch.no_grad():
    output = model(input_tensor)
    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    mask = (mask > args.threshold).astype(np.uint8) * 255

# 保存mask
mask_img = Image.fromarray(mask)
mask_img.save(args.output_path)
print(f'Mask saved to {args.output_path}') 