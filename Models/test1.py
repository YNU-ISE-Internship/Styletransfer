# -*- coding: utf-8 -*- 
# @Time : 1/4/24 17:05 
# @Author : ANG
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import os

# 参数设置
STYLE_IMAGE_PATHS = ['path/to/style1.jpg', 'path/to/style2.jpg', ...]  # 风格图像路径列表
IMAGE_SHAPE = (300, 450)  # 图像尺寸
FEATURES_SAVE_DIR = 'path/to/save/features'  # 保存特征的目录路径
STYLE_LAYERS = [0, 5, 10, 19, 28]  # 风格特征层

# 创建保存特征的目录
os.makedirs(FEATURES_SAVE_DIR, exist_ok=True)

# 预处理函数
def preprocess(img, image_shape):
    transform = transforms.Compose([
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# 提取特征函数
def extract_style_features(style_img_tensor, model, style_layers):
    features = []
    x = style_img_tensor
    for i, layer in enumerate(model.features):
        x = layer(x)
        if i in style_layers:
            features.append(x)
    return features

# 加载VGG模型
vgg = vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)

# 运行在GPU上，如果可用的话
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# 加载风格图像，提取特征并保存
for i, style_image_path in enumerate(STYLE_IMAGE_PATHS):
    style_img = Image.open(style_image_path)
    style_img_tensor = preprocess(style_img, IMAGE_SHAPE).to(device)
    style_features = extract_style_features(style_img_tensor, vgg, STYLE_LAYERS)
    torch.save(style_features, os.path.join(FEATURES_SAVE_DIR, f'style_features_{i}.pt'))
