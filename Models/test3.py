# -*- coding: utf-8 -*- 
# @Time : 1/4/24 17:05 
# @Author : ANG
import torch
from torchvision.models import vgg19

# 加载VGG模型，确保只在评估模式下使用，不会计算梯度
vgg = vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)
vgg.eval()  # Set the VGG network to evaluation mode

# 假设你的特征被保存在下面这个目录
FEATURES_SAVE_DIR = 'path/to/save/features'

# 加载保存的风格特征
def load_style_features(feature_dir, style_idx):
    feature_path = os.path.join(feature_dir, f'style_features_{style_idx}.pt')
    style_features = torch.load(feature_path)
    return style_features

# 使用加载的特征进行风格迁移的函数
def style_transfer_with_loaded_features(content_image_tensor, style_features, vgg, num_steps=300):
    # 这里需要实现具体的风格迁移代码，包括定义合成图像，计算损失和优化过程
    # ...
    pass

# 示例：将风格迁移应用到新的内容图像上
# 加载内容图像并预处理
content_image = Image.open('path/to/your/content/image.jpg')
content_image_tensor = preprocess(content_image, IMAGE_SHAPE).to(device)

# 加载之前保存的风格特征
style_idx = 0  # 假设你想要使用的风格索引为0
loaded_style_features = load_style_features(FEATURES_SAVE_DIR, style_idx)

# 执行风格迁移
output_image_tensor = style_transfer_with_loaded_features(content_image_tensor, loaded_style_features, vgg)

# 后处理并保存/显示最终的风格迁移图像
# ...
