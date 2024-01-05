import torch
from torch import nn
import torchvision
from PIL import Image

# 参数设置

CONTENT_IMAGE_PATH = '../Images/thestarrynight.jpeg'  # 内容图像路径
STYLE_IMAGE_PATH = '../Images/Ukiyo-e.jpeg'  # 风格图像路径
# 图像尺寸：用于调整输入图像的大小。这个元组中的两个整数表示图像的高度和宽度。
IMAGE_SHAPE = (300, 450)

# RGB均值：用于图像归一化处理的均值，对应于ImageNet数据集的均值。
# 这个三元素向量的每个值都在0到1之间，表示R、G、B三个通道的均值。
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406])

# RGB标准差：用于图像归一化处理的标准差，对应于ImageNet数据集的标准差。
# 这个三元素向量的每个值都在0到1之间，表示R、G、B三个通道的标准差。
RGB_STD = torch.tensor([0.229, 0.224, 0.225])

# 风格层：这是VGG网络中将用来提取图像风格特征的层的索引列表。
# 取值范围通常是0到网络层的总数，这个例子中使用VGG19网络。
STYLE_LAYERS = [0, 5, 10, 19, 28]

# 内容层：这是VGG网络中将用来提取图像内容特征的层的索引。
# 取值范围通常是0到网络层的总数，这个例子中使用VGG19网络。
CONTENT_LAYERS = [25]

# 内容权重：在总损失函数中内容损失的权重。
# 这个值可以根据需要调整，以平衡内容和风格之间的重要性。
# 取值范围通常从0到无穷大，根据具体问题调整。
CONTENT_WEIGHT = 1

# 风格权重：在总损失函数中风格损失的权重。
# 这个值通常比内容权重大，因为风格特征比内容特征更抽象、更难以捕捉。
# 取值范围通常从0到无穷大，根据具体问题调整。
STYLE_WEIGHT = 1e3

# 总变差权重：在总损失函数中总变差损失（图像平滑性）的权重。
# 调整这个值可以帮助减少生成图像的噪点。
# 取值范围通常从0到无穷大，根据具体问题调整。
TV_WEIGHT = 10

# 学习率：用于优化器的学习率。
# 这个值决定了每次迭代时参数更新的步长。
# 取值范围通常从0到1，较小的值可能导致学习过程缓慢，较大的值可能导致学习过程不稳定。
LEARNING_RATE = 0.3

# 迭代次数：训练过程中迭代的总次数。
# 取值范围通常是一个正整数，具体取决于训练数据的大小和复杂性。
NUM_EPOCHS = 500

# 学习率衰减周期：学习率衰减的周期（以迭代次数为单位）。
# 每过这么多次迭代，学习率会按一定比例减少。
# 取值范围通常是一个正整数，根据训练的需要调整。
LR_DECAY_EPOCH = 50



# 图像预处理
def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)])
    return transforms(img).unsqueeze(0)


# 图像后处理
def postprocess(img):
    img = img[0].to(RGB_STD.device)
    img = torch.clamp(img.permute(1, 2, 0) * RGB_STD + RGB_MEAN, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))


# 加载VGG模型
pretrained_net = torchvision.models.vgg19(pretrained=True)
net = nn.Sequential(*[pretrained_net.features[i] for i in range(max(CONTENT_LAYERS + STYLE_LAYERS) + 1)])


# 提取特征
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


# 计算内容损失
def content_loss(Y_hat, Y):
    return torch.square(Y_hat - Y.detach()).mean()


# 计算格拉姆矩阵
def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)


# 计算风格损失
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()


# 计算总变差损失
def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


# 计算损失
def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    contents_l = [content_loss(Y_hat, Y) * CONTENT_WEIGHT for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * STYLE_WEIGHT for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * TV_WEIGHT
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


# 合成图像的类
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape):
        super(SynthesizedImage, self).__init__()
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


# 训练
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    # 初始化合成图像，并复制内容图像的数据
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    # 创建优化器
    trainer = torch.optim.Adam(gen_img.parameters(), lr)
    # 计算风格图像的格拉姆矩阵
    styles_Y_gram = [gram(Y) for Y in styles_Y]

    # 训练过程
    for epoch in range(num_epochs):
        # 清除梯度
        trainer.zero_grad()
        # 提取生成图像的内容和风格特征
        contents_Y_hat, styles_Y_hat = extract_features(gen_img(), CONTENT_LAYERS, STYLE_LAYERS)
        # 计算内容损失、风格损失和总变差损失
        contents_l, styles_l, tv_l, l = compute_loss(gen_img(), contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        # 反向传播
        l.backward()
        # 更新权重
        trainer.step()
        # 学习率衰减
        if (epoch + 1) % lr_decay_epoch == 0:
            trainer.param_groups[0]['lr'] *= 0.8
        # 可选：打印训练进度

    # 返回训练后的图像
    return gen_img()


# 主程序
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
net = net.to(device)

# 加载和处理图像
content_img = Image.open(CONTENT_IMAGE_PATH)
style_img = Image.open(STYLE_IMAGE_PATH)
content_X, contents_Y = preprocess(content_img, IMAGE_SHAPE).to(device), \
extract_features(preprocess(content_img, IMAGE_SHAPE).to(device), CONTENT_LAYERS, STYLE_LAYERS)[0]
style_X, styles_Y = preprocess(style_img, IMAGE_SHAPE).to(device), \
extract_features(preprocess(style_img, IMAGE_SHAPE).to(device), CONTENT_LAYERS, STYLE_LAYERS)[1]

# 训练并生成图像
output = train(content_X, contents_Y, styles_Y, device, LEARNING_RATE, NUM_EPOCHS, LR_DECAY_EPOCH)
# Assuming `output` is the tensor obtained after the style transfer process.
output_image = postprocess(output)  # Convert the tensor to an image
output_image_path = '../Images/generated_image.jpg'  # Specify the local path to save the image
output_image.save(output_image_path)  # Save the image
