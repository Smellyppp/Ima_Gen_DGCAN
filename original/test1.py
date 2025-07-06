import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
 
# 超参数设置
batch_size = 128
image_size = 64
nz = 256  # 生成器输入噪声向量的维度
ngf = 64  # 生成器特征图深度
ndf = 64  # 判别器特征图深度
num_epochs = 300
lr = 0.0001
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# 数据预处理
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
 
dataset = datasets.ImageFolder(root=r"C:\Users\G1581\Desktop\课设\img_pro\database3", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
 
# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
 
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
 
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
 
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
 
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
 
    def forward(self, x):
        return self.model(x)
 
 
# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
 
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
 
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
 
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
 
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        return self.model(x)
 
 
# 初始化网络
netG = Generator().to(device)
netD = Discriminator().to(device)
 
# 损失函数和优化器
criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
 
# 训练GAN
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        real_images = data[0].to(device)
        batch_size = real_images.size(0)
 
        # 训练判别器
        netD.zero_grad()
        output_real = netD(real_images).view(-1)
        lossD_real = criterion(output_real, torch.ones_like(output_real))
 
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach()).view(-1)
        lossD_fake = criterion(output_fake, torch.zeros_like(output_fake))
 
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()
 
        # 训练生成器
        netG.zero_grad()
        output_fake = netD(fake_images).view(-1)
        lossG = criterion(output_fake, torch.ones_like(output_fake))
        lossG.backward()
        optimizerG.step()
 
    print(f'Epoch [{epoch + 1}/{num_epochs}] Loss_D: {lossD.item():.4f}, Loss_G: {lossG.item():.4f}')
 
    if epoch % 10 == 0:
        vutils.save_image(fake_images.data[:25], f"C:/Users/G1581/Desktop/课设/img_pro/output/fake_samples_epoch_{epoch}.png", normalize=True)