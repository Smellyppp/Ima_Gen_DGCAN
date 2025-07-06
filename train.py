import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import time
from datetime import datetime
import lpips
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from evaluate import Generator, Discriminator, weights_init

def train_gan(config):
    # 读取配置参数
    batch_size = config['batch_size']              # 批量大小
    image_size = config['image_size']              # 图像尺寸
    nz = config['nz']                              # 噪声向量维度
    ngf = config['ngf']                            # 生成器特征图数
    ndf = config['ndf']                            # 判别器特征图数
    num_epochs = config['num_epochs']              # 训练轮数
    lr = config['lr']                              # 学习率
    beta1 = config['beta1']                        # Adam优化器beta1参数
    device = config['device']                      # 设备（CPU或GPU）
    output_dir = config['output_dir']              # 输出目录
    model_dir = os.path.join(output_dir, "models") # 模型保存目录
    os.makedirs(model_dir, exist_ok=True)          # 创建模型目录

    # 数据预处理与加载
    transform = transforms.Compose([
        transforms.Resize(image_size),                         # 调整图片大小
        transforms.CenterCrop(image_size),                     # 居中裁剪
        transforms.ToTensor(),                                 # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化到[-1, 1]
    ])
    dataset = datasets.ImageFolder(root=config['data_root'], transform=transform) # 加载数据集
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)         # 数据加载器

    # 初始化生成器和判别器
    netG = Generator(nz, ngf).to(device)
    netD = Discriminator(ndf).to(device)
    netG.apply(weights_init) # 权重初始化
    netD.apply(weights_init)

    criterion = nn.BCELoss() # 二分类交叉熵损失
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999)) # 生成器优化器
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999)) # 判别器优化器

    # 训练历史记录
    training_history = {
        'D_loss': [],            # 判别器损失
        'G_loss': [],            # 生成器损失
        'best_G_loss': float('inf'), # 最佳生成器损失
        'best_loss_epoch': 0,        # 最佳损失所在轮数
        'epoch_times': []            # 每轮耗时
    }

    fixed_noise = torch.randn(25, nz, 1, 1, device=device) # 固定噪声用于可视化
    start_time = time.time()

    # 训练主循环
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        D_losses, G_losses = [], []
        for i, data in enumerate(dataloader):
            real_images = data[0].to(device)                  # 真实图片
            current_batch_size = real_images.size(0)          # 当前批量大小

            # 判别器训练
            netD.zero_grad()
            label_real = torch.ones(current_batch_size, 1, device=device) # 真实标签
            output_real = netD(real_images).view(-1, 1)                  # 判别器对真实图片的输出
            lossD_real = criterion(output_real, label_real)              # 判别器在真实图片上的损失

            noise = torch.randn(current_batch_size, nz, 1, 1, device=device) # 随机噪声
            fake_images = netG(noise)                                       # 生成假图片
            label_fake = torch.zeros(current_batch_size, 1, device=device)  # 假标签
            output_fake = netD(fake_images.detach()).view(-1, 1)            # 判别器对假图片的输出
            lossD_fake = criterion(output_fake, label_fake)                 # 判别器在假图片上的损失

            lossD = lossD_real + lossD_fake # 判别器总损失
            lossD.backward()                # 反向传播
            optimizerD.step()               # 优化判别器参数

            # 生成器训练
            netG.zero_grad()
            output_fake = netD(fake_images).view(-1)                  # 判别器对假图片的输出
            lossG = criterion(output_fake, torch.ones_like(output_fake)) # 生成器希望判别器输出都为真
            lossG.backward()                                         # 反向传播
            optimizerG.step()                                        # 优化生成器参数

            D_losses.append(lossD.item()) # 记录判别器损失
            G_losses.append(lossG.item()) # 记录生成器损失

        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            grid = vutils.make_grid(fake, nrow=5, normalize=True)
            sample_dir = os.path.join(output_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            plt.figure(figsize=(6, 6))
            plt.axis("off")
            plt.title(f"Samples at Epoch {epoch+1}")
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.savefig(os.path.join(sample_dir, f"sample_epoch_{epoch+1}.png"), bbox_inches='tight')
            plt.close()

        avg_D_loss = sum(D_losses) / len(D_losses) # 判别器平均损失
        avg_G_loss = sum(G_losses) / len(G_losses) # 生成器平均损失
        training_history['D_loss'].append(avg_D_loss)
        training_history['G_loss'].append(avg_G_loss)

        # 保存最佳模型
        if avg_G_loss < training_history['best_G_loss']:
            training_history['best_G_loss'] = avg_G_loss
            training_history['best_loss_epoch'] = epoch + 1
            torch.save(netG.state_dict(), os.path.join(model_dir, 'generator_best_loss.pth'))
            torch.save(netD.state_dict(), os.path.join(model_dir, 'discriminator_best_loss.pth'))

        epoch_time = time.time() - epoch_start_time
        training_history['epoch_times'].append(epoch_time)
        print(f'Epoch [{epoch + 1}/{num_epochs}] | D_loss: {avg_D_loss:.4f} | G_loss: {avg_G_loss:.4f} | Time: {epoch_time:.1f}s')

    # 保存最终模型
    torch.save(netG.state_dict(), os.path.join(model_dir, 'generator_final.pth'))
    torch.save(netD.state_dict(), os.path.join(model_dir, 'discriminator_final.pth'))

    # 保存训练历史
    np.save(os.path.join(output_dir, 'training_history.npy'), training_history)
    print("Training finished.")
    
    # 绘制损失曲线并保存
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure()
    plt.plot(training_history['D_loss'], label='Discriminator Loss')
    plt.plot(training_history['G_loss'], label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.savefig(os.path.join(plot_dir, 'loss_curve.png'))
    plt.close()

    # 只保存最终结果到txt
    history_txt = os.path.join(output_dir, 'train_and_eval_log.txt')
    with open(history_txt, 'w', encoding='utf-8') as f:
        f.write(f"最佳生成器损失: {training_history['best_G_loss']:.4f} (第{training_history['best_loss_epoch']}轮)\n")

    print(f"训练结果已保存到 {history_txt}，损失曲线已保存到 {plot_dir}")

if __name__ == "__main__":
    from config import get_config
    train_gan(get_config())