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
import time
from datetime import datetime
from scipy import linalg
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d
from sklearn.metrics import precision_score, recall_score
from skimage.metrics import structural_similarity as ssim
import lpips
import torch.nn.functional as F


# 超参数设置
batch_size = 128
image_size = 64
nz = 256  # 生成器输入噪声向量的维度
ngf = 64  # 生成器特征图深度
ndf = 64  # 判别器特征图深度
num_epochs = 11
lr = 0.0001
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建输出目录
output_dir = r"C:\Users\G1581\Desktop\课设\img_pro\output1"
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.join(output_dir, "models")
os.makedirs(model_dir, exist_ok=True)
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)
samples_dir = os.path.join(output_dir, "samples")  
os.makedirs(samples_dir, exist_ok=True)
metrics_dir = os.path.join(output_dir, "metrics")
os.makedirs(metrics_dir, exist_ok=True)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 修正为3通道
])

dataset = datasets.ImageFolder(root=r"C:\Users\G1581\Desktop\课设\img_pro\database2", transform=transform)
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

# 权重初始化函数 (DCGAN标准初始化)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # 卷积层初始化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # 批量归一化层初始化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 初始化Inception模型用于FID计算
def get_inception_model():
    inception_model = inception_v3(pretrained=True)
    inception_model.fc = nn.Identity()  # 移除最后的全连接层
    inception_model = inception_model.to(device)
    inception_model.eval()
    return inception_model

# 计算FID分数
def calculate_fid(real_imgs, fake_imgs, model, batch_size=50):
    model.eval()
    # Resize到299x299
    def resize(imgs):
        return F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    # 提取真实图像特征
    real_features = []
    for i in range(0, len(real_imgs), batch_size):
        batch = real_imgs[i:i+batch_size].to(device)
        batch = resize(batch)  # 新增
        with torch.no_grad():
            features = model(batch)
        real_features.append(features.cpu())
    real_features = torch.cat(real_features, dim=0)
    # 提取生成图像特征
    fake_features = []
    for i in range(0, len(fake_imgs), batch_size):
        batch = fake_imgs[i:i+batch_size].to(device)
        batch = resize(batch)  # 新增
        with torch.no_grad():
            features = model(batch)
        fake_features.append(features.cpu())
    fake_features = torch.cat(fake_features, dim=0)
    
    # 计算特征统计量
    mu_real, sigma_real = real_features.mean(dim=0), torch_cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(dim=0), torch_cov(fake_features, rowvar=False)
    
    # 计算FID
    fid_value = calculate_frechet_distance(mu_real.numpy(), sigma_real.numpy(), 
                                          mu_fake.numpy(), sigma_fake.numpy())
    return fid_value

# 辅助函数：计算协方差矩阵
def torch_cov(m, rowvar=False):
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()

# 计算Fréchet距离
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # 乘积的平方根
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # 数值误差可能导致虚部很小
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

# 计算SSIM（结构相似性）
def calculate_ssim(real_imgs, fake_imgs, window_size=7):
    ssim_values = []
    for i in range(real_imgs.size(0)):
        real_img = real_imgs[i].permute(1, 2, 0).numpy()
        fake_img = fake_imgs[i].permute(1, 2, 0).numpy()
        
        # 将图像归一化到[0,1]
        real_img = (real_img + 1) / 2
        fake_img = (fake_img + 1) / 2
        
        # skimage >=0.19 需要指定 channel_axis
        ssim_val = ssim(
            real_img, fake_img,
            win_size=window_size,
            channel_axis=-1,
            data_range=1.0
        )
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

# 计算LPIPS（感知相似性）
def calculate_lpips(real_imgs, fake_imgs, loss_fn):
    lpips_values = []
    for i in range(real_imgs.size(0)):
        real_img = real_imgs[i].unsqueeze(0).to(device)
        fake_img = fake_imgs[i].unsqueeze(0).to(device)
        
        # 将图像归一化到[0,1]
        real_img = (real_img + 1) / 2
        fake_img = (fake_img + 1) / 2
        
        # 计算LPIPS
        lpips_val = loss_fn(real_img, fake_img).item()
        lpips_values.append(lpips_val)
    
    return np.mean(lpips_values)

# 初始化网络
netG = Generator().to(device)
netD = Discriminator().to(device)

# 应用权重初始化
netG.apply(weights_init)
netD.apply(weights_init)

# 初始化评估模型
inception_model = get_inception_model()
lpips_loss = lpips.LPIPS(net='alex').to(device)  # 使用AlexNet作为骨干网络

# 损失函数和优化器
criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

# 训练记录
training_history = {
    'D_loss': [],
    'G_loss': [],
    'fid': [],
    'ssim': [],
    'lpips': [],
    'diversity': [],  # 多样性分数
    'best_G_loss': float('inf'),
    'best_fid': float('inf'),
    'best_epoch': 0,
    'best_loss_epoch': 0,  # 新增：记录loss最优的epoch
    'epoch_times': []
}


# 固定噪声用于可视化
fixed_noise = torch.randn(25, nz, 1, 1, device=device)

# 准备真实图像样本用于评估
num_eval_samples = 1000
real_eval_samples = []
for i, data in enumerate(dataloader):
    real_eval_samples.append(data[0])
    if len(real_eval_samples) * batch_size >= num_eval_samples:
        break
real_eval_samples = torch.cat(real_eval_samples, dim=0)[:num_eval_samples].to(device)

# 训练GAN
start_time = time.time()
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    # 记录每个epoch的平均损失
    D_losses = []
    G_losses = []
    
    for i, data in enumerate(dataloader):
        real_images = data[0].to(device)
        current_batch_size = real_images.size(0)
 
        # 训练判别器
        netD.zero_grad()
        
        # 真实样本
        label_real = torch.ones(current_batch_size, 1, device=device)  # 修正标签形状
        output_real = netD(real_images).view(-1, 1)  # 确保形状匹配
        lossD_real = criterion(output_real, label_real)
 
        # 生成样本
        noise = torch.randn(current_batch_size, nz, 1, 1, device=device)
        fake_images = netG(noise)
        label_fake = torch.zeros(current_batch_size, 1, device=device)  # 修正标签形状
        output_fake = netD(fake_images.detach()).view(-1, 1)  # 确保形状匹配
        lossD_fake = criterion(output_fake, label_fake)
 
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()
 
        # 训练生成器
        netG.zero_grad()
        output_fake = netD(fake_images).view(-1)
        lossG = criterion(output_fake, torch.ones_like(output_fake))
        lossG.backward()
        optimizerG.step()
        
        # 记录损失
        D_losses.append(lossD.item())
        G_losses.append(lossG.item())
    
    # 计算epoch平均损失
    avg_D_loss = sum(D_losses) / len(D_losses)
    avg_G_loss = sum(G_losses) / len(G_losses)
    
    # 更新训练历史
    training_history['D_loss'].append(avg_D_loss)
    training_history['G_loss'].append(avg_G_loss)
    
    # 检查是否为最佳生成器loss模型
    if avg_G_loss < training_history.get('best_G_loss', float('inf')):
        training_history['best_G_loss'] = avg_G_loss
        training_history['best_loss_epoch'] = epoch + 1
        torch.save(netG.state_dict(), os.path.join(model_dir, 'generator_best_loss.pth'))
        torch.save(netD.state_dict(), os.path.join(model_dir, 'discriminator_best_loss.pth'))

    
    # 计算评估指标（每10个epoch或最后）
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        netG.eval()
        with torch.no_grad():
            # 生成评估样本
            eval_noise = torch.randn(num_eval_samples, nz, 1, 1, device=device)
            fake_eval_samples = netG(eval_noise)
            
            # 计算FID
            fid_value = calculate_fid(real_eval_samples, fake_eval_samples, inception_model)
            training_history['fid'].append(fid_value)
            
            # 计算SSIM（取前100个样本）
            ssim_value = calculate_ssim(real_eval_samples[:100].cpu(), fake_eval_samples[:100].cpu())
            training_history['ssim'].append(ssim_value)
            
            # 计算LPIPS（取前100个样本）
            lpips_value = calculate_lpips(real_eval_samples[:100].cpu(), fake_eval_samples[:100].cpu(), lpips_loss)
            training_history['lpips'].append(lpips_value)
            
            # 计算多样性（不同噪声样本之间的LPIPS）
            diversity_noise1 = torch.randn(100, nz, 1, 1, device=device)
            diversity_noise2 = torch.randn(100, nz, 1, 1, device=device)
            fake_diversity1 = netG(diversity_noise1)
            fake_diversity2 = netG(diversity_noise2)
            diversity_value = calculate_lpips(fake_diversity1.cpu(), fake_diversity2.cpu(), lpips_loss)
            training_history['diversity'].append(diversity_value)
            
            # 保存样本图像
            fake = netG(fixed_noise).detach().cpu()
            grid = vutils.make_grid(fake, nrow=5, normalize=True)
            plt.figure(figsize=(10, 10))
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.axis('off')
            plt.title(f'Epoch {epoch+1}')
            plt.savefig(os.path.join(samples_dir, f'fake_samples_epoch_{epoch+1}.png'), bbox_inches='tight')
            plt.close()
            
            # 保存评估指标
            np.save(os.path.join(metrics_dir, f'metrics_epoch_{epoch+1}.npy'), {
                'fid': fid_value,
                'ssim': ssim_value,
                'lpips': lpips_value,
                'diversity': diversity_value
            })
        
        netG.train()
        
        # 检查是否为最佳模型（基于FID）
        if fid_value < training_history.get('best_fid', float('inf')):
            training_history['best_fid'] = fid_value
            training_history['best_epoch'] = epoch + 1
            torch.save(netG.state_dict(), os.path.join(model_dir, 'generator_best.pth'))
            torch.save(netD.state_dict(), os.path.join(model_dir, 'discriminator_best.pth'))
    
    # 计算epoch时间
    epoch_time = time.time() - epoch_start_time
    training_history['epoch_times'].append(epoch_time)
    
    # 打印进度
    epoch_time_str = time.strftime("%M:%S", time.gmtime(epoch_time))
    total_time = time.time() - start_time
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    
    # 添加评估指标显示
    metrics_str = ""
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        metrics_str = (f" | FID: {training_history['fid'][-1]:.2f} "
                      f"| SSIM: {training_history['ssim'][-1]:.4f} "
                      f"| LPIPS: {training_history['lpips'][-1]:.4f} "
                      f"| Diversity: {training_history['diversity'][-1]:.4f}")
    
    print(f'Epoch [{epoch + 1}/{num_epochs}] | '
          f'D_loss: {avg_D_loss:.4f} | G_loss: {avg_G_loss:.4f} '
          f'{metrics_str}'
          f' | Epoch Time: {epoch_time_str} | Total Time: {total_time_str}')

# 保存最终模型
torch.save(netG.state_dict(), os.path.join(model_dir, 'generator_final.pth'))
torch.save(netD.state_dict(), os.path.join(model_dir, 'discriminator_final.pth'))

# 绘制判别器损失曲线
plt.figure(figsize=(8, 5))
plt.plot(training_history['D_loss'], label='Discriminator Loss', color='tab:blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Discriminator Training Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'discriminator_loss_curve.png'))
plt.close()

# 绘制生成器损失曲线
plt.figure(figsize=(8, 5))
plt.plot(training_history['G_loss'], label='Generator Loss', color='tab:orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator Training Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'generator_loss_curve.png'))
plt.close()

# 绘制每个epoch训练时间曲线
plt.figure(figsize=(8, 5))
plt.plot(training_history['epoch_times'], color='tab:green')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.title('Epoch Training Time')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'epoch_training_time.png'))
plt.close()

# 绘制评估指标曲线
plt.figure(figsize=(12, 8))

# 统一x轴
metric_epochs = np.linspace(0, num_epochs-1, len(training_history['fid']), dtype=int)

# FID曲线
plt.subplot(2, 2, 1)
plt.plot(metric_epochs, training_history['fid'], 'o-', color='tab:red')
plt.xlabel('Epoch')
plt.ylabel('FID')
plt.title('Fréchet Inception Distance (FID)')
plt.grid(True)

# SSIM曲线
plt.subplot(2, 2, 2)
plt.plot(metric_epochs, training_history['ssim'], 'o-', color='tab:purple')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.title('Structural Similarity Index (SSIM)')
plt.grid(True)

# LPIPS曲线
plt.subplot(2, 2, 3)
plt.plot(metric_epochs, training_history['lpips'], 'o-', color='tab:brown')
plt.xlabel('Epoch')
plt.ylabel('LPIPS')
plt.title('Learned Perceptual Image Patch Similarity (LPIPS)')
plt.grid(True)

# 多样性曲线
plt.subplot(2, 2, 4)
plt.plot(metric_epochs, training_history['diversity'], 'o-', color='tab:cyan')
plt.xlabel('Epoch')
plt.ylabel('Diversity')
plt.title('Generator Diversity Score')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'evaluation_metrics.png'))
plt.close()

with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
    f.write(f"Training Summary\n")
    f.write(f"================\n")
    f.write(f"Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Training Time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))}\n")
    f.write(f"Best Epoch (FID): {training_history['best_epoch']} (G_loss={training_history['G_loss'][training_history['best_epoch']-1]:.4f}, FID={training_history['best_fid']:.2f})\n")
    f.write(f"Best Epoch (G_loss): {training_history['best_loss_epoch']} (G_loss={training_history['best_G_loss']:.4f})\n")  # 新增loss最优模型记录

    f.write(f"\nHyperparameters:\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Image Size: {image_size}\n")
    f.write(f"Latent Size (nz): {nz}\n")
    f.write(f"Generator Features: {ngf}\n")
    f.write(f"Discriminator Features: {ndf}\n")
    f.write(f"Epochs: {num_epochs}\n")
    f.write(f"Learning Rate: {lr}\n")
    f.write(f"Beta1: {beta1}\n")

    # 添加评估指标摘要
    f.write(f"\nEvaluation Metrics Summary:\n")
    min_fid_idx = np.argmin(training_history['fid'])
    max_ssim_idx = np.argmax(training_history['ssim'])
    min_lpips_idx = np.argmin(training_history['lpips'])
    max_div_idx = np.argmax(training_history['diversity'])

    f.write(f"Best FID: {training_history['fid'][min_fid_idx]:.2f} at epoch {metric_epochs[min_fid_idx]+1}\n")
    f.write(f"Best SSIM: {training_history['ssim'][max_ssim_idx]:.4f} at epoch {metric_epochs[max_ssim_idx]+1}\n")
    f.write(f"Best LPIPS: {training_history['lpips'][min_lpips_idx]:.4f} at epoch {metric_epochs[min_lpips_idx]+1}\n")
    f.write(f"Best Diversity: {training_history['diversity'][max_div_idx]:.4f} at epoch {metric_epochs[max_div_idx]+1}\n")

print("Training completed!")
print(f"Best generator model (FID) saved at epoch {training_history['best_epoch']} with FID {training_history['best_fid']:.2f}")
print(f"Best generator model (G_loss) saved at epoch {training_history['best_loss_epoch']} with G_loss {training_history['best_G_loss']:.4f}")
print(f"Models and plots saved to: {output_dir}")