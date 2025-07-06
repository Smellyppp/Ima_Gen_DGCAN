import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3
import lpips
from skimage.metrics import structural_similarity as ssim

# ====== 网络结构和工具 ======
class Generator(nn.Module):
    def __init__(self, nz=256, ngf=64):
        super(Generator, self).__init__()
        # 生成器结构，输入为噪声向量，输出为3通道图像
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

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        # 判别器结构，输入为3通道图像，输出为概率
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

def weights_init(m):
    # 权重初始化
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_inception_model(device):
    # 获取Inception v3模型用于FID特征提取
    inception_model = inception_v3(pretrained=True)
    inception_model.fc = nn.Identity()
    inception_model = inception_model.to(device)
    inception_model.eval()
    return inception_model

def torch_cov(m, rowvar=False):
    # 计算协方差矩阵
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

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # 计算Frechet距离（FID核心公式）
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid(real_imgs, fake_imgs, model, device, batch_size=50):
    # 计算FID分数
    model.eval()
    def resize(imgs):
        return F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    real_features, fake_features = [], []
    for i in range(0, len(real_imgs), batch_size):
        batch = real_imgs[i:i+batch_size].to(device)
        batch = resize(batch)
        with torch.no_grad():
            features = model(batch)
        real_features.append(features.cpu())
    for i in range(0, len(fake_imgs), batch_size):
        batch = fake_imgs[i:i+batch_size].to(device)
        batch = resize(batch)
        with torch.no_grad():
            features = model(batch)
        fake_features.append(features.cpu())
    real_features = torch.cat(real_features, dim=0)
    fake_features = torch.cat(fake_features, dim=0)
    mu_real, sigma_real = real_features.mean(dim=0), torch_cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(dim=0), torch_cov(fake_features, rowvar=False)
    fid_value = calculate_frechet_distance(mu_real.numpy(), sigma_real.numpy(), mu_fake.numpy(), sigma_fake.numpy())
    return fid_value

def calculate_ssim(real_imgs, fake_imgs, window_size=7):
    # 计算SSIM分数
    ssim_values = []
    for i in range(real_imgs.size(0)):
        real_img = real_imgs[i].permute(1, 2, 0).numpy()
        fake_img = fake_imgs[i].permute(1, 2, 0).numpy()
        real_img = (real_img + 1) / 2
        fake_img = (fake_img + 1) / 2
        ssim_val = ssim(real_img, fake_img, win_size=window_size, channel_axis=-1, data_range=1.0)
        ssim_values.append(ssim_val)
    return np.mean(ssim_values)

def calculate_lpips(real_imgs, fake_imgs, loss_fn, device):
    # 计算LPIPS分数
    lpips_values = []
    for i in range(real_imgs.size(0)):
        real_img = real_imgs[i].unsqueeze(0).to(device)
        fake_img = fake_imgs[i].unsqueeze(0).to(device)
        real_img = (real_img + 1) / 2
        fake_img = (fake_img + 1) / 2
        lpips_val = loss_fn(real_img, fake_img).item()
        lpips_values.append(lpips_val)
    return np.mean(lpips_values)

def evaluate_gan(config):
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt

    # 读取配置参数
    batch_size = config['batch_size']
    nz = config['nz']
    num_eval_samples = config['num_eval_samples']
    device = config['device']
    output_dir = config['output_dir']
    model_dir = os.path.join(output_dir, "models")
    plot_dir = os.path.join(output_dir, "plots")
    samples_dir = os.path.join(output_dir, "samples")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # 加载生成器模型（默认加载最终模型）
    netG = Generator(nz, config['ngf']).to(device)
    netG.load_state_dict(torch.load(os.path.join(model_dir, 'generator_final.pth'), map_location=device))
    netG.eval()

    # 加载Inception模型和LPIPS损失
    inception_model = get_inception_model(device)
    lpips_loss = lpips.LPIPS(net='alex').to(device)

    # 加载真实图片数据
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.CenterCrop(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(root=config['data_root'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    real_eval_samples = []
    for i, data in enumerate(dataloader):
        real_eval_samples.append(data[0])
        if len(real_eval_samples) * batch_size >= num_eval_samples:
            break
    real_eval_samples = torch.cat(real_eval_samples, dim=0)[:num_eval_samples].to(device)

    # 生成假图片
    eval_noise = torch.randn(num_eval_samples, nz, 1, 1, device=device)
    with torch.no_grad():
        fake_eval_samples = netG(eval_noise)

    # 计算FID
    fid_value = calculate_fid(real_eval_samples, fake_eval_samples, inception_model, device)
    # 计算SSIM
    ssim_value = calculate_ssim(real_eval_samples[:100].cpu(), fake_eval_samples[:100].cpu())
    # 计算LPIPS
    lpips_value = calculate_lpips(real_eval_samples[:100].cpu(), fake_eval_samples[:100].cpu(), lpips_loss, device)
    # 计算多样性（LPIPS）
    diversity_noise1 = torch.randn(100, nz, 1, 1, device=device)
    diversity_noise2 = torch.randn(100, nz, 1, 1, device=device)
    fake_diversity1 = netG(diversity_noise1)
    fake_diversity2 = netG(diversity_noise2)
    diversity_value = calculate_lpips(fake_diversity1.cpu(), fake_diversity2.cpu(), lpips_loss, device)

    # # 保存生成样本图像
    # fixed_noise = torch.randn(25, nz, 1, 1, device=device)
    # fake = netG(fixed_noise).detach().cpu()
    # grid = vutils.make_grid(fake, nrow=5, normalize=True)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(np.transpose(grid, (1, 2, 0)))
    # plt.axis('off')
    # plt.title('Generated Samples')
    # plt.savefig(os.path.join(samples_dir, 'fake_samples_eval.png'), bbox_inches='tight')
    # plt.close()

    # 保存评估指标
    np.save(os.path.join(metrics_dir, 'metrics_eval.npy'), {
        'fid': fid_value,
        'ssim': ssim_value,
        'lpips': lpips_value,
        'diversity': diversity_value
    })
    
    # 将评估结果追加到训练历史txt
    history_txt = os.path.join(output_dir, 'train_and_eval_log.txt')
    with open(history_txt, 'a', encoding='utf-8') as f:
        f.write("\n评估结果：\n")
        f.write(f"FID: {fid_value:.2f}\n")
        f.write(f"SSIM: {ssim_value:.4f}\n")
        f.write(f"LPIPS: {lpips_value:.4f}\n")
        f.write(f"Diversity: {diversity_value:.4f}\n")

    print(f"评估完成！FID: {fid_value:.2f} | SSIM: {ssim_value:.4f} | LPIPS: {lpips_value:.4f} | Diversity: {diversity_value:.4f}")
    print(f"评估结果已追加到 {history_txt}")

if __name__ == "__main__":
    from config import get_config
    evaluate_gan(get_config())