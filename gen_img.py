import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os

# 定义生成器（必须与训练代码相同）
class Generator(torch.nn.Module):
    def __init__(self, nz=256, ngf=64):
        super(Generator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(ngf * 8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 4),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

# 定义判别器（必须与训练代码相同）
class Discriminator(torch.nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # 配置参数（必须与训练一致）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nz = 256
    ngf = 64
    ndf = 64

    # 初始化生成器和判别器
    netG1 = Generator(nz=nz, ngf=ngf).to(device)
    netD = Discriminator(ndf=ndf).to(device)

    # 加载生成器和判别器权重
    netG1.load_state_dict(torch.load(r"C:\Users\G1581\Desktop\GitHub\img_pro\output\models\generator_best.pth", map_location=device))
    # 判别器
    netD.load_state_dict(torch.load(r"C:\Users\G1581\Desktop\GitHub\img_pro\output\models\discriminator_best.pth", map_location=device))
    netG1.eval()
    netD.eval()

    output_dir = r"C:\Users\G1581\Desktop\GitHub\img_pro\output1\generated_images"
    os.makedirs(output_dir, exist_ok=True)

    selected_images = []
    selected_sources = []
    max_images = 25
    batch_size = 16  # 每次生成的图片数

    round_id = 1
    while len(selected_images) < max_images:
        # 只用一个生成器
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        with torch.no_grad():
            fake_imgs = netG1(noise)
            scores = netD(fake_imgs).view(-1)
            sorted_idx = torch.argsort(scores, descending=True)

            # 判别器筛选
            for rank, idx in enumerate(sorted_idx):
                img = fake_imgs[idx].cpu()
                score = scores[idx].item()
                if score > 0.65:
                    selected_images.append(img)
                    selected_sources.append('best')
                    if len(selected_images) >= max_images:
                        break
            if len(selected_images) >= max_images:
                break
        print(f"已生成第 {round_id} 轮")
        round_id += 1


    # 保存最终筛选的网格图
    final_scores = []
    with torch.no_grad():
        for img in selected_images[:max_images]:
            img_tensor = img.unsqueeze(0).to(device)
            score = netD(img_tensor).item()
            final_scores.append(score)

    # 按分数从高到低排序
    sorted_indices = np.argsort(final_scores)[::-1]
    selected_images_sorted = [selected_images[i] for i in sorted_indices]
    final_scores_sorted = [final_scores[i] for i in sorted_indices]

    grid = vutils.make_grid(selected_images_sorted, nrow=5 if max_images > 16 else 4, normalize=True, scale_each=True)
    npimg = np.transpose(grid.numpy(), (1, 2, 0))

    plt.figure(figsize=(10, 10) if max_images > 16 else (8, 8))
    plt.imshow(npimg)
    plt.axis('off')

    # 在每个子图上写分数
    nrow = 5 if max_images > 16 else 4
    for i in range(len(selected_images_sorted)):
        row = i // nrow
        col = i % nrow
        h, w = selected_images_sorted[0].shape[1], selected_images_sorted[0].shape[2]
        x = col * (w + 2) + 5
        y = row * (h + 2) + 20
        plt.text(x, y, f"{final_scores_sorted[i]:.2f}", color='yellow', fontsize=12, weight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_generation_selected.png'))
    plt.close()