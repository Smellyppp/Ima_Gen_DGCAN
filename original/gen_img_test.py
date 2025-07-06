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

if __name__ == "__main__":
    # 配置参数（必须与训练一致）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nz = 256
    ngf = 64
    
    # 初始化生成器
    netG = Generator(nz=nz, ngf=ngf).to(device)
    
    # 加载模型
    model_path = r"C:\Users\G1581\Desktop\课设\img_pro\output3\models\generator_best.pth"
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    
    # 生成图像
    num_images = 25
    noise = torch.randn(num_images, nz, 1, 1, device=device)
    
    with torch.no_grad():
        generated = netG(noise).cpu()
    
    # 保存结果
    output_dir = r"C:\Users\G1581\Desktop\课设\img_pro\output3\generated_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存网格图
    grid = vutils.make_grid(generated, nrow=5, normalize=True, scale_each=True)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'final_generation.png'))
    
    
    # 加载模型
    model_path = r"C:\Users\G1581\Desktop\课设\img_pro\output3\models\generator_best_loss.pth"
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    
    # 生成图像
    num_images = 25
    noise = torch.randn(num_images, nz, 1, 1, device=device)
    
    with torch.no_grad():
        generated = netG(noise).cpu()
    
    # 保存结果
    output_dir = r"C:\Users\G1581\Desktop\课设\img_pro\output3\generated_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存网格图
    grid = vutils.make_grid(generated, nrow=5, normalize=True, scale_each=True)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'final_generation_loss.png'))
    
    # # 保存单张图像
    # for i, img in enumerate(generated):
    #     vutils.save_image(img, os.path.join(output_dir, f'image_{i}.png'), normalize=True)
    
    # print(f"成功生成 {num_images} 张图像至 {output_dir}")