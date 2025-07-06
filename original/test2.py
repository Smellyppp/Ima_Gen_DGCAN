import glob
import torch
from PIL import Image
from torch import nn
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms

images_path = glob.glob(r'C:/Users/G1581/Desktop/课设/img_pro/database3/2c/*.png')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


class FaceDataset(data.Dataset):
    def __init__(self, images_path):
        self.images_path = images_path

    def __getitem__(self, index):
        image_path = self.images_path[index]
        pil_img = Image.open(image_path)
        pil_img = transform(pil_img)
        return pil_img

    def __len__(self):
        return len(self.images_path)


BATCH_SIZE = 32
dataset = FaceDataset(images_path)
data_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
image_batch = next(iter(data_loader))


# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(100, 256*16*16)
        self.bn1 = nn.BatchNorm1d(256*16*16)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)  # 输出：128*16*16
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 输出：64*32*32
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)  # 输出：3*64*64

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.bn1(x)
        x = x.view(-1, 256, 16, 16)
        x = F.relu(self.deconv1(x))
        x = self.bn2(x)
        x = F.relu(self.deconv2(x))
        x = self.bn3(x)
        x = F.tanh(self.deconv3(x))
        return x


# 定义判别器
class Discrimination(nn.Module):
    def __init__(self):
        super(Discrimination, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2)  # 64*31*31
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)  # 128*15*15
        self.bn1 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128*15*15, 1)

    def forward(self, x):
        x = F.dropout(F.leaky_relu(self.conv1(x)), p=0.3)
        x = F.dropout(F.leaky_relu(self.conv2(x)), p=0.3)
        x = self.bn1(x)
        x = x.view(-1, 128*15*15)
        x = torch.sigmoid(self.fc(x))
        return x


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
gen = Generator().to(device)
dis = Discrimination().to(device)
loss_fn = torch.nn.BCELoss()
gen_opti = torch.optim.Adam(gen.parameters(), lr=0.0001)
dis_opti = torch.optim.Adam(dis.parameters(), lr=0.00001)

output_dir = r"C:\Users\G1581\Desktop\课设\img_pro\output"
os.makedirs(output_dir, exist_ok=True)

# 定义可视化函数
def generate_and_save_images(model, epoch, test_noise_):
    predictions = model(test_noise_).permute(0, 2, 3, 1).cpu().numpy()
    fig = plt.figure(figsize=(20, 160))
    for i in range(predictions.shape[0]):
        plt.subplot(1, 8, i+1)
        plt.imshow((predictions[i]+1)/2)
        # plt.axis('off')
    # 保存图像到 output 文件夹
    plt.savefig(f"{output_dir}/epoch_{epoch}.png")
    plt.close()  # 关闭图像窗口，避免占用内存


test_noise = torch.randn(8, 100, device=device)

#############################
D_loss = []
G_loss = []
# 创建 img_list，用于保存每轮生成的图像
img_list = []

# 修改训练循环部分
for epoch in range(500):
    D_epoch_loss = 0
    G_epoch_loss = 0
    batch_count = len(data_loader)   # 返回批次数
    for step, img in enumerate(data_loader):
        img = img.to(device)
        size = img.shape[0]
        random_noise = torch.randn(size, 100, device=device)  # 生成随机输入

        # 固定生成器，训练判别器
        dis_opti.zero_grad()
        real_output = dis(img)
        d_real_loss = loss_fn(real_output, torch.ones_like(real_output, device=device))
        d_real_loss.backward()
        generated_img = gen(random_noise)
        fake_output = dis(generated_img.detach())
        d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output, device=device))
        d_fake_loss.backward()

        dis_loss = d_real_loss + d_fake_loss
        dis_opti.step()

        # 固定判别器，训练生成器
        gen_opti.zero_grad()
        fake_output = dis(generated_img)
        gen_loss = loss_fn(fake_output, torch.ones_like(fake_output, device=device))
        gen_loss.backward()
        gen_opti.step()

        with torch.no_grad():
            D_epoch_loss += dis_loss.item()
            G_epoch_loss += gen_loss.item()

    with torch.no_grad():
        D_epoch_loss /= batch_count
        G_epoch_loss /= batch_count
        D_loss.append(D_epoch_loss)
        G_loss.append(G_epoch_loss)

        print(f"Epoch: {epoch}, D_loss: {D_epoch_loss:.4f}, G_loss: {G_epoch_loss:.4f}")

        # 每10轮保存一次成果
        if epoch % 10 == 0:
            generate_and_save_images(gen, epoch, test_noise)

# 绘制损失曲线
plt.plot(range(1, len(D_loss)+1), D_loss, label="D_loss")
plt.plot(range(1, len(D_loss)+1), G_loss, label="G_loss")
plt.xlabel('epoch')
plt.legend()
plt.show()


# 创建动画
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i.numpy(), (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

# 在 Jupyter Notebook 中显示动画
HTML(ani.to_jshtml())
