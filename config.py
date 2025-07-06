import torch

def get_config():
    """
    返回训练和评估所需的配置参数字典。
    """
    return {
        'batch_size': 128,  # 每个批次的图片数量
        'image_size': 64,   # 输入图片的尺寸（宽和高）
        'nz': 256,          # 噪声向量的维度（生成器输入）
        'ngf': 64,          # 生成器特征图数量基数
        'ndf': 64,          # 判别器特征图数量基数
        'num_epochs': 21,   # 训练轮数
        'lr': 0.0001,       # 学习率
        'beta1': 0.5,       # Adam优化器的beta1参数
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # 训练设备
        'output_dir': r"C:\Users\G1581\Desktop\GitHub\img_pro\output1",            # 输出文件夹
        'data_root': r"C:\Users\G1581\Desktop\GitHub\img_pro\database2",            # 数据集根目录
        'num_eval_samples': 1000  # 评估时采样的图片数量
    }