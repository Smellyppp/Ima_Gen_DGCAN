from PIL import Image
import os

# 输入文件夹路径
input_folder = r"C:\Users\G1581\Desktop\课设\img_pro\database\cat"  # 替换为你的文件夹路径
output_folder = r"C:\Users\G1581\Desktop\课设\img_pro\database2\cat"  # 替换为输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):  # 支持的图片格式
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")  # 保存为 PNG 格式

        try:
            # 打开图片并转换为 64×64 分辨率
            with Image.open(input_path) as img:
                img = img.resize((64, 64), Image.Resampling.LANCZOS)  # 使用 LANCZOS 替代 ANTIALIAS
                img.save(output_path, format="PNG")  # 保存为 PNG 格式
                print(f"已保存: {output_path}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")