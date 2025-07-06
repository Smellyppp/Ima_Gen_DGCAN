import os
import json
import subprocess
from PIL import Image

#重命名
def rename_jpg_files(folder_path):
    files = os.listdir(folder_path)
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    
    for index, filename in enumerate(jpg_files):
        new_filename = f"211_{index + 1}.jpg"   
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        os.rename(old_file_path, new_file_path)
    print(f"重命名完成: {folder_path}")
    
if __name__ == "__main__":
    normal_folder_path = r"C:\Users\G1581\Desktop\Janus-main-1B\database\normal"
    abnormal_folder_path = r"C:\Users\G1581\Desktop\Janus-main-1B\database\abnormal"
    
    rename_jpg_files(normal_folder_path)
    rename_jpg_files(abnormal_folder_path)  