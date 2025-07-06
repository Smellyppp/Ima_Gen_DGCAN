# 基于DGCAN的二次元头像生成

本项目用于图像生成与训练，包含数据集处理、模型训练、评估和辅助工具脚本。

## 目录结构

.
├── config.py                # 配置文件
├── evaluate.py              # 模型评估脚本
├── gen_img.py               # 图像生成脚本
├── main.py                  # 主程序入口
├── README.md                # 项目说明文档
├── requirements.txt         # Python依赖包列表
├── train.py                 # 模型训练脚本
├── database/                # 数据集目录
│   └── 2c/
├── database2/               # 备用数据集目录
│   └── 2c/
├── original/                # 原始脚本和测试文件
│   ├── gen_img_test.py
│   ├── main_original.py
│   ├── test1.py
│   └── test2.py
├── output/                  # 输出结果目录
│   ├── training_summary.txt
│   ├── generated_images/
│   ├── metrics/
│   ├── models/
│   ├── plots/
│   └── samples/
├── output1/                 # 备用输出目录
│   ├── train_and_eval_log.txt
│   ├── training_history.npy
│   ├── metrics/
│   ├── models/
│   ├── plots/
│   └── samples/
└── tool/                    # 工具脚本
    ├── change_resolution.py
    ├── cuda_test.py
    └── rename.py


## 环境依赖

已安装 Python 3.8 及以上版本。安装依赖包：

```sh
pip install -r requirements.txt
```

## 主要脚本说明

- `main.py`：主程序入口，用于训练或评估模型。
- `train.py`：用于模型训练。
- `evaluate.py`：用于模型评估。
- `gen_img.py`：用于生成图像。
- `config.py`：包含项目的参数配置。
- `tool/`：包含分辨率调整、CUDA测试、文件重命名等辅助工具脚本。

## 数据集

- `database/` 原始数据集（多）
- `database2/` 简易测试数据集（少）

## 输出

- `output/` 目录下保存训练日志、模型、生成图片、评估指标等结果。 
- `output1/` 简易测试输出目录。

## 使用方法

### 运行主程序

```sh
#训练和评估
python main.py  
```

### 生成图片

```sh
python gen_img.py
```


## 工具脚本

- `tool/change_resolution.py`：批量修改图片分辨率。
- `tool/cuda_test.py`：测试CUDA环境。
- `tool/rename.py`：批量重命名文件。
