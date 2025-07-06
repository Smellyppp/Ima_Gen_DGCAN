from train import train_gan
from evaluate import evaluate_gan
from config import get_config

def main():
    config = get_config()
    print("开始训练...")
    train_gan(config)
    print("训练完成，开始评估...")
    evaluate_gan(config)
    print("全部流程结束！")

if __name__ == "__main__":
    main()