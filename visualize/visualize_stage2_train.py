import json
import matplotlib.pyplot as plt

# 配置全局参数以支持中文字符和负号的正确渲染
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_stage2_history(file_path):
    # 加载数据
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Error: 找不到文件 {file_path}")
        return

    # 提取字段
    epochs = [x['epoch'] for x in history]

    # Train Loss
    train_loss = [x['train']['loss'] for x in history]

    # Val Loss
    val_loss = [x['val']['loss'] for x in history]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='训练集损失')
    plt.plot(epochs, val_loss, label='验证集损失')

    # 图表装饰
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('阶段二训练历史曲线')
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    path = 'agent_control/checkpoints/stage2_rgbdheat_without/history.json'
    plot_stage2_history(path)