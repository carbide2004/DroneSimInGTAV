import json
import matplotlib.pyplot as plt

def plot_history(file_path):
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
    train_act = [x['train']['action_loss'] for x in history]
    train_b1 = [x['train']['b1_loss'] for x in history]
    
    # Val Loss
    val_act = [x['val']['action_loss'] for x in history]
    val_b1 = [x['val']['b1_loss'] for x in history]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_act, label='train_act')
    plt.plot(epochs, train_b1, label='train_b1')
    plt.plot(epochs, val_act, label='val_act')
    plt.plot(epochs, val_b1, label='val_b1')

    # 图表装饰
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    path = 'agent_control/checkpoints/stage1_heatmap_depth/history.json'
    plot_history(path)