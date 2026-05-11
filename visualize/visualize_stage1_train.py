import json
import matplotlib.pyplot as plt

# 配置全局参数以支持中文字符和负号的正确渲染
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

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
    
    # 训练损失
    train_act = [x['train']['action_loss'] for x in history]
    train_b1 = [x['train']['b1_loss'] for x in history]
    
    # 验证损失
    val_act = [x['val']['action_loss'] for x in history]
    val_b1 = [x['val']['b1_loss'] for x in history]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_act, label='训练集动作预测损失(L_action)')
    plt.plot(epochs, train_b1, label='训练集语义对齐损失(L_align)')
    plt.plot(epochs, val_act, label='验证集动作预测损失(L_action)')
    plt.plot(epochs, val_b1, label='验证集语义对齐损失(L_align)')

    # 图表装饰
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('阶段一训练历史曲线')
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    path = 'agent_control/checkpoints/stage1_rgbdheat_10/history.json'
    plot_history(path)
