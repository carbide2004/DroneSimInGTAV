import json
import matplotlib.pyplot as plt

# 配置全局参数以支持中文字符和负号的正确渲染
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_history(file_paths, notes):
    # 加载数据
    historys = []
    try:
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                historys.append(json.load(f))
    except FileNotFoundError:
        print(f"Error: 找不到文件 {file_path}")
        return

    plt.figure(figsize=(10, 6))

    for i, history in enumerate(historys):
        note = notes[i]
        # 提取字段
        epochs = [x['epoch'] for x in history]
        
        # 验证损失
        val_act = [x['val']['action_loss'] for x in history]
        val_b1 = [x['val']['b1_loss'] for x in history]

        # 绘图
        
        # plt.plot(epochs, train_act, label='训练集动作预测损失(L_action)')
        # plt.plot(epochs, train_b1, label='训练集语义对齐损失(L_align)')
        plt.plot(epochs, val_act, label=f'验证集动作预测损失 - {note}')
        plt.plot(epochs, val_b1, label=f'验证集语义对齐损失 - {note}')

    # 图表装饰
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('阶段一在不同lambda下训练历史曲线')
    plt.legend()
    plt.grid(True)
        
    plt.show()

if __name__ == "__main__":
    paths = [
        'agent_control/checkpoints/stage1_rgbdheat_0/history.json',
        'agent_control/checkpoints/stage1_rgbdheat_0.1/history.json',
        'agent_control/checkpoints/stage1_rgbdheat_1/history.json',
        'agent_control/checkpoints/stage1_rgbdheat_10/history.json',
        'agent_control/checkpoints/stage1_rgbdheat_100/history.json',
    ]
    notes = [
        'lambda=0',
        'lambda=0.1',
        'lambda=1',
        'lambda=10',
        'lambda=100',
    ]
    plot_history(paths, notes)
