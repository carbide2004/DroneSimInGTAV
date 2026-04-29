import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import zoom
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from mpl_toolkits.axes_grid1 import ImageGrid

# ---------------------------------------------------------
# 1. 全局科研绘图配置
# ---------------------------------------------------------
# 使用 seaborn-v0_8-paper 提供适合论文的紧凑基础样式
plt.style.use('seaborn-v0_8-paper') 
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], # 英文无衬线字体优先
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300 # 确保屏幕预览时的清晰度，不影响矢量图导出
})

# 配置全局参数以支持中文字符和负号的正确渲染
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def fast_tiled_clip_match(image_path, task_text, model, processor, window_size=144, stride=48):
    """提取图像的滑动窗口切片并计算与目标文本的CLIP相似度热力图"""
    device = next(model.parameters()).device
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # 生成所有切片的坐标和图像块
    tiles = []
    coords = []
    y_range = list(range(0, H - window_size, stride)) + [H - window_size]
    x_range = list(range(0, W - window_size, stride)) + [W - window_size]
    for y in y_range:
        for x in x_range:
            tile = img.crop((x, y, x + window_size, y + window_size))
            tiles.append(tile)
            coords.append((x, y))

    # 批量预处理
    inputs = processor(images=tiles, return_tensors="pt").to(device)
    texts = [task_text, ""]  # 使用空文本作为基准进行对比

    # 提取文本和图像特征
    with torch.no_grad():
        text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        text_outputs = model.text_model(**text_inputs)
        text_pooled = text_outputs.pooler_output
        text_features = model.text_projection(text_pooled)
        text_features = F.normalize(text_features, p=2, dim=-1)

        vision_outputs = model.vision_model(**inputs)
        vision_pooled = vision_outputs.pooler_output
        image_features = model.visual_projection(vision_pooled)
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        # 计算相似度
        logit_scale = model.logit_scale.exp()
        logits = torch.matmul(image_features, text_features.t()) * logit_scale
        sim_target = logits[:, 0] 
        sim_null = logits[:, 1]
        rel_sim = sim_target - sim_null

    # 将结果拼回原图网格计算平均得分
    score_map = np.zeros((H, W))
    count_map = np.zeros((H, W))
    logits_np = rel_sim.cpu().numpy()
    
    for i, (x, y) in enumerate(coords):
        score_map[y:y+window_size, x:x+window_size] += logits_np[i]
        count_map[y:y+window_size, x:x+window_size] += 1

    heatmap = score_map / (count_map + 1e-8)
    return heatmap

def visualize_tiled_clip_match_academic(
    image_path,
    task_texts,
    model_name="openai/clip-vit-base-patch32",
    window_size=144,
    stride=48,
    save_path="academic_clip_heatmap.pdf"
):
    if isinstance(task_texts, str):
        task_texts = [task_texts]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    # 读取原图
    img = plt.imread(image_path)
    H, W, _ = img.shape

    # 计算热力图
    heatmaps = []
    for task_text in task_texts:
        heatmap = fast_tiled_clip_match(image_path, task_text, model, processor, window_size, stride)
        scale_h = H / heatmap.shape[0]
        scale_w = W / heatmap.shape[1]
        heatmap_resized = zoom(heatmap, (scale_h, scale_w), order=3)
        heatmaps.append(heatmap_resized)

    all_values = np.concatenate([h.flatten() for h in heatmaps])
    vmin = np.min(all_values)
    vmax = np.max(all_values)

    num_heatmaps = len(task_texts)
    num_cols = 1 + num_heatmaps
    
    # 创建画布
    fig = plt.figure(figsize=(3.5 * num_cols, 4))
    
    # 核心改动：使用 ImageGrid 替代 subplots 和 GridSpec
    grid = ImageGrid(
        fig, 
        111,                      # 类似于 subplot(111)
        nrows_ncols=(1, num_cols),# 网格布局：1行多列
        axes_pad=0.1,             # 子图之间的间距
        share_all=True,           # 共享坐标轴
        cbar_location="right",    # 颜色条放置在右侧
        cbar_mode="single",       # 所有图共用一个颜色条
        cbar_size="5%",           # 颜色条的宽度
        cbar_pad=0.15             # 颜色条与右侧图片的间距
    )

    # 绘制原图 (放置在网格的第 0 个位置)
    grid[0].imshow(img)
    grid[0].set_title("(a) Original Image", loc='center', pad=10)
    grid[0].axis('off')

    # 循环绘制热力图
    labels = ["(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]
    for i, (text, h_map) in enumerate(zip(task_texts, heatmaps)):
        ax = grid[i + 1]
        ax.imshow(img)
        im = ax.imshow(
            h_map, 
            alpha=0.6, 
            cmap='magma', 
            vmin=vmin, 
            vmax=vmax,
            interpolation='bicubic'
        )
        ax.set_title(f"{labels[i]} Text: '{text}'", loc='center', pad=10)
        ax.axis('off')

    # 将颜色条绘制在 ImageGrid 自动分配并对齐的 cax 上
    cbar = fig.colorbar(im, cax=grid.cbar_axes[0])
    cbar.set_label('CLIP相似度得分', labelpad=15)

    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.show()

if __name__ == '__main__':
    visualize_tiled_clip_match_academic(
        image_path=r"E:\Workplace_Tanhaowen\DroneSimInGTAV\dataset\imgs\20260402_182134_step_000012_rgb.jpg",
        task_texts=["people fighting against others", "a car on fire"],
        window_size=144,
        stride=48,
        save_path="academic_clip_heatmap.pdf"
    )