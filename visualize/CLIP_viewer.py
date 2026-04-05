import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer

def fast_tiled_clip_match(image_path, task_text, model, processor, window_size=144, stride=48):
    device = next(model.parameters()).device
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # 1. 生成所有切片的坐标和图像块
    tiles = []
    coords = []
    y_range = list(range(0, H - window_size, stride)) + [H - window_size]
    x_range = list(range(0, W - window_size, stride)) + [W - window_size]
    for y in y_range:
        for x in x_range:
            tile = img.crop((x, y, x + window_size, y + window_size))
            tiles.append(tile)
            coords.append((x, y))

    # 2. 批量预处理 (Processor 内部会自动处理 Resize 和 Normalization)
    # 将所有 PIL Image 转化为一个大的 Batch Tensor
    inputs = processor(images=tiles, return_tensors="pt").to(device)
    texts = [task_text, ""]  # 同样使用空文本作为基准

    # 3. 提取文本和图像特征
    with torch.no_grad():
        # 预计算文本特征
        text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        text_outputs = model.text_model(**text_inputs)

        text_pooled = text_outputs.pooler_output
        text_features = model.text_projection(text_pooled)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # 批量提取图像特征 (Shape: [N, 512])
        vision_outputs = model.vision_model(**inputs)
        vision_pooled = vision_outputs.pooler_output
        image_features = model.visual_projection(vision_pooled)
        image_features = F.normalize(image_features, p=2, dim=-1)
        # 计算相似度 (Shape: [N,2])
        logit_scale = model.logit_scale.exp()
        logits = torch.matmul(image_features, text_features.t()) * logit_scale
        sim_target = logits[:, 0] 
        sim_null = logits[:, 1]
        rel_sim = sim_target - sim_null

    # 4. 将结果拼回原图网格
    score_map = np.zeros((H, W))
    count_map = np.zeros((H, W))
    
    logits_np = rel_sim.cpu().numpy()
    for i, (x, y) in enumerate(coords):
        score_map[y:y+window_size, x:x+window_size] += logits_np[i]
        count_map[y:y+window_size, x:x+window_size] += 1

    # 计算平均得分
    heatmap = score_map / (count_map + 1e-8)
    
    return heatmap

def visualize_tiled_clip_match(
    image_path,
    task_text,
    model_name="openai/clip-vit-base-patch32",
    window_size=144,
    stride=48,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    # 1. 获取原始相似度矩阵 (此时 heatmap 是 float，包含原始 logit 差值)
    heatmap = fast_tiled_clip_match(image_path, task_text, model, processor, window_size, stride)

    # 2. 图像读取
    img = plt.imread(image_path)
    H, W, _ = img.shape

    # 3. 核心改进：使用 scipy.ndimage.zoom 进行高质量浮点数缩放
    # 计算缩放比例，将 heatmap (h, w) 缩放到 (H, W)
    scale_h = H / heatmap.shape[0]
    scale_w = W / heatmap.shape[1]
    # order=3 对应双三次插值 (Bicubic)，保留浮点数精度
    from scipy.ndimage import zoom
    heatmap_resized = zoom(heatmap, (scale_h, scale_w), order=3)

    # 4. 绘图
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    # 左图：原图
    ax[0].imshow(img)
    ax[0].set_title(f"Input: {task_text}")
    ax[0].axis('off')

    # 右图：原始数值热力图
    ax[1].imshow(img)
    # 关键点：直接传入浮点数数组，通过 vmin/vmax 控制颜色映射范围
    # 如果你想让“0”始终对应颜色条的中心，可以设置 vmin=-abs_max, vmax=abs_max
    v_max = np.max(np.abs(heatmap_resized))
    im = ax[1].imshow(
        heatmap_resized, 
        alpha=0.6, 
        cmap='jet', 
        # vmin=0, # 如果你只想看正向相关性
        # vmax=v_max # 动态设置最大值
    )
    ax[1].set_title("Raw Logit Difference (Unnormalized)")
    ax[1].axis('off')

    # 5. 添加颜色条，这能让你看到真实的 Logit 差异数值
    cbar = plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.set_label('Logit Difference (Target - Null)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    visualize_tiled_clip_match(
        r"E:\Workplace_Tanhaowen\DroneSimInGTAV\dataset\imgs\20260402_182134_step_000012_rgb.jpg",
        "people fighting against others",
        window_size=144,
        stride=48
    )