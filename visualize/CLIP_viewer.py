import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer

def visualize_clip_match(
    image_path,
    task_text,
    model_name="openai/clip-vit-base-patch32",
    null_weight=1.0,
    use_pooler=True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载模型
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. 图像预处理
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 3. 提取文本特征 (目标文本 vs 空文本)
    texts = [task_text, ""]  # 使用空字符串作为基准来抵消背景偏置
    text_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        text_outputs = model.text_model(**text_inputs)

        # 2. 取出语义向量
        if use_pooler and getattr(text_outputs, 'pooler_output', None) is not None:
            text_pooled = text_outputs.pooler_output
        else:
            # 通常CLIP token 0 是 [EOS]/[CLS]
            text_pooled = text_outputs.last_hidden_state[:, 0, :]

        text_features = model.text_projection(text_pooled)
        text_features = F.normalize(text_features, p=2, dim=-1)

        target_text_feat = text_features[0:1]
        null_text_feat = text_features[1:2]

        vision_outputs = model.vision_model(**inputs)
        patch_features = vision_outputs.last_hidden_state[:, 1:, :]

        patch_proj = model.visual_projection(patch_features)
        patch_proj = F.normalize(patch_proj, p=2, dim=-1)

        # 计算相似度，带 logit_scale 与空文本权重
        logit_scale = model.logit_scale.exp()
        sim_target = torch.matmul(patch_proj, target_text_feat.t()) * logit_scale
        sim_null = torch.matmul(patch_proj, null_text_feat.t()) * logit_scale

        rel_sim = sim_target - null_weight * sim_null

        # 将 (batch, num_patches, 1) 之类换成一维 [num_patches]
        rel_sim = rel_sim.view(-1)

        # 归一化到 [0, 1]
        rel_sim = (rel_sim - rel_sim.min()) / (rel_sim.max() - rel_sim.min() + 1e-8)

    # 6. 还原为二维网格
    num_patches = rel_sim.shape[0]
    grid_size = int(math.sqrt(num_patches))
    assert grid_size * grid_size == num_patches, f"patch count {num_patches} 不是平方数，无法reshape为grid"

    heatmap = rel_sim.reshape(grid_size, grid_size).cpu().numpy()

    # 7. 可视化
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title(f"Input: {task_text}")
    ax[0].axis('off')

    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size, resample=Image.BICUBIC)
    ax[1].imshow(image)
    im = ax[1].imshow(heatmap_img, alpha=0.6, cmap='jet')
    ax[1].set_title("Semantic Alignment Heatmap (Bias Removed)")
    ax[1].axis('off')

    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    plt.show()

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


# 使用示例
# visualize_clip_match("path/to/your/image.jpg", "a red cup on the table")