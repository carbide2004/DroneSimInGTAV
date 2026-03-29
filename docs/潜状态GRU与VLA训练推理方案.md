# 潜状态 GRU 与 VLA 训练/推理方案（定稿）

本文档汇总当前已对齐的技术方案，作为后续实现与论文「系统设计与实验设置」的依据。  
约定：**代码与脚本实现使用英文标识符**；本文档为中文说明。

---

## 1. 目标与总体架构

### 1.1 目标

在现有 **GTAV 仿真 + 观测-动作-意识数据** 基础上，引入：

1. **独立视觉编码器（DINOv2）** 提取每帧特征 \(v_t\)；
2. **GRU 潜状态** \(h_t \in \mathbb{R}^{512}\)，承担时序记忆；
3. **B1：对比对齐**——\(h_t\) 与 awareness 文本经 Text Encoder 得到的 \(e_t\) 在投影空间中拉近（InfoNCE 或 cosine/MSE 简化版）；
4. **VLA（Qwen3-VL）**：在阶段 2 将 \(h_t\) 经 **\(P: \mathbb{R}^{512} \to \mathbb{R}^{16 \times d_{\text{llm}}}\)** 映射为 **16 个 soft token**，与原有 RGB/Depth、任务文本一并输入 LLM，输出离散动作 `AUTO_*`；
5. **阶段 2 冻结 GRU**，仅训练 soft prompt 投影 \(P\) 与 VLA 侧适配部分（如 LoRA）。

### 1.2 双视觉支路（设计意图）

| 支路 | 作用 | 说明 |
|------|------|------|
| DINOv2-B/14 | 为 GRU 提供 \(v_t\) | **冻结**为主；与 Qwen3-VL 自带视觉塔 **分离**，通过 \(h_t\) → soft prompt **桥接**语义 |
| Qwen3-VL 视觉 | VLA 多模态理解 | 与现有 SFT/推理流程一致 |

**注意**：训练 GRU 使用的 vision encoder 与推理须 **一致**（同为 DINOv2-B，同一预处理）。若日后因显存将 B 换为 S，需 **蒸馏或适配层** 对齐特征空间，不可裸换。

---

## 2. 数据与轨迹约定

### 2.1 样本单位

- 训练以 **轨迹** 为单位：\(\{(I_t, w_t, a_t)\}_{t=0}^{T-1}\)，\(T \le 100\)，实际常见约 **20** 步。
- 需 **显式** 字段（在现有 `train_data_all*.json` 之上扩展或 sidecar）：

| 字段 | 说明 |
|------|------|
| `trajectory_id` | 唯一轨迹 ID，用于组 batch 与 **按轨迹划分 train/val**（禁止同轨迹泄漏） |
| `step_index` | 整数，与 `steps.jsonl` 一致 |
| `images` / 路径 | RGB、Depth（与现有一致） |
| `awareness` | 四段式英文文本 \(w_t\)（无则该步可跳过 B1 或 mask） |
| `action` / `action_id` | `AUTO_*` 及可选整数标签，便于 CE |

### 2.2 可选加速

- 离线预计算并缓存 **\(v_t = \text{DINO}(I_t)\)**（建议 fp16），训练 GRU 阶段可 **不再重复前向 ViT**，节省服务器算力。

### 2.3 已实现的数据脚本（`data_processor/`）

| 脚本 / 模块 | 作用 |
|-------------|------|
| `action_vocab.py` | `AUTO_*` ↔ `action_id`（0..5），与 `agent_control/action_mapping.py` 动作集合一致 |
| `trajectory_utils.py` | 解析/补全 `trajectory_id`、`step_index`、`action_id`（优先显式字段，否则从 `imgs/*_step_*_rgb.jpg` 解析） |
| `data_processor.py` | 由 `manual/*/steps.jsonl` 生成 `dataset/train_data_all.json`，**每条含** `trajectory_id`、`step_index`、`action_id` |
| `build_trajectory_index.py` | 从列表 JSON 生成 `dataset/trajectories.jsonl`（按轨迹聚合、步序排序） |
| `split_trajectories.py` | **按 trajectory_id** 划分 train/val 列表 JSON，避免同轨迹泄漏 |
| `enrich_trajectory_fields.py` | 给旧版无字段的 JSON **补字段** 并另存 |
| `cache_dino_features.py` | 用 **timm DINOv2 ViT-B/14** 预计算 RGB 特征（默认 fp16 `.npy`）+ `features/.../manifest.jsonl` |
| `shuffle_data.py` | 仅打乱列表顺序；**GRU 训练请勿依赖行级 shuffle**（已打印提示） |

**依赖**：`data_processor/requirements.txt`（含 `timm` 等，用于特征缓存；可与 `agent_control/requirements.txt` 合并安装）。

---

## 3. 维度与超参（定稿）

| 项目 | 取值 |
|------|------|
| DINOv2 | **ViT-B/14**，单帧全局特征 **768 维**（\(v_t\)） |
| GRU `hidden_size` | **512**，即 **\(h_t \in \mathbb{R}^{512}\)** |
| 输入投影 | `Linear(768 → 512)` 或与 GRU `input_size` 对齐后再进 GRU（实现时二选一，保持 **输出 \(h_t\) 为 512 维**） |
| B1 文本编码 | 选用小型 **Sentence-Transformer / BERT 类** 均可；**\(e_t\)** 经 **`proj_e`**，**\(h_t\)** 经 **`proj_h`**，统一到 **\(d_{\text{align}}\)**（建议 **256**，可写入论文为可调超参） |
| Soft token 数 **K** | **16** |
| Soft 投影 \(P\) | \(P(h_t) \in \mathbb{R}^{16 \times d_{\text{llm}}}\)，**\(d_{\text{llm}}\)** 与 **Qwen3-VL 的 hidden size** 一致（以实际 `config.json` 为准） |
| 阶段 2 | **GRU 及 DINO 冻结**；训练 **\(P\)** + VLA 可训练部分 |

---

## 4. 训练流程

### 4.1 阶段 1：GRU + 动作监督 + B1

**数据流（每个有效时间步 \(t\)）**

1. \(v_t = \text{DINO}(I_t)\) 或读取缓存，\(v_t \in \mathbb{R}^{768}\)。
2. \(h_t = \text{GRU}(\phi(v_t), h_{t-1})\)，\(h_{-1}\) 为可学习零状态或零向量；**teacher forcing** 使用 GT 轨迹展开。
3. \(e_t = \text{Enc}_{\text{text}}(w_t)\)，normalize 视对比损失实现而定。
4. **损失**：  
   - \(L_{\text{act}} = \text{CE}(\text{Head}_{\text{act}}(h_t), a_t)\)（仅 mask 有效步）；  
   - \(L_{\text{B1}}\)：在 \(\text{proj}_h(h_t)\) 与 \(\text{proj}_e(e_t)\) 上做 **InfoNCE**（或简化 cosine/MSE）；  
   - \(L = L_{\text{act}} + \lambda L_{\text{B1}}\)。

**可训练参数（默认）**：GRU、\(\phi\)、\(\text{Head}_{\text{act}}\)、\(\text{Enc}_{\text{text}}\)（或部分冻结）、\(\text{proj}_h/\text{proj}_e\)；**DINO 冻结**。

### 4.2 阶段 2：VLA + Soft Prompt（GRU 冻结）

**数据流（每个有效时间步 \(t\)）**

1. 用 **阶段 1 权重** 计算 \(h_t\)（**不反传**进 GRU）。
2. \(z_t = P(h_t)\)，形状 **\(16 \times d_{\text{llm}}\)**，按框架规则 **拼入** LLM 输入（soft prompt）。
3. Qwen3-VL：当前帧 **RGB/Depth** + 任务与 pose 文本 + \(z_t\) → 自回归预测 **动作 token/字符串**，与现有 `AUTO_*` 标签对齐。

**可训练参数**：**\(P\)**、VLA 侧 LoRA/适配层；**GRU、DINO 冻结**。

### 4.3 资源与实现提示

- **训练**：可使用 **8×32G** 服务器；轨迹 pad 至 100，**padding 步 mask**。
- **本地 24G**：主要用于 **推理**；若显存紧张，优先 **FP16/量化 VLA** 或后续 **DINO-S + 蒸馏**，见第 1.2 节。

---

## 5. 推理流程（闭环，每步）

对 \(t = 0,1,\ldots\) 直至 `AUTO_STOP_REACHED` 或达 `max_steps`：

1. 从仿真读取 **RGB、Depth**（与现有 `capture()` 一致）。
2. \(v_t = \text{DINO}(I_t)\)，\(h_t = \text{GRU}(\phi(v_t), h_{t-1})\)（**同一套阶段 1 权重与预处理**）。
3. \(z_t = P(h_t)\)（**阶段 2 权重**）。
4. Qwen3-VL：**图像 + 文本 + \(z_t\)** → decode → **`parse_action`**（与 `agent_control/action_mapping.py` 一致）→ **`dispatch_action`**。
5. 持久化 **\(h_t\)**（及可选上一步动作）；**无需** B1 与 \(e_t\)。

---

## 6. 与现有仓库的衔接

| 现有模块 | 衔接方式 |
|----------|----------|
| `data_processor/train_data_all*.json` | 补充 `trajectory_id`、`step_index`；或独立 `trajectories.jsonl` 索引 |
| `annotate_awareness.py` 产出 | 作为 \(w_t\) 来源；B1 使用 **proj 后对比**，不强制依赖旧版 `representation_vector` |
| `agent_control/run_verification.py` | 在「模型生成动作」前插入：**DINO → GRU → P → VLA**；其余评测逻辑可沿用 |
| `agent_control/action_mapping.py` | 动作解析与执行 **保持不变** |

---

## 7. 后续开工任务清单（实现向）

1. **数据**：轨迹索引与划分脚本；可选 **\(v_t\) 缓存** 脚本。（**已完成**，见 §2.3）  
2. **阶段 1**：Dataset（按轨迹）、GRU + 动作头 + B1 + 训练入口。  
3. **阶段 2**：加载冻结 GRU；实现 **\(P\)** 与 VLA 拼接；训练入口。  
4. **推理**：整合到 `run_verification` 或新入口；保存/加载 checkpoint 约定。  
5. **评测**：离线动作准确率；与现有 `samples.jsonl` / `results.json` 指标对齐；闭环仿真评测。

---

## 8. 文档版本

- **状态**：定稿（与讨论一致：DINO-B，\(h_t\) 512，B1，soft token **16**，阶段 2 **冻结 GRU**）。  
- **修订**：若更换 Text Encoder、\(d_{\text{align}}\) 或 VLA 拼接方式，请更新第 3 节与第 4 节并保留版本日期。
