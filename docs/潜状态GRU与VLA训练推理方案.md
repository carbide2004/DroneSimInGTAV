# 潜状态 GRU 与 VLA 训练/推理方案（定稿）

本文档汇总当前已对齐的技术方案，作为后续实现与论文「系统设计与实验设置」的依据。\
约定：**代码与脚本实现使用英文标识符**；本文档为中文说明。

***

## 1. 目标与总体架构

### 1.1 目标

在现有 **GTAV 仿真 + 观测-动作-意识数据** 基础上，引入：

1. **独立视觉编码器（DINOv2）** 提取每帧特征 (v\_t)；
2. **GRU 潜状态** (h\_t \in \mathbb{R}^{512})，承担时序记忆；
3. **B1：对比对齐**——(h\_t) 与 awareness 文本经 Text Encoder 得到的 (e\_t) 在投影空间中拉近（InfoNCE 或 cosine/MSE 简化版）；
4. **VLA（Qwen3-VL）**：在阶段 2 将 (h\_t) 经 **(P: \mathbb{R}^{512} \to \mathbb{R}^{16 \times d\_{\text{llm}}})** 映射为 **16 个 soft token**，与原有 RGB/Depth、任务文本一并输入 LLM，输出离散动作 `AUTO_*`；
5. **阶段 2 冻结 GRU**，仅训练 soft prompt 投影 (P) 与 VLA 侧适配部分（如 LoRA）。

### 1.2 双视觉支路（设计意图）

| 支路          | 作用              | 说明                                                                |
| ----------- | --------------- | ----------------------------------------------------------------- |
| DINOv2-B/14 | 为 GRU 提供 (v\_t) | **冻结**为主；与 Qwen3-VL 自带视觉塔 **分离**，通过 (h\_t) → soft prompt **桥接**语义 |
| Qwen3-VL 视觉 | VLA 多模态理解       | 与现有 SFT/推理流程一致                                                    |

**注意**：训练 GRU 使用的 vision encoder 与推理须 **一致**（同为 DINOv2-B，同一预处理）。若日后因显存将 B 换为 S，需 **蒸馏或适配层** 对齐特征空间，不可裸换。

***

## 2. 数据与轨迹约定

### 2.1 样本单位

- 训练以 **轨迹** 为单位：({(I\_t, w\_t, a\_t)}\_{t=0}^{T-1})，(T \le 100)，实际常见约 **20** 步。
- 需 **显式** 字段（在现有 `train_data_all*.json` 之上扩展或 sidecar）：

| 字段                     | 说明                                               |
| ---------------------- | ------------------------------------------------ |
| `trajectory_id`        | 唯一轨迹 ID，用于组 batch 与 **按轨迹划分 train/val**（禁止同轨迹泄漏） |
| `step_index`           | 整数，与 `steps.jsonl` 一致                            |
| `images` / 路径          | RGB、Depth（与现有一致）                                 |
| `awareness`            | 四段式英文文本 (w\_t)（无则该步可跳过 B1 或 mask）                |
| `action` / `action_id` | `AUTO_*` 及可选整数标签，便于 CE                           |

### 2.2 可选加速

- 离线预计算并缓存 **(v\_t = \text{DINO}(I\_t))**（建议 fp16），训练 GRU 阶段可 **不再重复前向 ViT**，节省服务器算力。

<br />

**依赖**：`data_processor/requirements.txt`（含 `timm` 等，用于特征缓存；可与 `agent_control/requirements.txt` 合并安装）。

***

## 3. 维度与超参（定稿）

| 项目                 | 取值                                                                                                                                               |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| DINOv2             | **ViT-B/14**，单帧全局特征 **768 维**（(v\_t)）                                                                                                            |
| GRU `hidden_size`  | **512**，即 **(h\_t \in \mathbb{R}^{512})**                                                                                                        |
| 输入投影               | `Linear(768 → 512)` 或与 GRU `input_size` 对齐后再进 GRU（实现时二选一，保持 **输出 (h\_t) 为 512 维**）                                                               |
| B1 文本编码            | 选用小型 **Sentence-Transformer / BERT 类** 均可；**(e\_t)** 经 **`proj_e`**，**(h\_t)** 经 **`proj_h`**，统一到 **(d\_{\text{align}})**（建议 **256**，可写入论文为可调超参） |
| Soft token 数 **K** | **16**                                                                                                                                           |
| Soft 投影 (P)        | (P(h\_t) \in \mathbb{R}^{16 \times d\_{\text{llm}}})，**(d\_{\text{llm}})** 与 **Qwen3-VL 的 hidden size** 一致（以实际 `config.json` 为准）                 |
| 阶段 2               | **GRU 及 DINO 冻结**；训练 **(P)** + VLA 可训练部分                                                                                                         |

***

## 4. 训练流程

### 4.1 阶段 1：GRU + 动作监督 + B1

**数据流（每个有效时间步 (t)）**

1. (v\_t = \text{DINO}(I\_t)) 或读取缓存，(v\_t \in \mathbb{R}^{768})。
2. (h\_t = \text{GRU}(\phi(v\_t), h\_{t-1}))，(h\_{-1}) 为可学习零状态或零向量；**teacher forcing** 使用 GT 轨迹展开。
3. (e\_t = \text{Enc}\_{\text{text}}(w\_t))，normalize 视对比损失实现而定。
4. **损失**：
   - (L\_{\text{act}} = \text{CE}(\text{Head}\_{\text{act}}(h\_t), a\_t))（仅 mask 有效步）；
   - (L\_{\text{B1}})：在 (\text{proj}\_h(h\_t)) 与 (\text{proj}\_e(e\_t)) 上做 **InfoNCE**（或简化 cosine/MSE）；
   - (L = L\_{\text{act}} + \lambda L\_{\text{B1}})。

**可训练参数（默认）**：GRU、(\phi)、(\text{Head}_{\text{act}})、(\text{Enc}_{\text{text}})（或部分冻结）、(\text{proj}\_h/\text{proj}\_e)；**DINO 冻结**。

### 4.2 阶段 2：VLA + Soft Prompt（GRU 冻结）

**数据流（每个有效时间步 (t)）**

1. 用 **阶段 1 权重** 计算 (h\_t)（**不反传**进 GRU）。
2. (z\_t = P(h\_t))，形状 **(16 \times d\_{\text{llm}})**，按框架规则 **拼入** LLM 输入（soft prompt）。
3. Qwen3-VL：当前帧 **RGB/Depth** + 任务与 pose 文本 + (z\_t) → 自回归预测 **动作 token/字符串**，与现有 `AUTO_*` 标签对齐。

**可训练参数**：**(P)**、VLA 侧 LoRA/适配层；**GRU、DINO 冻结**。

### 4.3 资源与实现提示

- **训练**：可使用 **8×32G** 服务器；轨迹 pad 至 100，**padding 步 mask**。
- **本地 24G**：主要用于 **推理**；若显存紧张，优先 **FP16/量化 VLA** 或后续 **DINO-S + 蒸馏**，见第 1.2 节。

***

## 5. 推理流程（闭环，每步）

对 (t = 0,1,\ldots) 直至 `AUTO_STOP_REACHED` 或达 `max_steps`：

1. 从仿真读取 **RGB、Depth**（与现有 `capture()` 一致）。
2. (v\_t = \text{DINO}(I\_t))，(h\_t = \text{GRU}(\phi(v\_t), h\_{t-1}))（**同一套阶段 1 权重与预处理**）。
3. (z\_t = P(h\_t))（**阶段 2 权重**）。
4. Qwen3-VL：**图像 + 文本 + (z\_t)** → decode → **`parse_action`**（与 `agent_control/action_mapping.py` 一致）→ **`dispatch_action`**。
5. 持久化 **(h\_t)**（及可选上一步动作）；**无需** B1 与 (e\_t)。

***

## 6. 与现有仓库的衔接

| 现有模块                                  | 衔接方式                                                            |
| ------------------------------------- | --------------------------------------------------------------- |
| `data_processor/train_data_all*.json` | 补充 `trajectory_id`、`step_index`；或独立 `trajectories.jsonl` 索引     |
| `annotate_awareness.py` 产出            | 作为 (w\_t) 来源；B1 使用 **proj 后对比**，不强制依赖旧版 `representation_vector` |
| `agent_control/run_verification.py`   | 在「模型生成动作」前插入：**DINO → GRU → P → VLA**；其余评测逻辑可沿用                 |
| `agent_control/action_mapping.py`     | 动作解析与执行 **保持不变**                                                |

***

## 7. 后续开工任务清单（实现向）

1. **数据**：轨迹索引与划分脚本； **(v\_t) 缓存** 脚本。
2. **阶段 1**：Dataset（按轨迹）、GRU + 动作头 + B1 + 训练入口。
3. **阶段 2**：加载冻结 GRU；实现 **(P)** 与 VLA 拼接；训练入口。
4. **推理**：整合到 `run_verification` 或新入口；保存/加载 checkpoint 约定。
5. **评测**：离线动作准确率；与现有 `samples.jsonl` / `results.json` 指标对齐；闭环仿真评测。

***

## 8. 文档版本

- **状态**：定稿（与讨论一致：DINO-B，(h\_t) 512，B1，soft token **16**，阶段 2 **冻结 GRU**）。
- **修订**：若更换 Text Encoder、(d\_{\text{align}}) 或 VLA 拼接方式，请更新第 3 节与第 4 节并保留版本日期。

