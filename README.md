# DroneSimInGTAV / DroneSim

这是一个基于 GTA V + ScriptHookV 的无人机仿真、数据采集、轨迹处理、模型训练和验证项目。仓库同时包含：

- GTA V 插件端：负责相机控制、采集 RGB/Depth、异常场景生成、位姿控制。
- Python 控制端：连接插件，执行在线验证、模型推理和动作分发。
- 数据处理脚本：把原始采集数据转换成训练数据，补充 awareness 标注，切分数据集。
- 训练脚本：Stage1、Stage2、E2E SMT-GRU 等训练入口。
- 离线 replay 验证：把轨迹帧构造成 SQLite 数据库，在没有 GTA V 环境的服务器上做覆盖式离线验证。

## 目录结构

```text
DroneSim/
├── DroneSim/                 # C++ GTA V 插件源码
├── agent_control/            # Python 控制、训练、验证脚本
├── data_processor/           # 数据转换、清洗、awareness 标注脚本
├── visualize/                # 数据和训练结果可视化脚本
├── docs/                     # 训练管线和设计说明
├── dataset/                  # 本地训练数据，默认被 .gitignore 忽略
├── data/                     # 本地采集/验证数据，默认被 .gitignore 忽略
├── deps/                     # ScriptHookV / DirectXTK 相关依赖
├── 3rdParty/                 # 第三方库
└── DroneSim.sln              # Visual Studio 解决方案
```

## 环境准备

### 目标运行环境

当前说明面向两类实际运行环境：

- 24GB RTX 4090 工作机：用于 Windows / GTA V / ScriptHookV / `DroneSim.asi` 插件、轨迹采集、人工审查和在线验证。
- RTX 5090 服务器：用于数据处理、awareness 标注、模型训练和离线 replay 验证；服务器通常不需要运行 GTA V。

当前编写 README 的机器不作为项目运行环境。命令中的路径应按 4090 工作机或 5090 服务器上的实际 clone 位置替换。

### Windows / GTA V 插件端

插件端需要 Windows、GTA V、ScriptHookV 和 Visual Studio 2022。项目中已有 `DroneSim.sln`，通常用 Visual Studio 打开后选择 `x64 Release` 编译。

命令行构建可使用 Visual Studio 2022 的 MSBuild：

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe" DroneSim.sln /p:Configuration=Release /p:Platform=x64
```

编译产物通常是：

```text
DroneSim/x64/Release/DroneSim.asi
```

把 `.asi` 放到 GTA V 可加载 ScriptHookV 插件的位置后，Python 控制端才能连接到插件服务。

## C++ 插件端控制

插件启动后会在 GTA V 进程内初始化 `ServerV2`，监听本机 TCP 端口：

```text
127.0.0.5:23456
```

Python 端的 `agent_control/dronesim_client.py` 使用同一套 `DSV2` 二进制协议连接插件。正常流程是先进入 GTA V，确认 `DroneSim.asi` 已加载，再运行 Python 控制或验证脚本。

### 键盘控制

插件端支持直接在 GTA V 内操作相机和采集：

| 按键 | 作用 |
| --- | --- |
| `F10` | 创建脚本相机，进入无人机相机模式 |
| `F11` | 停止脚本相机，返回玩家视角；如果正在记录，也会停止记录 |
| `W` | 相机向当前朝向前进一个 `STEPSIZE` |
| `Shift` | 相机上升一个 `STEPSIZE` |
| `Ctrl` | 相机下降一个 `STEPSIZE` |
| `Q` | 相机左转一个 `YAW_STEPSIZE` |
| `E` | 相机右转一个 `YAW_STEPSIZE` |
| `F12` | 批量自动采集，当前代码默认采集 `AUTO_EVENT_FIRE`，数量为 100 |
| `F6` | 连续手动采集，当前代码默认采集 `AUTO_EVENT_FIRE`，数量为 100 |
| `F7` | 在普通相机模式下记录 `AUTO_STOP_REACHED` 并停止记录；在连续手动采集中记录失败停止 |

当前 `F12` 和 `F6` 默认都是火灾事件。如果要改成事故或 arrest，需要在 `DroneSim/script.cpp` 中把对应调用从 `AUTO_EVENT_FIRE` 改成 `AUTO_EVENT_ACCIDENT` 或 `AUTO_EVENT_ARREST` 后重新编译插件。

### 采集输出

C++ 端采集数据默认写到 GTA V 工作目录下的：

```text
data/manual/<session>/
├── RGB/
├── Depth/
├── steps.jsonl
└── metadata.jsonl
```

`steps.jsonl` 记录每一步的动作、位姿、RGB/Depth 文件路径和图像尺寸；`metadata.jsonl` 记录任务、异常类型、目标位置、起点和步数等信息。后续可用 `data_processor/data_processor.py` 转成 `dataset/train_data_all.json`。

### Python 可调用的插件命令

Python 客户端封装在 `agent_control/dronesim_client.py`，常用接口如下：

| Python 方法 | C++ 消息 | 说明 |
| --- | --- | --- |
| `create_camera()` | `MSG_CREATE_CAMERA` | 创建并启用脚本相机 |
| `stop_camera()` | `MSG_STOP_CAMERA` | 停止脚本相机 |
| `move(dx, dy, dz)` | `MSG_MOVE` | 按相机局部坐标移动 |
| `rotate(rx, ry, rz)` | `MSG_ROTATE` | 按增量旋转相机 |
| `set_posture(x, y, z, rx, ry, rz)` | `MSG_SET_POSTURE` | 设置相机绝对位姿 |
| `get_pose()` | `MSG_GET_POSE` | 读取当前相机位姿 |
| `capture()` | `MSG_CAPTURE` | 获取当前 RGB/Depth buffer |
| `set_fov(fov)` | `MSG_SET_FOV` | 设置相机 FOV |
| `set_time(h, m, s)` | `MSG_SET_TIME` | 设置游戏时间 |
| `set_weather(name)` | `MSG_SET_WEATHER` | 设置天气 |
| `create_fire()` | `MSG_CREATE_FIRE` | 在相机附近生成火灾异常 |
| `create_arrest()` | `MSG_CREATE_ARREST` | 在相机附近生成 arrest 场景 |
| `create_accident()` | `MSG_CREATE_ACCIDENT` | 在相机附近生成事故场景 |
| `teleport_player(x, y, z)` | `MSG_TELEPORT_PLAYER` | 将玩家传送到异常中心，并设为不可见/无敌 |
| `restore_player()` | `MSG_RESTORE_PLAYER` | 验证结束后恢复玩家状态 |
| `set_recording_session(name, task)` | `MSG_SET_RECORDING_SESSION` | 设置下一次采集 session 名和任务描述 |
| `get_recording_info()` | `MSG_GET_RECORDING_INFO` | 查询当前记录状态、步数和 session 路径 |

最小连通性检查：

```powershell
python -c "from agent_control.dronesim_client import DroneSimClient; c=DroneSimClient(); c.create_camera(); print(c.get_pose()); c.stop_camera()"
```

如果这里连接失败，优先确认 GTA V 正在运行、`DroneSim.asi` 已加载、端口 `23456` 没被防火墙或其他进程拦截。

## Python 环境

建议使用 conda 独立环境。PowerShell 中先切到仓库根目录：

```powershell
cd <repo>
```

创建并激活环境：

```powershell
conda create -n dronesim python=3.10 -y
```

```powershell
conda activate dronesim
```

安装控制和训练依赖：

```powershell
pip install -r agent_control\requirements.txt
```

安装数据处理依赖：

```powershell
pip install -r data_processor\requirements.txt
```

如果需要训练 Qwen / LoRA / 多 GPU，请根据机器 CUDA 版本安装匹配的 PyTorch，并确认模型目录已准备好。

## 数据准备

训练脚本不直接读取 GTA V 采集出的原始轨迹，而是读取 schema v2 训练数据。完整生产链路是：

```text
编译 DroneSim.asi
  -> 加载到 GTA V
  -> F12 自动采集或 F6 连续手动采集
  -> 得到 GTA V 目录下 data/manual/<session> 原始轨迹
  -> 人工审查轨迹，删除明显坏数据
  -> 移动标记为探索失败的轨迹到 failed
  -> 整理通过审批的轨迹到 checked
  -> 转换为 dataset/train_data_all.json 和 dataset/imgs
  -> awareness 标注
  -> 得到 dataset/train_data_all_with_awareness.json
```

通常在 4090 工作机完成 GTA V 插件加载、`data/manual` 原始轨迹采集和人工审查；整理后的 `checked` 轨迹、`dataset/`、模型权重和验证文件再同步到 5090 服务器，用于标注、训练和离线验证。

最终训练脚本默认使用：

```text
dataset/train_data_all.json
dataset/train_data_all_with_awareness.json
dataset/imgs/
```

其中每条样本通常包含：

- `sample_id`
- `trajectory_id`
- `step_index`
- `pose`: `x, y, z, rx, ry, rz`
- `action.name`
- `action_id`
- `task`
- `observations.rgb.path`
- `observations.depth.path`
- `awareness`，如果已经完成 awareness 标注

`dataset/` 和 `data/` 默认不进 git。新机器 clone 后需要单独拷贝数据集、模型权重和验证文件。

## 从 GTA V 采集原始轨迹

先按前面的 C++ 插件端说明编译 `DroneSim.asi`，并确认 GTA V 中插件已经加载。

进入相机模式：

```text
F10
```

采集方式：

- `F12`：批量自动采集。当前代码默认采集 100 条 `AUTO_EVENT_FIRE` 轨迹。
- `F6`：连续手动采集。插件先生成场景，再由你用 `W / Shift / Ctrl / Q / E` 控制相机移动。
- 手动采集中，靠近目标会自动记录 `AUTO_STOP_REACHED`；也可以用 `F7` 标记失败停止。
- `F11`：退出相机模式并停止当前记录。

采集结果写到 GTA V 工作目录：

```text
<GTA V>/data/manual/<session>/
├── RGB/
├── Depth/
├── steps.jsonl
└── metadata.jsonl
```

## 审查和整理原始轨迹

先用 `judge_trajectory.py` 播放轨迹，人工删除明显错误或损坏的数据。当前脚本默认路径是：

```text
E:\ToolApps\Steam\steamapps\common\Grand Theft Auto V\data\manual
```

默认路径可直接运行：

```powershell
python data_processor\judge_trajectory.py
```

如果 GTA V 路径不同，可以用 inline 方式指定目录：

```powershell
python -c "from data_processor.judge_trajectory import SessionViewer; SessionViewer(r'E:\ToolApps\Steam\steamapps\common\Grand Theft Auto V\data\manual').run()"
```

审查窗口快捷键：

| 按键 | 作用 |
| --- | --- |
| `K` | 下一条轨迹 |
| `J` | 上一条轨迹 |
| `P` | 暂停/继续 |
| `D` | 直接删除当前轨迹目录 |

然后把最后一帧标记为 `AUTO_STOP_FAILED` 的失败轨迹移动到 `failed` 子目录：

```powershell
python data_processor\move_failed_trajectories.py "E:\ToolApps\Steam\steamapps\common\Grand Theft Auto V\data\manual" --dry-run
```

确认输出无误后再真正移动：

```powershell
python data_processor\move_failed_trajectories.py "E:\ToolApps\Steam\steamapps\common\Grand Theft Auto V\data\manual"
```

当前 `data_processor.py` 默认读取：

```text
<GTA V>/data/manual/checked
```

所以建议把通过人工审查、且没有被移动到 `failed` 的轨迹整理到：

```text
<GTA V>/data/manual/checked/<session>/
```

注意：`judge_trajectory.py` 目前只负责播放和删除坏轨迹，不会自动把通过审批的轨迹移动到 `checked`。这一步需要手动整理，或者后续再补一个审批移动脚本。

## 转换为 schema v2 训练数据

如果你有 GTA V 端采集出的 `steps.jsonl`、`metadata.jsonl`、RGB/Depth bin 文件，可以用：

```powershell
python data_processor\data_processor.py
```

如果不想改脚本里的默认路径，也可以直接指定输入目录调用函数：

```powershell
python -c "from data_processor.data_processor import process_all_datasets; process_all_datasets(r'E:\ToolApps\Steam\steamapps\common\Grand Theft Auto V\data\manual\checked')"
```

这个步骤会把原始 `.bin` 图像转换成 `.jpg`，并生成 schema v2 JSON。输出默认写入：

```text
dataset/train_data_all.json
dataset/imgs/
```

## awareness 标注

如果要给 `train_data_all.json` 补 awareness，默认可以用本地 VLM 标注脚本：

```powershell
python data_processor\annotate_awareness.py --input_json dataset\train_data_all.json --output_json dataset\train_data_all_with_awareness.json --gpu_ids 0
```

多 GPU：

```powershell
python data_processor\annotate_awareness.py --input_json dataset\train_data_all.json --output_json dataset\train_data_all_with_awareness.json --gpu_ids 0,1,2,3
```

如果使用 OpenAI-compatible 视觉 API，可以使用 API 版本：

```powershell
python data_processor\annotate_awareness_api.py --input_json dataset\train_data_all.json --output_json dataset\train_data_all_with_awareness.json
```

需要配置 API key，例如：

```powershell
$env:DASHSCOPE_API_KEY="你的 key"
```

小规模测试建议先加 `--limit`：

```powershell
python data_processor\annotate_awareness_api.py --input_json dataset\train_data_all.json --output_json dataset\train_data_all_with_awareness.json --limit 10
```

## 切分训练集和验证集

按轨迹切分，避免同一条轨迹的不同帧同时出现在训练和验证中：

```powershell
python data_processor\split_by_trajectory.py --input_json dataset\train_data_all_with_awareness.json --train_output dataset\train_split.json --val_output dataset\val_split.json --manifest_output dataset\split_manifest.json --val_ratio 0.2 --seed 42
```

## Stage1 / Stage2 训练

### 生成 CLIP cache

Stage1 / Stage2 使用缓存特征时，先生成 cache：

```powershell
python agent_control\prepare_clip_cache.py --dataset_json dataset\train_data_all_with_awareness.json --dataset_root dataset --cache_dir dataset\clip_cache --device cuda
```

多 GPU 生成：

```powershell
python agent_control\prepare_clip_cache.py --dataset_json dataset\train_data_all_with_awareness.json --dataset_root dataset --cache_dir dataset\clip_cache --gpu_ids 0,1,2,3 --workers_per_gpu 1
```

### 训练 Stage1

```powershell
python agent_control\train_stage1.py --dataset_json dataset\train_data_all_with_awareness.json --cache_dir dataset\clip_cache --output_dir agent_control\checkpoints\stage1 --device cuda
```

如果已经有固定切分：

```powershell
python agent_control\train_stage1.py --train_json dataset\train_split.json --val_json dataset\val_split.json --cache_dir dataset\clip_cache --output_dir agent_control\checkpoints\stage1 --device cuda
```

### 训练 Stage2

Stage2 需要 Stage1 checkpoint：

```powershell
python agent_control\train_stage2.py --dataset_json dataset\train_data_all_with_awareness.json --dataset_root dataset --cache_dir dataset\clip_cache --stage1_ckpt agent_control\checkpoints\stage1\best.pt --output_dir agent_control\checkpoints\stage2 --device cuda
```

多 GPU：

```powershell
python agent_control\train_stage2.py --dataset_json dataset\train_data_all_with_awareness.json --dataset_root dataset --cache_dir dataset\clip_cache --stage1_ckpt agent_control\checkpoints\stage1\best.pt --output_dir agent_control\checkpoints\stage2 --gpu_ids 0,1,2,3
```

## E2E SMT-GRU 训练

E2E 管线详见：

```text
docs/e2e_smt_gru_pipeline.md
```

最小命令形态：

```powershell
python agent_control\train_e2e_smt_gru.py --dataset_json dataset\train_data_all_with_awareness.json --dataset_root dataset --model_dir agent_control\models\qwen3_vl_sft_merged --output_dir agent_control\checkpoints\e2e_smt_gru --device cuda
```

多 GPU：

```powershell
python agent_control\train_e2e_smt_gru.py --dataset_json dataset\train_data_all_with_awareness.json --dataset_root dataset --model_dir agent_control\models\qwen3_vl_sft_merged --output_dir agent_control\checkpoints\e2e_smt_gru --gpu_ids 0,1,2,3
```

## 在线验证

在线验证需要 GTA V、ScriptHookV、`DroneSim.asi` 插件和 Python 控制端同时可用。

验证脚本默认读取：

```text
data/verification/samples.jsonl
```

这个文件不是训练数据切分出来的帧级 JSON，而是从通过审查的采集 session 的 `metadata.jsonl` 抽样生成的场景级验证集。每一行代表一个验证场景，告诉验证脚本要生成什么异常、从哪个起点开始飞、预期参考步数是多少。

从 `checked` 轨迹生成验证样本：

```powershell
python data_processor\create_validation_set.py --manual_dir "E:\ToolApps\Steam\steamapps\common\Grand Theft Auto V\data\manual\checked" --output_file data\verification\samples.jsonl --ratio 0.2 --seed 42
```

先只生成少量样本做 smoke test：

```powershell
python data_processor\create_validation_set.py --manual_dir "E:\ToolApps\Steam\steamapps\common\Grand Theft Auto V\data\manual\checked" --output_file data\verification\samples.jsonl --ratio 0.2 --seed 42 --limit 5
```

生成逻辑是读取每个 session 的：

```text
<GTA V>/data/manual/checked/<session>/metadata.jsonl
```

并把其中的 `scenario_id`、`anomaly_type`、`anomaly_position`、`start_pose`、`expected_steps`、`task_description` 写入 `samples.jsonl`。每行至少包含：

```json
{
  "scenario_id": "case_001",
  "anomaly_type": "fire",
  "anomaly_position": {"x": 1.0, "y": 2.0, "z": 3.0},
  "start_pose": {"x": 10.0, "y": 20.0, "z": 30.0, "rx": 0.0, "ry": 0.0, "rz": 90.0},
  "expected_steps": 20,
  "task_description": "find the closest burning car"
}
```

直接 VLA 在线验证：

```powershell
python agent_control\run_verification.py --model_dir agent_control\models\qwen3_vl_sft_merged --verification_file data\verification\samples.jsonl --output_file data\verification\results.json
```

Stage2 / E2E 在线验证：

```powershell
python agent_control\run_verification_stage2.py --policy_mode e2e_smt_gru --e2e_ckpt agent_control\checkpoints\e2e_smt_gru\best.pt --model_dir agent_control\models\qwen3_vl_sft_merged --verification_file data\verification\samples.jsonl --output_file data\verification\results_stage2.json
```

如果只想确认流程，先加 `--sample_limit 1`。

## 离线 replay 验证

离线 replay 验证用于没有 GTA V 环境的服务器。它把已有轨迹帧做成 SQLite 数据库，然后在数据库里按位姿和动作 replay。

coverage miss 的默认判定是：最近数据库状态与目标位姿的 3D 欧氏距离 `xyz < 5m`，且朝向差 `yaw < 15deg`。可用 `--xyz_threshold` 和 `--yaw_threshold` 调整。

### 1. 构建 replay DB

如果服务器只传一个 DB，使用 `--store_images` 把图片打包进去：

```powershell
python agent_control\offline_replay_db.py --dataset_json dataset\train_data_all_with_awareness.json --dataset_root dataset --db_path data\verification\offline_replay.sqlite --store_images --overwrite
```

如果服务器也有 `dataset/imgs`，可以不打包图片：

```powershell
python agent_control\offline_replay_db.py --dataset_json dataset\train_data_all_with_awareness.json --dataset_root dataset --db_path data\verification\offline_replay.sqlite --overwrite
```

### 2. 运行离线验证

DB 内已打包图片：

```powershell
python agent_control\run_offline_verification.py --db_path data\verification\offline_replay.sqlite --verification_file data\verification\samples.jsonl --output_file data\verification\results_offline.json --misses_file data\verification\coverage_misses.jsonl
```

DB 未打包图片：

```powershell
python agent_control\run_offline_verification.py --db_path data\verification\offline_replay.sqlite --dataset_root dataset --verification_file data\verification\samples.jsonl --output_file data\verification\results_offline.json --misses_file data\verification\coverage_misses.jsonl
```

小规模测试：

```powershell
python agent_control\run_offline_verification.py --db_path data\verification\offline_replay.sqlite --dataset_root dataset --verification_file data\verification\samples.jsonl --output_file data\verification\results_offline.json --misses_file data\verification\coverage_misses.jsonl --sample_limit 5
```

输出文件：

- `results_offline.json`：离线验证结果、宽松成功率、严格成功率、覆盖统计。
- `coverage_misses.jsonl`：数据库缺覆盖的目标位姿，用于回到 GTA V 本地补采。

结果中：

- `success` 表示允许跳到最近数据库状态后的宽松结果。
- `strict_success` 表示全程没有 coverage miss 的严格结果。

## 常见注意事项

- 终端中文显示异常时，先确认文件本身是 UTF-8，不要直接判断文件损坏。
- `dataset/`、`data/`、模型权重和 checkpoints 默认不进 git，新机器需要单独同步。
- 在线验证依赖 GTA V 插件运行；离线 replay 验证不依赖 GTA V，但依赖已有轨迹覆盖。
- 训练脚本默认路径可能指向 `dataset/train_data_all_with_awareness.json`，如果你的文件名不同，需要显式传参。
- 大规模训练或验证前，先用 `--sample_limit`、`--limit` 或少量 epoch 做 smoke test。
- `coverage_misses.jsonl` 不是最终失败列表，而是下一轮补采的候选位姿列表。
