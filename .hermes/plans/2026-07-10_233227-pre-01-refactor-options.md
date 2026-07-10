# DroneSimInGTAV Pre-01 Refactor Options

> **For Hermes:** Do not implement core source changes until the user selects Option A, B, or C. Once selected, turn the chosen option into a task-by-task TDD implementation plan and execute in an isolated branch/worktree.

**Goal:** 在进行“01-GTAV 响应生态验证”前，建立一个可理解、可测试、可迁移且不会继续积累版本漂移的项目基线。

**Architecture:** 当前仓库实际由 GTA V C++ 插件、Python 仿真客户端/在线控制、数据加工与标注、Stage1/Stage2/E2E 模型实验、离线 replay 和可视化脚本组成，但边界主要靠文件名和 README 约定维持。本计划提供三种重构强度；三者都先建立行为基线，再逐层迁移，避免“边整理边改变科研结论”。

**Tech Stack:** C++17 / Visual Studio 2022 / ScriptHookV / Boost.Asio / Python 3.10 / PyTorch / Transformers / SQLite / JSON(JSONL)

---

## 1. 审计范围与已验证状态

- Revision: `7837644216a0940038c11995ee9317ba857c1f02` (`main`, 与 `origin/main` 同步)
- Working tree: clean
- 最近提交: `feat: 实体清理逻辑 & 缺失帧补采脚本`
- 当前分支/远端: 仅 `main` / `origin/main`，无 tag
- 仓库主体: 40 个 Python 文件、9 个 C++ 源文件、34 个头文件，另有 vendored `.lib` 和 SDK headers
- Python 静态语法检查: 40/40 通过
- dependency-light `--help` smoke:
  - `split_by_trajectory.py`: pass
  - `create_validation_set.py`: pass
  - `offline_replay_db.py`: pass
  - `run_offline_verification.py`: 当前环境因缺少 numpy 无法 import；这不是已确认代码缺陷，但说明 CLI/import 边界较重
- 未验证: Windows/MSBuild 构建、GTA V/ScriptHookV 插件运行、GPU 训练、在线协议端到端

## 2. 当前真实架构

```text
GTA V + ScriptHookV
  -> DroneSim/script.cpp
       场景生成 + 自动/手动采集 + 录制 + 玩家/相机控制 + 命令执行
  -> command_queue
  <- DroneSim/server_v2.cpp
       DSV2 TCP 协议适配
  <-> agent_control/dronesim_client.py
       Python 协议客户端

原始轨迹 steps.jsonl + metadata.jsonl + RGB/Depth .bin
  -> data_processor/data_processor.py
  -> schema_version=2 frame JSON + JPG
  -> awareness 标注 / trajectory split / CLIP cache
  -> Stage1 GRU / Stage2 soft prompt / E2E SMT-GRU
  -> 在线验证 或 SQLite offline replay
```

核心问题不是“文件太多”，而是**实验代际、生产管线、平台适配和协议实现都挤在同一层级，缺少明确的 canonical path 与兼容边界**。

## 3. 高优先级审计结论

### P0/P1 correctness 与可重复性

1. `agent_control/verification_runtime.py:79-85` 的 `build_movement_params()` 把 `up_step` 错误读取为 `args.down_step`。当上下移动步长不一致时，在线和离线评测行为会静默错误。
2. `agent_control/verification_runtime.py:131-149` 中 pose/capture/图像处理失败不会增加 step，也没有总 wall-clock deadline，持续失败时循环可能永久不退出。
3. `agent_control/verification_runtime.py:137-184` 的 `final_pose` 在动作执行前更新；达到最大步数时，final distance 可能使用最后动作前的位姿。
4. `agent_control/verification_runtime.py:154` 在动作解析失败时静默回退 `AUTO_FORWARD`。这会把模型/协议失败伪装成策略动作，污染成功率和路径指标。
5. `DroneSim/server_v2.cpp:43-45` 实际绑定 `0.0.0.0:<port>`，但 README 与 Python 默认值写成 `127.0.0.5:23456`。接口无认证；当前服务并非文档暗示的 loopback-only。
6. `DroneSim/proto.h:10` 和 Python `dronesim_client.py:13` 声明 PING，但 server switch 中未见实现；协议能力与实际实现不一致。
7. Python 客户端 `dronesim_client.py:45-59` 没有 connect/read timeout，也没有校验响应 magic/version/type/request_id/length；所有命令复用固定 request_id。
8. C++ server 大量失败路径仍返回正常 type + 空 payload，缺少结构化 status/error。客户端无法区分合法空结果、超时、错误 payload 和未知消息。

### P1 架构与实验代际漂移

9. 设计文档 `docs/潜状态GRU与VLA训练推理方案.md` 定稿为 DINOv2 768-d；当前活跃 cache 是 tiled CLIP heatmap + depth + grayscale，feature dim 为 `heatmap_size²*3`。论文/设计文档已与实现明显漂移。
10. `prepare_clip_cache.py` 默认 `heatmap_size=24`，而 `run_verification_stage2.py` 默认 `48`，在线端通过 checkpoint 输入维度推断并自动修正。这是“运行时补救配置漂移”，不是稳定契约。
11. `prepare_clip_cache.py:63-86,119-228` 与 `run_verification_stage2.py:130-227` 复制了 tile、resize、normalization、CLIP 提取逻辑，训练/推理预处理存在继续漂移风险。
12. Stage1/Stage2 与 E2E 是两套部分重复的数据、split、文本编码、InfoNCE、DDP 和 checkpoint 逻辑：`trajectory_dataset.py` 已有通用实现，但 `train_e2e_smt_gru.py` 又内嵌一套 trajectory dataset/split/collate。
13. 模型默认路径存在多个历史命名：`qwen3_vl_sft_merged`、`qwen3_vl_sft_GTAV_20260403`、`qwen3_vl_sft_GTAV_20260509`。没有统一 experiment config 能说明哪个是 canonical model。
14. `run_verification_stage2.py` 约 1000 行，同时承担配置解析、CLIP 特征、模型加载、在线控制、指标、profiling 和多策略分支，是典型“实验汇合点”。
15. `DroneSim/script.cpp` 超过 1300 行，混合场景生成、实体生命周期、采集、metadata schema、键盘 UI、自动策略和网络命令执行；它也是全历史修改次数最高文件（105 次），是主要冲突热点。

### P1/P2 数据与仓库治理

16. schema v2 只在 Python loader 中手写验证，没有 JSON Schema、schema 包、迁移器和 fixture；`data_processor.py` 同时负责 bin 解码、深度可视化、prompt 生成、schema 构造和文件落盘。
17. depth 原始 float 在 `data_processor.py:54-60` 被每帧 min-max + JET 转成 JPG；此输出同时承担“可视化图”和“模型输入”，物理深度语义已经丢失且不可逆。
18. task/prompt/action 常量在多个文件重复；数据 JSON 同时保留结构化字段、`images` legacy alias 和完整 `messages`，缺少字段权威性说明。
19. 没有 `tests/`、CI、lint/type-check 配置，也没有协议、schema、split leakage、RGBD parity 或 replay fixture 测试。
20. Python requirements 只有宽松下限或无版本 pin；C++ 使用 NuGet packages + vendored ScriptHookV/DirectXTK/MinHook libs，但缺少来源、版本、hash 和许可证清单。
21. tracked `DroneSim/DroneSim.vcxproj.user` 属于用户级 IDE 文件；`.gitignore` 未覆盖 `*.pyc`、`.pytest_cache` 等常见生成物（当前 pycache 未跟踪，但工作目录存在）。
22. Git 历史有大量 `wip`/`temp`/`有问题的完整版` 类提交；仓库没有 tag/release，也没有明确记录“当前数据 schema、插件协议、模型管线”三类版本。
23. README 563 行把安装、架构、数据操作、训练、在线验证、离线 replay 全部集中在一页，已接近运维手册而非入口文档。

## 3.1 并行深度审计补充：必须纳入决策的发现

三路独立审计（C++、Python 模型管线、数据/仓库治理）完成后，新增确认了以下高优先级问题：

1. **Capture 可能返回旧帧。** `server_v2.cpp:562-590` 在采集启动或完成超时后仍继续读取 snapshot；`export.cpp:315-332` 只判断缓存非空，没有 frame sequence/timestamp；响应也不携带 frame ID。任务 01 若要分析实体动态响应，旧帧会直接破坏速度、方向和响应延迟统计。
2. **Stage2 存在确定性的训练/推理归一化偏差。** cache 生成在 `prepare_clip_cache.py:83-86,212-225` 使用 `clip(logit/20,0,1)`；在线端在 `run_verification_stage2.py:146-151,216-230` 使用每帧 min-max。仅检查 feature dimension 无法发现该语义偏差。
3. **验证 step 抽样不稳定。** Stage2/E2E 训练和评估共享随机 `_select_steps`，验证 epoch 不重置独立 RNG；`best.pt` 可能由不同 timestep 子集决定。E2E 还不能直接消费 Stage1/Stage2 的固定 split manifest。
4. **E2E 推理忽略 checkpoint 中保存的 image size。** loader 返回 payload，但调用方丢弃，在线预处理使用独立 CLI 默认值；adaptive pooling 会使错误静默发生而不是 shape mismatch。
5. **Cache 增量复用会伪造 metadata。** `prepare_clip_cache.py:330-354` 在 `.npy` 已存在时不重算，却使用当前参数覆盖 manifest metadata；同一 cache 目录可混入不同 backbone/window/normalization 的特征。
6. **LoRA 加载失败会降级后继续评估。** Stage2/E2E 在线和 offline evaluator 捕获异常后继续，结果仍可能被标为 Stage2/E2E，实际却运行未适配 base VLA。
7. **C++ 网络线程可被半包永久阻塞。** `server_v2.cpp:58-68` 是无 deadline 的同步 `read_some`，且唯一 accept/io 线程直接进入阻塞式 `handle_client`；一个不完整请求可以阻断后续所有客户端。
8. **Capture 响应丢失 depth 独立尺寸。** 服务端内部有 RGB/depth 两组尺寸，但 wire payload 只发送 `w_rgb/h_rgb`；Python 用同一尺寸 reshape 两者。
9. **无参 stop 会生成没有 metadata 的 session。** `script.cpp:196-205` 的兼容 overload 只关闭录制文件；F7、F11、STOP_CAMERA 会走该路径，留下 `steps.jsonl` 但没有 `metadata.jsonl`。
10. **人工审查包含不可逆自动删除。** `judge_trajectory.py` 对缺 metadata、图像异常或按 D 的轨迹直接 `shutil.rmtree`，没有 quarantine、确认、manifest 或审计日志。
11. **action schema 不完整。** `AUTO_STOP_FAILED` 不在训练动作枚举内；若失败轨迹未提前移走，转换器会写入 action name 但令 `action_id=None`，产生无效标签而不报错。
12. **干净 clone 构建链没有被证明可用。** VS 工程强制依赖未提交的 NuGet targets；四种配置不等价，只有 Release x64 明确具备完整插件链接配置；Debug/Release x64 还关闭了全部编译警告。

这些结果使“只做目录整理”的 Option A 风险高于最初估计。特别是 stale capture 与 train/inference skew 都会直接污染后续科研结论。

## 4. Option A — 保守整理

### 适用

- 希望尽快开始任务 01；
- 不想现在触碰 C++ 场景/协议核心；
- 接受 Stage1/Stage2/E2E 暂时共存；
- 目标是先让仓库“可运行、可定位、少踩坑”。

### 范围

1. 建立 baseline 分支与 pre-refactor tag（不改历史）。
2. 清理仓库卫生：移除 tracked `.vcxproj.user`，扩充 `.gitignore`，添加依赖/第三方清单。
3. 把 README 拆成入口 + `docs/setup/`、`docs/data/`、`docs/training/`、`docs/evaluation/`。
4. 增加 `configs/` 示例，统一 host/port/path/model/checkpoint 默认值；移除脚本中的个人绝对路径。
5. 修复上述确定性 correctness bugs，但保持 CLI 和输出文件兼容。
6. 增加最小 `tests/`：action mapping、movement config、schema v2 loader、trajectory split、protocol header、tiny replay DB。
7. 明确方法状态矩阵：`legacy / active baseline / experimental / deprecated`，不删模型文件。
8. 给当前 schema v2 增加 validator，但暂不重塑字段。

### 不做

- 不拆 `script.cpp`；
- 不改变 DSV2 协议；
- 不合并 Stage1/E2E 训练框架；
- 不改变数据文件布局或深度表示；
- 不建立新 Python package。

### 预计迁移风险

低。主要风险是默认值统一后暴露过去依赖隐式路径的脚本。

### 对任务 01 的价值

足以让 01 在一个干净基线上开展，但 01 增加实体观测/事件关联字段时仍会继续往 `script.cpp`、协议和 schema v2 上堆功能。

## 5. Option B — 分层重构（推荐）

### 适用

- 希望 01 之后持续做 benchmark/schema/baseline，而不是一次性 demo；
- 希望控制改动风险，暂不重写 C++ 协议；
- 需要让实验代际和生产代码分离。

### 范围

包含 Option A，并增加：

1. 建立 Python package（名称可选 `dronesim` 或 `drone_event_nav`），按领域拆分：
   - `protocol/`: DSV2 codec + client
   - `data/`: schema/models/io/validation/split
   - `features/`: shared RGBD/CLIP preprocessing
   - `models/`: stage1/smt/bridge/policies
   - `evaluation/`: online runtime/offline replay/metrics
   - `scenarios/`: verification sample contract
2. 保留现有脚本名作为薄 CLI wrapper，先发 deprecation warning，避免已有命令立即失效。
3. 把 CLIP tiled feature extraction做成唯一共享实现；cache 与 online inference 从同一 config/代码生成 feature signature。
4. 合并 trajectory dataset/split/collate；Stage1、Stage2、E2E 复用同一数据层。
5. 建立 experiment config + run manifest：代码 revision、dataset manifest/hash、schema version、feature signature、model revision、seed、movement config、success definition。
6. 将 `run_verification_stage2.py` 拆成 policy adapter、feature extractor、runtime loop、metrics、CLI。
7. schema v2 进入 `legacy/v2`，定义供任务 01 使用的新 event/track schema 的扩展点；但不在本轮替用户提前决定完整 v3 内容。
8. C++ 保持现有对外消息号和行为；只做不改变协议的内部 helper 提取和 correctness cleanup。
9. 增加 pytest + Windows C++ build workflow（如 GitHub Actions 可取得依赖）或至少 MSBuild 文档化 smoke script。

### 不做

- 不改变 DSV2 wire format/message IDs；
- 不重写 GTA V 场景行为；
- 不一次性删除 Stage1/Stage2/E2E 任一路径；
- 不改变现有数据集，先通过 adapter 读取。

### 推荐目标结构

```text
src/drone_event_nav/
  protocol/
  data/
  features/
  models/
  evaluation/
  scenarios/
cli/
  prepare_dataset.py
  annotate_awareness.py
  train.py
  evaluate_online.py
  evaluate_offline.py
legacy/
  README.md
configs/
tests/
DroneSim/
```

### 预计迁移风险

中等。风险主要来自 import path、checkpoint loader 和 CLI wrapper；可通过 characterization tests 和双路径对比控制。

### 对任务 01 的价值

高。01 可以把“响应实体采集”作为清晰的新 schema/protocol extension 接入，而不是继续扩大现有脚本耦合。

## 6. Option C — 全面重构

### 适用

- 已决定把项目建设成长期论文 benchmark/platform；
- 可以安排 Windows + GTA V 实机回归；
- 接受协议和数据格式双版本迁移；
- 愿意冻结旧实验，并明确新研究主线。

### 范围

包含 Option B，并增加：

1. 拆分 `DroneSim/script.cpp`：
   - `scene/` event generators + entity registry
   - `recording/` recorder + metadata writer
   - `control/` camera/player/action execution
   - `runtime/` ScriptHook loop + key bindings
2. DSV2 升级为显式协议版本（或 DSV3）：
   - loopback bind/configurable bind
   - unique request IDs
   - status/error codes
   - payload codecs + strict length/type/version validation
   - timeouts/health check/capabilities
   - entity snapshot / event association / lifecycle messages，为任务 01 直接服务
3. 新数据契约：raw event/entity/frame/trajectory 分层；原始 metric depth 与可视化 depth 分离；schema version + migration + JSON Schema。
4. 场景生成配置化：事件类型、seed、spawn params、responders、cleanup ownership、时间天气统一配置，而不是改 `script.cpp` 重新编译。
5. 将旧 Stage1/Stage2/E2E 标为 archived experiments；研究主线从统一 policy interface 启动。
6. 建立 Windows GTA V contract test checklist 和录制 golden fixtures；Linux 侧用 mock server/replay 做协议和数据测试。
7. 正式 release/tag：legacy baseline、protocol v2 compatibility、new benchmark foundation。

### 兼容策略

- Python 同时支持 DSV2 和新协议一段迁移期；
- schema v2 只读 adapter；
- 旧 checkpoint 可评估但不再作为新训练默认入口；
- old CLI wrappers 保留一个 release cycle。

### 预计迁移风险

高。没有 GTA V Windows 实机回归就不能宣称完成；协议、场景清理和 capture threading 都可能产生平台特有问题。

### 对任务 01 的价值

最高。任务 01 所需 entity ID/type/position/velocity/heading/task/event association 可以成为平台的一等数据，而非临时日志字段。

## 7. 三方案对比

| 维度 | A 保守 | B 分层 | C 全面 |
|---|---:|---:|---:|
| 开始 01 的速度 | 最快 | 中等 | 最慢 |
| 改动风险 | 低 | 中 | 高 |
| 保留现有命令/数据 | 高 | 高（wrapper/adapter） | 中（双版本迁移） |
| 解决 Python 实验混乱 | 部分 | 是 | 是 |
| 解决 C++ `script.cpp` 耦合 | 否 | 仅轻量 | 是 |
| 解决协议脆弱性 | 仅修 bug | 测试/封装，不改 wire | 是 |
| 支撑 01 实体响应采集 | 临时扩展 | 良好扩展点 | 原生支持 |
| 需要 Windows/GTA V 回归 | 少量 | 中等 | 必须且广泛 |

## 8. 推荐决策

推荐 **Option B**，并把 C 中的两项提前纳入 B 的接口设计但不立即实现：

1. 为未来 entity/event snapshot 预留清晰 schema/protocol extension point；
2. 把场景生成参数从键盘硬编码迁往配置对象的方向先固定。

理由：A 会很快，但任务 01 马上会再次扩大 `script.cpp`、DSV2 和 schema v2；C 最符合长期方向，但在未先做 response ecology audit 前，可能为尚未验证的研究假设过度建设。B 能先把结构整理好，同时把是否升级 C 留给 01 的实证结果。

## 9. 无论选哪项都应遵守的迁移顺序

1. 创建隔离分支/worktree；记录 baseline commit/tag。
2. 添加 characterization tests，先锁定 action、schema、protocol、split、replay 当前行为。
3. 修确定性 correctness bugs；分别提交。
4. 整理配置、路径、仓库卫生和文档。
5. 再做结构移动；每次迁移保持 wrapper/adapter。
6. 对训练/推理 feature 用同一 fixture 做字节/数组级 parity test。
7. Linux 测试通过后，在 Windows 构建插件。
8. 对 GTA V 做最小 contract run：camera/create scene/capture/move/cleanup。
9. 产出 migration guide 和 deprecation matrix。
10. 才开始任务 01 的新能力开发。

## 10. 需要用户决定的事项

- 选择 A/B/C。
- 若选 B：新包名使用 `dronesim` 还是 `drone_event_nav`。
- 现有 Stage1/Stage2/E2E 是否全部保留为可运行实验，还是把其中某些直接归档。
- 是否允许建立新协议版本；这只在 C 中立即执行。
- Windows GTA V 机器是否可用于每个迁移批次后的实机验证。
