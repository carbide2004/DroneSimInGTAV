# Option B Layered Refactor Implementation Plan

> **For Hermes:** Execute task-by-task with strict TDD. Core source changes remain scoped to the user-selected Option B: preserve DSV2 wire compatibility, existing CLI names, existing schema-v2 readability, and existing experiment checkpoints where practical.

**Goal:** 在不重写 DSV2 wire protocol 和 GTA V 场景系统的前提下，先修复会污染实验的确定性问题，再建立可测试的 `drone_event_nav` 分层 Python 包和旧入口兼容层，为任务 01 提供稳定扩展点。

**Architecture:** 使用纵向迁移而不是一次性搬家。先为现有行为建立 dependency-light tests，修 correctness，再把动作/配置、数据契约、特征预处理、评测运行时按领域提取到 `src/drone_event_nav/`。旧 `agent_control/` 和 `data_processor/` 脚本继续作为 wrapper，避免立即破坏命令和 checkpoint。

**Tech Stack:** Python 3.10+, unittest/pytest-compatible tests, JSON/JSONL, SQLite, existing PyTorch/Transformers code; C++17/DSV2 remains wire-compatible in this phase.

---

## Constraints

- Baseline commit: `7837644216a0940038c11995ee9317ba857c1f02`
- Working branch: `refactor/option-b-foundation`
- 不修改 DSV2 message IDs/header layout。
- 不删除 Stage1/Stage2/E2E 路径；先标记和复用公共层。
- 不要求 Linux 环境安装大型 ML 依赖才能运行核心单元测试。
- 每个生产代码变更必须先有失败测试。
- 旧 CLI 至少保留一个迁移周期。
- Windows/GTA V 专属行为只能静态验证；最终需实机 contract test。

## Proposed target structure

```text
src/drone_event_nav/
  __init__.py
  config.py
  actions.py
  protocol/
    __init__.py
    dsv2.py
  data/
    __init__.py
    schema_v2.py
    io.py
    split.py
  features/
    __init__.py
    clip_heatdepth.py
    signature.py
  evaluation/
    __init__.py
    control_loop.py
    metrics.py
  scenarios/
    __init__.py
    verification.py
  models/
    __init__.py
    registry.py
    checkpoint_manifest.py

tests/
  unit/
  fixtures/
```

## Task 1 — Test harness and import boundary

**Files:**
- Create: `pyproject.toml`
- Create: `src/drone_event_nav/__init__.py`
- Create: `tests/test_import_boundary.py`
- Modify: `.gitignore`

**RED:** test that `drone_event_nav` imports without torch/numpy/transformers and exposes a version string.

**GREEN:** minimal package metadata and import path configuration. Use standard-library-only package root.

**Verify:**

```bash
python3 -m unittest tests.test_import_boundary -v
python3 -m compileall -q src tests
```

**Commit:** `chore: establish refactor package and test harness`

## Task 2 — Canonical action contract

**Files:**
- Create: `src/drone_event_nav/actions.py`
- Create: `tests/test_actions.py`
- Modify: `agent_control/action_mapping.py`
- Modify later wrappers in `data_processor/data_processor.py` and stats code only after parity tests

**RED:** tests for canonical six policy actions, explicit non-policy terminal outcomes, parsing, dispatch signs, and invalid actions.

**GREEN:** extract `POLICY_ACTIONS`, `TERMINAL_OUTCOMES`, parser and movement delta definitions. Existing module re-exports/wraps them.

**Verify:** old `parse_action` examples and new tests both pass.

**Commit:** `refactor: centralize action contract`

## Task 3 — Movement configuration correctness

**Files:**
- Create: `src/drone_event_nav/config.py`
- Create: `tests/test_movement_config.py`
- Modify: `agent_control/verification_runtime.py`

**RED:** distinct `up_step=7` and `down_step=3` must remain distinct; invalid non-positive or non-finite values are rejected.

**GREEN:** `MovementConfig` dataclass and compatibility adapter from argparse namespace. Fix current up/down bug.

**Verify:** targeted test, then full suite.

**Commit:** `fix: preserve independent movement step configuration`

## Task 4 — Bounded control-loop failures and final pose semantics

**Files:**
- Create: `src/drone_event_nav/evaluation/control_loop.py`
- Create: `tests/test_control_loop.py`
- Modify: `agent_control/verification_runtime.py`

**RED vertical slices:**
1. repeated pose failure terminates after configured failure budget;
2. repeated capture failure terminates;
3. invalid model output is recorded as policy error, not silently converted to forward;
4. final pose is sampled after the final dispatched action;
5. cleanup callback executes on exception.

**GREEN:** dependency-injected control-loop core using fake client/model in tests. Existing runtime becomes adapter.

**Compatibility:** add explicit opt-in legacy fallback if needed, but default fail-closed for scientific evaluation.

**Commit:** `fix: make verification loop bounded and observable`

## Task 5 — Protocol codec characterization without wire change

**Files:**
- Create: `src/drone_event_nav/protocol/dsv2.py`
- Create: `tests/test_dsv2_codec.py`
- Modify: `agent_control/dronesim_client.py`

**RED:** golden 20-byte header bytes; reject wrong magic/version/type/request ID/oversized payload; socket timeout is configured; request IDs are monotonic.

**GREEN:** shared Python DSV2 codec and hardened client while preserving existing message IDs/layout and one-request-per-connection behavior.

**Note:** C++ protocol changes beyond strict version check/PING/timeout are deferred to a later Windows-validated batch because Option B preserves wire compatibility.

**Commit:** `refactor: harden DSV2 client codec`

## Task 6 — Schema-v2 canonical adapter and action validation

**Files:**
- Create: `src/drone_event_nav/data/schema_v2.py`
- Create: `tests/fixtures/schema_v2_minimal.json`
- Create: `tests/test_schema_v2.py`
- Modify: `agent_control/trajectory_dataset.py`
- Modify: `data_processor/data_processor.py`

**RED:** valid sample accepted; missing version rejected; duplicate representations checked for consistency; `AUTO_STOP_FAILED` rejected as trainable policy label or represented as trajectory outcome, never `action_id=None`.

**GREEN:** standard-library validator/model functions and compatibility adapters.

**Commit:** `refactor: define canonical schema v2 contract`

## Task 7 — Safe trajectory review workflow

**Files:**
- Create: `src/drone_event_nav/data/review.py`
- Create: `tests/test_review_quarantine.py`
- Modify: `data_processor/judge_trajectory.py`

**RED:** rejected/corrupt sessions move to quarantine with manifest; no default permanent deletion; collisions produce unique targets; dry-run has no side effects.

**GREEN:** quarantine service and viewer integration. Permanent delete requires explicit CLI flag and confirmation outside normal key flow.

**Commit:** `fix: quarantine rejected trajectories instead of deleting`

## Task 8 — Shared CLIP heat-depth preprocessing

**Files:**
- Create: `src/drone_event_nav/features/clip_heatdepth.py`
- Create: `src/drone_event_nav/features/signature.py`
- Create: `tests/test_clip_heatdepth_preprocessing.py`
- Modify: `agent_control/prepare_clip_cache.py`
- Modify: `agent_control/run_verification_stage2.py`

**RED:** offline and online array-only postprocessing produce exactly equal features for identical heatmap/depth/RGB inputs; fixed-scale normalization is canonical; signature changes when any semantic parameter changes.

**GREEN:** extract dependency-light numerical transformation helpers; model loading remains in wrappers. Online and cache paths call the same code.

**Commit:** `fix: unify Stage2 train and inference preprocessing`

## Task 9 — Immutable cache manifest fingerprint

**Files:**
- Create: `tests/test_feature_manifest.py`
- Modify: `src/drone_event_nav/features/signature.py`
- Modify: `agent_control/prepare_clip_cache.py`
- Modify: `agent_control/train_stage1.py`

**RED:** configuration mismatch refuses incremental reuse; existing item metadata cannot be relabeled without recomputation; checkpoint stores feature signature.

**GREEN:** top-level manifest schema/version/fingerprint and validation.

**Commit:** `fix: prevent mixed feature caches`

## Task 10 — Deterministic split and evaluation-step selection

**Files:**
- Create: `src/drone_event_nav/data/split.py`
- Create: `tests/test_split_and_sampling.py`
- Modify: `agent_control/trajectory_dataset.py`
- Modify: `agent_control/train_stage2.py`
- Modify: `agent_control/evaluate_stage2_offline.py`
- Modify: `agent_control/train_e2e_smt_gru.py`

**RED:** same manifest produces identical IDs; train/val disjoint; validation step selection deterministic and independent of training RNG; E2E can consume fixed manifest/train+val JSON.

**Commit:** `fix: make experiment splits and validation deterministic`

## Task 11 — Checkpoint/run manifest and fail-closed model loading

**Files:**
- Create: `src/drone_event_nav/models/checkpoint_manifest.py`
- Create: `src/drone_event_nav/models/registry.py`
- Create: `tests/test_checkpoint_manifest.py`
- Modify: Stage1/Stage2/E2E train/eval loaders

**RED:** manifest requires code revision, schema, split, feature signature, model identifier, preprocessing and seed; mismatched image size rejected unless explicit override; missing LoRA fails by default.

**Commit:** `refactor: version experiment and checkpoint metadata`

## Task 12 — CLI/config consolidation with compatibility wrappers

**Files:**
- Create: `configs/example.toml`
- Create: `src/drone_event_nav/cli.py`
- Modify: selected old entry scripts into thin wrappers
- Create: `docs/migration/option_b.md`

**RED:** config precedence defaults < config file < CLI; old command imports/calls new handler; no personal absolute paths are required.

**Commit:** `refactor: add unified configuration and compatibility CLIs`

## Task 13 — Repository hygiene and documentation split

**Files:**
- Modify: `.gitignore`
- Remove from tracking: `DroneSim/DroneSim.vcxproj.user`
- Create: `THIRD_PARTY.md`
- Create: `docs/architecture.md`
- Create: `docs/method-status.md`
- Split README links to setup/data/training/evaluation docs

**Verify:** no generated/cache/user files tracked; docs distinguish verified implementation from historical design.

**Commit:** `docs: document architecture dependencies and method status`

## Task 14 — C++ compatibility stabilization batch

**Prerequisite:** Windows/GTA V test machine available.

**Files likely:**
- `DroneSim/server_v2.cpp/h`
- `DroneSim/proto.h`
- `DroneSim/script.cpp`
- `DroneSim/export.cpp/h`
- C++ protocol fixture/test project or standalone codec test

**Scope under Option B:**
- strict `version == 1` check;
- implement PING without changing header layout;
- explicit loopback bind by default;
- bounded read/session timeout;
- stop/capture errors become detectable within compatible response rules where possible;
- prevent stale snapshot via internal sequence tracking;
- preserve current message IDs and payloads for successful responses.

**Do not claim complete until:** Release x64 builds and GTA contract run passes camera/create/capture/move/cleanup.

## Verification gates

After every task:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src agent_control data_processor visualize tests
```

When ML dependencies are available:

```bash
pytest -q
# targeted tensor/checkpoint smoke tests
```

Before each commit with 2+ edited files:

- inspect diff;
- static secret/shell/eval scan;
- independent reviewer subagent;
- only commit after review passes.

## Immediate execution batch

This session should implement Tasks 1–4 first. They are dependency-light, directly fix confirmed evaluation correctness problems, and establish the package/test foundation without forcing premature C++ or ML architecture changes.
