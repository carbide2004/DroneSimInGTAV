# E2E SMT-GRU Training Pipeline

## Goal

The end-to-end training pipeline learns a policy built from an SMT-style observation encoder, a GRU memory module, and a VLM soft-token controller. The model uses RGB-D observations, camera poses, previous actions, task text, and awareness annotations to optimize navigation decisions.

In one sentence: RGB-D trajectory observations are encoded into an SMT-GRU historical representation, projected into soft tokens, and injected into a Qwen VLM so the final action decision is conditioned on both current visual input and learned trajectory memory.

## Data Input

Training uses `train_data_all_with_awareness.json`. Each trajectory is grouped by `trajectory_id` and contains multiple ordered steps:

```text
Trajectory
├── RGB image
├── Depth image
├── Pose: x, y, z, rx, ry, rz
├── Action label
├── Task description
└── Awareness text
```

The dataset is split by trajectory into train and validation subsets. `--max_trajectory_len` filters out whole trajectories longer than the specified threshold before the split.

## Observation Encoding

Each RGB and depth pair is loaded from disk and resized to `image_size`. RGB and depth are concatenated as a 4-channel tensor:

```text
RGB + Depth
  -> 4-channel RGBD tensor
  -> CNN Visual Encoder
  -> visual embedding
```

Pose and previous action are embedded separately:

```text
Pose / relative pose -> pose embedding
Previous action -> action embedding
```

The three streams are concatenated and projected:

```text
[visual embedding, pose embedding, previous-action embedding]
  -> obs_fc
  -> observation token sequence
```

## SMT Memory Module

For each timestep `t`, the model builds a memory from all observations up to the current step:

```text
Observation tokens up to t
  + relative pose encoding
  -> memory encoder attention
  -> memory decoder attention
  -> smt_context_t
```

The output over the trajectory is:

```text
smt_seq = [smt_context_1, smt_context_2, ..., smt_context_T]
```

## GRU Memory Module

The SMT sequence is fed into the GRU:

```text
smt_seq
  -> Stage1GRUModel
  -> h_seq
  -> gru_action_logits
```

The GRU is the core historical-memory component. It produces both action logits and hidden states used for awareness alignment and VLM soft-token conditioning.

## VLM Soft-Token Injection

For sampled training steps:

```text
h_t + smt_context_t
  -> Stage2BridgeModel
  -> soft prompt tokens
```

The soft prompt is prepended to Qwen VLM token embeddings:

```text
soft prompt + textual prompt + RGB image + Depth image
  -> Qwen VLM with LoRA
  -> action CE loss
```

The VLM prompt contains the task, current pose, two image placeholders, the action set, and the instruction to output one valid action.

## Losses

The total training objective is:

```text
L_total =
  lambda_vlm_action * L_vlm_action
+ lambda_gru_action * L_gru_action
+ lambda_awareness * L_awareness
```

`L_vlm_action` is the VLM action cross-entropy loss on the ground-truth action text.

`L_gru_action` is the GRU action-head cross-entropy loss over valid trajectory steps.

`L_awareness` aligns GRU hidden states with awareness text embeddings using InfoNCE. Awareness text is not directly inserted into the VLM prompt; it supervises the trajectory-memory representation.

## Trainable And Frozen Components

Trainable:

```text
RGBD Visual Encoder
SMT memory encoder / decoder
GRU memory module
GRU action head
Awareness projection heads
Stage2Bridge soft-token projector
Qwen LoRA adapters
```

Frozen:

```text
Qwen base weights except LoRA
Text encoder used for awareness embedding
```

## Multi-GPU Training

`--gpu_ids` launches DDP with one process per selected GPU:

```text
GPU 0 -> rank 0
GPU 1 -> rank 1
...
```

Each rank receives different trajectory batches. Rank 0 saves:

```text
best.pt
last.pt
lora_best/
lora_last/
config.json
history.json
```

## Architecture Diagram Skeleton

```text
Dataset: RGBD / Pose / Action / Awareness
        |
        v
RGBD Visual Encoder -----
Pose Encoder ------------|--> Observation Token
Prev Action Embedding ---|
        |
        v
SMT Memory Encoder / Decoder
        |
        v
GRU Memory Module
   |          |
   |          +--> GRU Action Head --> L_gru_action
   |
   +--> Awareness Projection <--> Text Encoder(Awareness) --> L_awareness
   |
   v
Stage2 Bridge
        |
        v
Soft Tokens + Prompt + RGB/Depth
        |
        v
Qwen VLM + LoRA
        |
        v
Action CE Loss: L_vlm_action
```
