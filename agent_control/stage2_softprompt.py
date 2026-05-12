from typing import Dict, List, Optional, Tuple


def _sync_cuda_for_profile(enabled: bool):
    if not enabled:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class _ProfileStage:
    def __init__(self, profile, name: str, sync_cuda: bool):
        self.profile = profile
        self.name = str(name)
        self.sync_cuda = bool(sync_cuda)
        self.start = None

    def __enter__(self):
        if self.profile is not None:
            import time

            _sync_cuda_for_profile(self.sync_cuda)
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.profile is not None and self.start is not None:
            import time

            _sync_cuda_for_profile(self.sync_cuda)
            self.profile[self.name] = self.profile.get(self.name, 0.0) + float(time.perf_counter() - self.start)
        return False


def _record_tensor_shape(profile, prefix: str, tensor):
    if profile is None or tensor is None or not hasattr(tensor, "shape"):
        return
    try:
        shape = tuple(int(v) for v in tensor.shape)
        profile[f"{prefix}_rank"] = float(len(shape))
        profile[f"{prefix}_numel"] = float(tensor.numel())
        for idx, size in enumerate(shape):
            profile[f"{prefix}_shape_{idx}"] = float(size)
    except Exception:
        pass


def _record_image_grid_thw(profile, prefix: str, tensor):
    _record_tensor_shape(profile, prefix, tensor)
    if profile is None or tensor is None:
        return
    try:
        rows = tensor.detach().cpu().tolist()
        if rows and isinstance(rows[0], (int, float)):
            rows = [rows]
        profile[f"{prefix}_rows"] = float(len(rows))
        profile[f"{prefix}_grid_tokens"] = float(sum(int(row[0]) * int(row[1]) * int(row[2]) for row in rows))
        for row_idx, row in enumerate(rows[:4]):
            if len(row) < 3:
                continue
            profile[f"{prefix}_{row_idx}_t"] = float(row[0])
            profile[f"{prefix}_{row_idx}_h"] = float(row[1])
            profile[f"{prefix}_{row_idx}_w"] = float(row[2])
    except Exception:
        pass


def _record_vlm_input_profile(profile, prefix: str, inputs):
    if profile is None or inputs is None:
        return
    for name in ("input_ids", "attention_mask", "pixel_values", "inputs_embeds"):
        value = inputs.get(name) if hasattr(inputs, "get") else None
        _record_tensor_shape(profile, f"{prefix}_{name}", value)
    grid = inputs.get("image_grid_thw") if hasattr(inputs, "get") else None
    _record_image_grid_thw(profile, f"{prefix}_image_grid_thw", grid)


def _model_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _unwrap_model(model):
    return getattr(model, "module", model)


def _prepare_base_inputs(processor, model, messages, images=None):
    try:
        chat_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    except Exception:
        chat_text = str(messages[-1].get("content", "")) if messages else ""

    if images is None:
        inputs = processor(text=[chat_text], return_tensors="pt")
    else:
        inputs = processor(text=[chat_text], images=[list(images)], return_tensors="pt")

    device = _model_device(model)
    if device is not None:
        try:
            inputs = inputs.to(device)
        except Exception:
            pass
    return inputs


def _prepend_soft_prompt(model, inputs, soft_prompt):
    import torch

    if soft_prompt.ndim != 3 or soft_prompt.shape[0] != 1:
        raise RuntimeError("soft_prompt must have shape [1, K, d_llm]")

    if "input_ids" not in inputs:
        raise RuntimeError("processor output missing input_ids")
    input_ids = inputs["input_ids"]

    base_model = _unwrap_model(model)
    embed_layer = base_model.get_input_embeddings()
    token_embeds = embed_layer(input_ids)
    soft_prompt = soft_prompt.to(token_embeds.device, dtype=token_embeds.dtype)
    inputs_embeds = torch.cat([soft_prompt, token_embeds], dim=1)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones(
            (input_ids.shape[0], input_ids.shape[1]),
            device=input_ids.device,
            dtype=torch.long,
        )
    soft_mask = torch.ones(
        (attention_mask.shape[0], soft_prompt.shape[1]),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    attention_mask = torch.cat([soft_mask, attention_mask], dim=1)

    model_inputs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
    for k, v in inputs.items():
        if k in ("input_ids", "attention_mask"):
            continue
        model_inputs[k] = v
    return model_inputs, int(soft_prompt.shape[1]), int(input_ids.shape[1])


def forward_action_ce_with_soft_prompt(
    processor,
    model,
    messages,
    images,
    action_text: str,
    soft_prompt,
):
    import torch

    base_inputs = _prepare_base_inputs(processor, model, messages, images=images)
    if "input_ids" not in base_inputs:
        raise RuntimeError("processor output missing input_ids")

    action_ids = processor.tokenizer(
        str(action_text),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    action_ids = action_ids.to(base_inputs["input_ids"].device)
    if action_ids.shape[1] == 0:
        raise RuntimeError(f"Empty action tokenization for action_text={action_text!r}")

    input_ids = torch.cat([base_inputs["input_ids"], action_ids], dim=1)
    extended_inputs = dict(base_inputs)
    extended_inputs["input_ids"] = input_ids
    if "attention_mask" in extended_inputs:
        ext_mask = torch.ones_like(action_ids, dtype=extended_inputs["attention_mask"].dtype)
        extended_inputs["attention_mask"] = torch.cat([extended_inputs["attention_mask"], ext_mask], dim=1)

    model_inputs, soft_len, prompt_len = _prepend_soft_prompt(model, extended_inputs, soft_prompt)
    total_len = model_inputs["inputs_embeds"].shape[1]
    action_len = action_ids.shape[1]

    labels = torch.full(
        (1, total_len),
        fill_value=-100,
        dtype=torch.long,
        device=model_inputs["inputs_embeds"].device,
    )
    target_end = total_len
    target_start = total_len - action_len
    labels[:, target_start:target_end] = action_ids
    model_inputs["labels"] = labels

    outputs = model(**model_inputs)
    return outputs.loss


def generate_action_with_soft_prompt(
    processor,
    model,
    messages,
    images,
    soft_prompt,
    max_new_tokens: int = 16,
    do_sample: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    profile: Optional[Dict[str, float]] = None,
    profile_sync_cuda: bool = False,
):
    import torch

    with _ProfileStage(profile, "vlm_prepare_inputs", profile_sync_cuda):
        base_inputs = _prepare_base_inputs(processor, model, messages, images=images)
    _record_vlm_input_profile(profile, "vlm_base", base_inputs)
    with _ProfileStage(profile, "vlm_prepend_soft_prompt", profile_sync_cuda):
        model_inputs, _, prompt_len = _prepend_soft_prompt(model, base_inputs, soft_prompt)
    _record_vlm_input_profile(profile, "vlm_model", model_inputs)
    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }
    if temperature is not None:
        gen_kwargs["temperature"] = float(temperature)
    if top_p is not None:
        gen_kwargs["top_p"] = float(top_p)
    if top_k is not None:
        gen_kwargs["top_k"] = int(top_k)

    with _ProfileStage(profile, "vlm_generate", profile_sync_cuda):
        with torch.inference_mode():
            out_ids = _unwrap_model(model).generate(**model_inputs, **gen_kwargs)

    if profile is not None:
        try:
            output_tokens = int(out_ids.shape[1])
            new_tokens = output_tokens - int(prompt_len) if output_tokens > int(prompt_len) else output_tokens
            profile["vlm_output_tokens"] = float(output_tokens)
            profile["vlm_prompt_tokens"] = float(prompt_len)
            profile["vlm_new_tokens_est"] = float(max(new_tokens, 0))
        except Exception:
            pass

    with _ProfileStage(profile, "vlm_decode", profile_sync_cuda):
        full_text = processor.batch_decode(out_ids, skip_special_tokens=True)
        if not full_text:
            return ""
        full_text = str(full_text[0]).strip()

    # 使用 inputs_embeds 生成时，部分后端只返回生成的 token；
    # 其他后端会返回提示词加生成 token，需要同时兼容两种路径。
    if out_ids.ndim == 2 and out_ids.shape[1] > prompt_len:
        with _ProfileStage(profile, "vlm_decode_tail", profile_sync_cuda):
            tail_ids = out_ids[:, prompt_len:]
            tail_text = processor.batch_decode(tail_ids, skip_special_tokens=True)
            tail_text = str(tail_text[0]).strip() if tail_text else ""
        if tail_text:
            return tail_text
    return full_text
