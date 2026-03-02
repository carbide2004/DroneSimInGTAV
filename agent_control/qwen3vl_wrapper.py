import os


class Qwen3VLWrapper:
    def __init__(
        self,
        model_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ):
        self.model_dir = os.fspath(model_dir)
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self._processor = None
        self._model = None
        self._base_generation_config = None

    def load(self):
        try:
            from transformers import AutoProcessor
        except Exception as e:
            raise RuntimeError(
                "无法导入 transformers。请确认已安装与 Qwen3VL 兼容的 transformers 版本。"
            ) from e

        self._processor = AutoProcessor.from_pretrained(
            self.model_dir, trust_remote_code=self.trust_remote_code
        )

        last_err = None
        for loader in (_load_explicit_qwen3vl, _load_auto_vision2seq, _load_auto_model):
            try:
                self._model = loader(
                    self.model_dir,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                    trust_remote_code=self.trust_remote_code,
                )
                break
            except Exception as e:
                last_err = e
                self._model = None

        if self._model is None:
            raise RuntimeError(
                f"模型加载失败：{self.model_dir}. 最后一次错误：{last_err}"
            ) from last_err

        try:
            self._model.eval()
        except Exception:
            pass

        try:
            from transformers import GenerationConfig

            self._base_generation_config = GenerationConfig.from_model_config(
                self._model.config
            )
            try:
                self._model.generation_config = self._base_generation_config
            except Exception:
                pass
        except Exception:
            self._base_generation_config = None

        return self

    @property
    def processor(self):
        if self._processor is None:
            raise RuntimeError("processor 尚未加载，请先调用 load()")
        return self._processor

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("model 尚未加载，请先调用 load()")
        return self._model

    def generate_action(
        self,
        prompt_text,
        rgb_pil,
        depth_pil,
        max_new_tokens=16,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": str(prompt_text)},
                    {"type": "image"},
                    {"type": "image"},
                ],
            }
        ]
        return self.generate_chat(
            messages=messages,
            images=[rgb_pil, depth_pil],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    def generate_chat(
        self,
        messages,
        images=None,
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    ):
        try:
            import torch
        except Exception as e:
            raise RuntimeError("无法导入 torch，请确认已安装 PyTorch。") from e

        try:
            chat_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
        except Exception:
            chat_text = str(messages[-1].get("content", "")) if messages else ""

        if images is None:
            inputs = self.processor(text=[chat_text], return_tensors="pt")
        else:
            inputs = self.processor(
                text=[chat_text],
                images=[list(images)],
                return_tensors="pt",
            )

        target_device = None
        try:
            target_device = next(self.model.parameters()).device
        except Exception:
            target_device = None

        if target_device is not None:
            try:
                inputs = inputs.to(target_device)
            except Exception:
                pass

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

        with torch.inference_mode():
            generation_config = None
            if self._base_generation_config is not None:
                try:
                    from transformers import GenerationConfig

                    generation_config = GenerationConfig.from_dict(
                        self._base_generation_config.to_dict()
                    )
                except Exception:
                    generation_config = self._base_generation_config

            if generation_config is not None:
                out_ids = self.model.generate(
                    **inputs, generation_config=generation_config, **gen_kwargs
                )
            else:
                out_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = None
        if isinstance(inputs, dict) and "input_ids" in inputs:
            prompt_len = int(inputs["input_ids"].shape[1])

        if prompt_len is not None:
            out_ids = out_ids[:, prompt_len:]

        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)
        if not text:
            return ""
        return str(text[0]).strip()


def _load_explicit_qwen3vl(model_dir, torch_dtype, device_map, trust_remote_code):
    from transformers import Qwen3VLForConditionalGeneration

    kwargs = {"trust_remote_code": trust_remote_code}
    if device_map is not None:
        kwargs["device_map"] = device_map
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return Qwen3VLForConditionalGeneration.from_pretrained(model_dir, **kwargs)


def _load_auto_vision2seq(model_dir, torch_dtype, device_map, trust_remote_code):
    from transformers import AutoModelForVision2Seq

    kwargs = {"trust_remote_code": trust_remote_code}
    if device_map is not None:
        kwargs["device_map"] = device_map
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return AutoModelForVision2Seq.from_pretrained(model_dir, **kwargs)


def _load_auto_model(model_dir, torch_dtype, device_map, trust_remote_code):
    from transformers import AutoModel

    kwargs = {"trust_remote_code": trust_remote_code}
    if device_map is not None:
        kwargs["device_map"] = device_map
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    model = AutoModel.from_pretrained(model_dir, **kwargs)
    if not hasattr(model, "generate"):
        raise RuntimeError("AutoModel 加载结果不支持 generate()")
    return model
