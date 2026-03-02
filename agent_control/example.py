import argparse
import time
from pathlib import Path

from qwen3vl_wrapper import Qwen3VLWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=str(Path(__file__).resolve().parent / "models" / "qwen3_vl_sft_merged"),
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    model = Qwen3VLWrapper(args.model_dir).load()

    messages = []
    print("输入 /exit 退出，/reset 清空上下文。")

    while True:
        try:
            user_text = input("user> ")
        except EOFError:
            break

        if user_text is None:
            continue
        user_text = user_text.strip()
        if not user_text:
            continue

        if user_text == "/exit":
            break
        if user_text == "/reset":
            messages = []
            print("已清空上下文")
            continue

        messages.append({"role": "user", "content": user_text})

        t0 = time.perf_counter()
        out = model.generate_chat(
            messages=messages,
            images=None,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.do_sample),
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        dt = time.perf_counter() - t0

        print(f"assistant> {out}")
        print(f"time_s> {dt:.3f}")

        messages.append({"role": "assistant", "content": out})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

