import os
from pathlib import Path


def load_hf_token_from_env_file(repo_root: Path):
    env_path = Path(repo_root) / ".env"
    token = os.getenv("HF_TOKEN")
    if token:
        return token.strip()
    if not env_path.exists():
        return None

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            if key.strip() != "HF_TOKEN":
                continue
            token = value.strip().strip('"').strip("'")
            if token:
                os.environ["HF_TOKEN"] = token
                return token
    return None
