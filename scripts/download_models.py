from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import HF_QWEN_BASE_REPO, HF_QWEN_REPO, HF_TOXICITY_REPO


MODELS_DIR = ROOT_DIR / "models"


def _download(repo_id: str, local_dir: Path, allow_patterns: list[str] | None = None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )


def download_models() -> None:
    print(f"Downloading toxicity model from {HF_TOXICITY_REPO} ...")
    _download(
        HF_TOXICITY_REPO,
        MODELS_DIR / "antiToxicidad",
        allow_patterns=["toxicity-classifier/*"],
    )

    print(f"Downloading LoRA adapter from {HF_QWEN_REPO} ...")
    _download(
        HF_QWEN_REPO,
        MODELS_DIR / "QwenDeporteData",
        allow_patterns=["qwen2.5-finetuned/checkpoint-1443/*"],
    )

    print(f"Downloading Qwen base model from {HF_QWEN_BASE_REPO} ...")
    _download(HF_QWEN_BASE_REPO, MODELS_DIR / "QwenBase")

    print(f"Models available under {MODELS_DIR}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    download_models()
