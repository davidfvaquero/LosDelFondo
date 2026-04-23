#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import boto3


S3_BUCKET = os.environ["S3_BUCKET"]
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
BASE_DIR = Path(__file__).resolve().parents[1]

s3 = boto3.client("s3", region_name=REGION)


def upload_tree(local_dir: Path, s3_prefix: str) -> int:
    if not local_dir.exists():
        print(f"[WARN] Skipping missing path: {local_dir}")
        return 0

    total = 0
    for local_path in sorted(local_dir.rglob("*")):
        if not local_path.is_file():
            continue
        if any(part in {"__pycache__", ".git", ".pytest_cache"} for part in local_path.parts):
            continue
        relative = local_path.relative_to(local_dir).as_posix()
        s3_key = f"{s3_prefix}/{relative}" if relative else s3_prefix
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"[UPLOAD] {local_path.relative_to(BASE_DIR)} -> s3://{S3_BUCKET}/{s3_key} ({size_mb:.1f} MB)")
        s3.upload_file(str(local_path), S3_BUCKET, s3_key)
        total += 1
    return total


def main() -> None:
    print(f"Uploading project artifacts to s3://{S3_BUCKET} in {REGION}")

    uploaded = 0
    uploaded += upload_tree(BASE_DIR / "data" / "processed", "data/processed")
    uploaded += upload_tree(BASE_DIR / "models" / "antiToxicidad", "models/antiToxicidad")
    uploaded += upload_tree(BASE_DIR / "models" / "QwenBase", "models/QwenBase")
    uploaded += upload_tree(BASE_DIR / "models" / "QwenDeporteData", "models/QwenDeporteData")

    print(f"Done. Uploaded {uploaded} files.")


if __name__ == "__main__":
    main()
