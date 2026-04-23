#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import boto3


BASE_DIR = Path(__file__).resolve().parents[1]
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "deportedata")
SOURCE_ARCHIVE = BASE_DIR / "scratch" / "deportedata-source.tar.gz"
SOURCE_ARCHIVE_KEY = os.environ.get("SOURCE_ARCHIVE_KEY", "releases/current.tar.gz")
DEPLOYMENT_INFO_PATH = BASE_DIR / "aws" / "deployment_info.json"


def run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def package_source() -> Path:
    SOURCE_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(SOURCE_ARCHIVE, "w:gz") as archive:
        for path in sorted(BASE_DIR.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(BASE_DIR)
            if any(
                part in {".git", ".venv", ".vendor", ".deploydeps", "__pycache__", ".pytest_cache", "models"}
                for part in relative.parts
            ):
                continue
            if relative.parts and relative.parts[0] == "scratch":
                continue
            archive.add(path, arcname=str(relative))
    return SOURCE_ARCHIVE


def ensure_bucket_name() -> str:
    session = boto3.Session(region_name=REGION)
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    return os.environ.get("S3_BUCKET", f"{PROJECT_NAME}-{account_id}-{REGION}")


def ensure_bucket_exists(bucket: str) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    try:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
    except Exception:
        pass


def upload_archive(bucket: str, archive_path: Path) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    print(f"Uploading source archive to s3://{bucket}/{SOURCE_ARCHIVE_KEY}")
    s3.upload_file(str(archive_path), bucket, SOURCE_ARCHIVE_KEY)


def wait_for_health(url: str, attempts: int = 60, delay: int = 20) -> bool:
    for index in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if payload.get("models_loaded"):
                    print(f"Health check passed on attempt {index + 1}")
                    return True
                print(f"Attempt {index + 1}: API up but models still loading")
        except Exception as exc:
            print(f"Attempt {index + 1}: waiting for service ({type(exc).__name__})")
        time.sleep(delay)
    return False


def main() -> None:
    print("Packaging source artifact ...")
    archive_path = package_source()

    bucket_name = ensure_bucket_name()
    os.environ["S3_BUCKET"] = bucket_name
    os.environ["SOURCE_ARCHIVE_KEY"] = SOURCE_ARCHIVE_KEY
    ensure_bucket_exists(bucket_name)

    print("Preparing processed data ...")
    run([sys.executable, "scripts/process_data.py"])

    print("Uploading application archive, models and parquet data ...")
    upload_archive(bucket_name, archive_path)
    run([sys.executable, "aws/upload_models_to_s3.py"])

    print("Creating AWS resources ...")
    run([sys.executable, "aws/setup_infrastructure.py"])

    deployment = json.loads(DEPLOYMENT_INFO_PATH.read_text(encoding="utf-8"))
    web_url = deployment.get("web_url")
    health_url = deployment.get("health_url")
    api_url = deployment.get("api_url")

    print(f"Web URL: {web_url}")
    print(f"API URL: {api_url}")
    print(f"Health URL: {health_url}")

    if health_url and not wait_for_health(health_url):
        print("The service is still warming up. Check the health endpoint again in a few minutes.")


if __name__ == "__main__":
    main()
