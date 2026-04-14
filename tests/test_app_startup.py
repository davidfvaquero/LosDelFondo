from __future__ import annotations

import os
import subprocess
import sys
import time


def test_streamlit_app_starts_without_import_errors() -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "dashboard/app.py",
            "--server.headless",
            "true",
            "--server.port",
            "8765",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    output_lines: list[str] = []
    started = False
    deadline = time.time() + 25

    try:
        while time.time() < deadline:
            line = process.stdout.readline()
            if line:
                output_lines.append(line)
                if "ModuleNotFoundError" in line or "Traceback" in line:
                    raise AssertionError("".join(output_lines))
                if "Local URL:" in line or "Network URL:" in line:
                    started = True
                    break

            if process.poll() is not None:
                break

        if not started:
            process.poll()
            raise AssertionError(
                "Streamlit app did not start successfully.\n" + "".join(output_lines)
            )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
