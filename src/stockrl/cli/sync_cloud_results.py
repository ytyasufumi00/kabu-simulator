from __future__ import annotations

import subprocess
import sys

from ..config import load_settings

GCS_BUCKET = "gs://kabu-simulator-runs-57608233882"


def main() -> None:
    settings = load_settings()
    runs_dir = settings.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"{GCS_BUCKET}/runs/ から {runs_dir} へ同期します...")
    # Windowsではgsutilがバッチファイルラッパーのため、shell経由で起動する必要がある
    subprocess.run(
        ["gsutil", "-m", "rsync", "-r", f"{GCS_BUCKET}/runs/", str(runs_dir)],
        check=True,
        shell=(sys.platform == "win32"),
    )
    print("完了")


if __name__ == "__main__":
    main()
