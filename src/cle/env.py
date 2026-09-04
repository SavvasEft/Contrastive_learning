import platform
import subprocess


def describe_env() -> dict:
    import torch

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha, dirty = None, None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "git_sha": sha,
        "git_dirty": dirty,
    }