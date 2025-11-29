import os
import subprocess
from pathlib import Path

from loguru import logger


def disable_hf_net():
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def hf_get_or_download(
    repo_id: str,
    repo_type: str = "model",
    force_download: bool = False,
    rename: str = None,
) -> str:
    """
    Download a model or dataset from Hugging Face Hub.

    Args:
        repo_id: Repository ID on Hugging Face Hub (e.g., "Qwen/Qwen2.5-7B-Instruct")
        repo_type: Type of repository ("model" or "dataset")
        force_download: Whether to force re-download even if cached
        rename: Optional rename for the repository

    Returns:
        Local directory path where the repository is cached
    """
    from huggingface_hub import HfFolder, login, snapshot_download
    from transformers import TRANSFORMERS_CACHE

    # download model to ~/.cache/huggingface/hub
    if rename is None:
        rename = repo_id.split("/")[-1]

    env = "MODEL-{}".format(rename.upper())
    logger.info("{} --> Using default HF cache directory".format(env))

    token = HfFolder.get_token()
    if not token:
        login(token=os.environ["HF_TOKEN"])

    model_cache = "/{}s--{}/snapshots".format(repo_type, repo_id.replace("/", "--"))
    repo_default_cache = TRANSFORMERS_CACHE + model_cache
    repo_default_cache = os.path.expanduser(repo_default_cache)

    if force_download:
        subprocess.run(["rm", "-rf", repo_default_cache])

    local_files_only = False
    if Path(repo_default_cache).exists():
        local_files_only = True

    logger.info(f"{repo_type.upper()} Repo Path: {repo_default_cache}")

    local_dir = snapshot_download(repo_id=repo_id, repo_type=repo_type, local_files_only=local_files_only)

    return local_dir
