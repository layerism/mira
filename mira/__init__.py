import os
import random
import subprocess
import sys
from pathlib import Path

import diskcache as dc
import numpy as np
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
random.seed(42)
np.random.seed(42)

cache_dir = Path(__file__).parent.parent.resolve() / ".cache"
cache = dc.Cache(cache_dir)

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level}</level> | "
    "<blue>{file}</blue> | <magenta>{function}</magenta>:<yellow>{line}</yellow> "
    "<cyan>{message}</cyan>",
    enqueue=True,
)


def hf_get_or_download(
    repo_id: str,
    repo_type: str = "model",
    force_download: bool = False,
    rename: str = None,
) -> None:
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


# Export public API
from mira.args import OpenAIArgs, OpenRouterArgs, VLLMArgs
from mira.openrouter import OpenRouterLLM
from mira.types import (
    AIMessage,
    Function,
    HumanMessage,
    LLMJson,
    LLMTool,
    Message,
    NameSpace,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

__all__ = [
    # Utility functions
    "cache",
    "hf_get_or_download",
    # Types
    "Message",
    "Function",
    "ToolCall",
    "SystemMessage",
    "AIMessage",
    "ToolMessage",
    "HumanMessage",
    "LLMTool",
    "LLMJson",
    "NameSpace",
    # Args
    "OpenAIArgs",
    "VLLMArgs",
    "OpenRouterArgs",
    # LLM classes
    "OpenRouterLLM",
]
