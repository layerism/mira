import random
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


# Export public API - import after initialization to avoid circular imports
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
from mira.utils import hf_get_or_download

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
