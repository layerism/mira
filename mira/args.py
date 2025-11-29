#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from mira import hf_get_or_download


@dataclass
class BaseArgs:
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        return copy.deepcopy(self)

    def to_dict(self, exclude=[]):
        return {k: v for k, v in asdict(self).items() if k not in exclude}


@dataclass
class ReasoningArgs:
    effort: Literal["minimal", "low", "medium", "high"] | None = None
    include: bool | None = False
    budget_tokens: Optional[int] = None


@dataclass
class OpenAIArgs(BaseArgs):
    """compatible with openai 2.8 protocol"""

    model: str = "gpt-5-mini"
    stop: list[str] = field(default_factory=lambda: ["<|im_end|>", "<|endoftext|>", "END", "STOP"])
    max_completion_tokens: int = 8192
    parallel_tool_calls: bool = True
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = None
    temperature: float = 0.7
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.1
    top_p: float = 0.95
    top_k: int = -1
    n: int = 1
    logprobs: bool = False
    top_logprobs: Optional[int] = 5
    stream: bool = False

    # addtional params
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    verbose: bool = False

    def __post_init__(self):
        if self.logprobs:
            self.top_logprobs = self.n


@dataclass
class OpenRouterArgs(BaseArgs):
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-2.5-flash"
    stop: list[str] = field(default_factory=lambda: ["END", "STOP"])
    max_completion_tokens: int = 8192
    reasoning: ReasoningArgs = field(default_factory=ReasoningArgs)
    temperature: float = 0.7
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    top_p: float = 0.95
    top_k: int = -1
    num_workers: int = 1
    verbose: bool = False


@dataclass
class VLLMArgs(BaseArgs):
    """
    EngineArgs rewritten with merged categories.
    All parameters preserved, each with clear English explanation.
    """

    # ==========================================
    # Model & Tokenizer
    # ==========================================
    model: str = "Qwen/Qwen3-1.7B"  # Path or HF repo (user adjustable)
    tokenizer: Optional[str] = None  # Tokenizer path (user adjustable)
    dtype: str = "auto"  # Model dtype, e.g. float16/bfloat16 (user adjustable)
    tokenizer: str = ""  # Tokenizer path (user adjustable)
    trust_remote_code: bool = True

    # ==========================================
    # Runtime / Scheduler / Output
    # ==========================================
    device: str = "cuda"  # CUDA device
    scheduling_policy: str = "fcfs"  # Scheduling (first-come-first-serve)
    multi_step_stream_outputs: bool = True  # Smooth streaming output
    num_scheduler_steps: int = 1  # Scheduler step count
    use_tqdm_on_load: bool = True  # tqdm for loading

    # ==========================================
    # GPU Memory / KV Cache / Speed
    # ==========================================
    kv_cache_dtype: str = "auto"  # KV cache dtype, e.g. float16/bfloat16 (user adjustable)
    quantization: Optional[str] = None  # Quantization type (user adjustable)
    gpu_memory_utilization: float = 0.95  # Use 90% GPU memory (highly recommended)
    block_size: int = 16  # KV block size (rarely changed)
    prefill_chunk_size: Optional[int] = 4096  # Prefill chunk size (user adjustable)
    enable_prefix_caching: bool = True  # Prefix cache for speed (recommended)
    swap_space: int = 4  # CPU swap space (GB) (user adjustable)
    cpu_offload_gb: int = 0  # CPU offload (optional for big models)
    max_model_len: Optional[int] = 1024 * 32  # Max seq len (user adjustable)
    max_num_seqs: Optional[int] = 8  # Max concurrent sequences (user adjustable)
    max_num_batched_tokens: Optional[int] = 32768  # Limit batch (advanced)
    max_num_partial_prefills: int = 1  # Partial prefill (rarely changed)
    max_long_partial_prefills: int = 1  # Long partial prefill (rarely changed)
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 1
    use_cuda_graph: bool = True
    attention_backend: str = "flashattn"
    enforce_eager: bool = False
    kv_sharing_fast_prefill: bool = False
    async_scheduling: bool = False

    # ==========================================
    # Sampling (logprobs/temperature/top-k)
    # ==========================================
    max_logprobs: int = 10  # Logprobs for sampling (user adjustable)
    generation_config: str = "auto"  # HF-style generation config

    # ==========================================
    # Multimodal (image/video/audio)
    # ==========================================
    limit_mm_per_prompt: Dict[str, int] = None  # Limit MM items per prompt
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    disable_mm_preprocessor_cache: bool = False  # Cache MM preprocessors

    # ==========================================
    # Sampling Params / Output Control
    # ==========================================
    seed: int = 42  # RNG seed
    stop: list[str] = field(default_factory=lambda: ["<|im_end|>", "<|endoftext|>"])
    max_tokens: int = 2048
    min_tokens: int = 0
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: int = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    length_penalty: float = 1.0
    logprobs: Optional[int] = None
    top_logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    logit_bias: Optional[Dict[int, float]] = None
    use_beam_search: bool = False
    beam_size: int = 1
    early_stopping: Optional[str] = "never"
    n: int = 1
    ignore_eos: bool = False
    skip_special_tokens: bool = False
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[int] = None
    include_stop_str_in_output: Optional[bool] = False
    clean_up_tokenization_spaces: Optional[bool] = False
    guided_decoding: Optional[dict] = None
    structured_outputs_config: Optional[dict] = field(default_factory=lambda: {"backend": "xgrammar"})

    # server
    host: str = "0.0.0.0"
    port: int = 8090

    def __post_init__(self):
        if not Path(self.model).exists():
            self.tokenizer = self.model
            self.model = hf_get_or_download(self.model)


@dataclass
class HFArgs(BaseArgs):
    model: str = "openai/o4-mini"


if __name__ == "__main__":
    import json

    import tyro

    args = tyro.cli(OpenAIArgs)
    print(json.dumps(args.to_dict(), indent=2, ensure_ascii=False))
