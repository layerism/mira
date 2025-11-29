import asyncio
import json
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import tyro
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel
from uvicorn.config import Config
from uvicorn.server import Server

from mira.args import VLLMArgs
from mira.inference import AsyncVLLMEngine


# 1. Load model at startup (global engine instance)
# 2. Use vllm async engine for inference
def lifespan_factory(args):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ──── startup ────
        app.state.engine = await asyncio.to_thread(AsyncVLLMEngine, args)

        yield

        # ──── shutdown ────
        await asyncio.to_thread(app.state.engine.shutdown)

    return lifespan


args = tyro.cli(VLLMArgs)
app = FastAPI(lifespan=lifespan_factory(args))
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"  # Allow all origins in development, should be restricted in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,  # Cache time for preflight request results (seconds)
)


class OpenaiChatRequest(BaseModel):
    model: str = ""
    messages: List[Any]
    tools: Optional[List[Any]] = None
    tool_choice: Optional[str] = "auto"
    response_format: Optional[Any] = None

    # sampling params
    parallel_tool_calls: bool = True
    reasoning_effort: Optional[str] = None
    model: str = "vllm/custom"
    temperature: Optional[float] = 0.7
    repetition_penalty: Optional[float] = 1.0
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    max_completion_tokens: Optional[int] = 8192
    stop: Optional[list[str]] = ["<|im_end|>", "<|endoftext|>"]
    seed: Optional[int] = 42
    logprobs: bool = False
    top_logprobs: Optional[int] = 5
    logit_bias: Optional[dict] = None
    n: Optional[int] = 1
    stream: bool = False


# 使用量查询接口
@app.get("/v1/usage")
async def get_usage():
    return {
        "total_usage": 100  # 示例数据
    }


# Subscription information interface
@app.get("/v1/subscription")
async def get_subscription():
    return {
        "hard_limit_usd": 100.0  # 示例数据
    }


# chat completions interface
@app.post("/v1/chat/completions")
async def chat_completions(request: OpenaiChatRequest):
    engine: AsyncVLLMEngine = app.state.engine

    logger.info(json.dumps(request.model_dump(), indent=2, ensure_ascii=False))

    if engine.tokenizer is None:
        engine.tokenizer = await engine.engine.get_tokenizer()

    prompt = engine.apply_chat_template(
        messages=request.messages,
        tools=request.tools,
        tool_choice=request.tool_choice,
        reasoning_effort=request.reasoning_effort,
    )

    request_id = await engine.add_request(
        prompt=prompt,
        response_format=request.response_format,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_completion_tokens,
        stop=request.stop,
        seed=request.seed,
        n=request.n,
        logprobs=request.top_logprobs if request.logprobs else None,
    )

    if request.stream:
        return StreamingResponse(
            stream_chat_response(engine, request_id),
            media_type="text/event-stream",
        )
    else:
        completion = await chat_response(engine, request_id)
        return completion


# chat response generator
async def chat_response(engine: AsyncVLLMEngine, request_id: str):
    created_time = int(time.time() * 1000)
    try:
        choices = []
        async for output in engine.completion(request_id):
            choices.append(
                {
                    "index": output.index,
                    "message": {
                        "role": "assistant",
                        "content": output.text,
                        "logprobs": output.logprobs,
                        "token_ids": output.token_ids,
                        "tool_calls": output.tool_calls,
                    },
                    "finish_reason": output.finish_reason,
                }
            )

        completion = {
            "id": request_id,
            "model": "vllm/custom",
            "object": "chat.completion",
            "created": created_time,
            "choices": choices,
        }
        logger.info(completion)
        return completion
    except Exception:
        logger.error(traceback.format_exc())


# streaming chat response generator
async def stream_chat_response(engine: AsyncVLLMEngine, request_id: str):
    created_time = int(time.time() * 1000)
    try:
        # send start event
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": "vllm/custom",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield "data: " + json.dumps(chunk) + "\n\n"

        # loop all delta
        async for delta in engine.delta(request_id):
            logger.info(delta)
            if delta:
                # make sure each chunk is correct SSE format
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "choices": [
                        {
                            "index": delta.index,
                            "delta": {
                                "role": "assistant",
                                "content": delta.text,
                                "logprobs": delta.logprobs,
                                "token_ids": delta.token_ids,
                                "tool_calls": delta.tool_calls,
                            },
                            "finish_reason": delta.finish_reason,
                        }
                    ],
                }
                yield "data: " + json.dumps(chunk) + "\n\n"

        # final event
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(traceback.format_exc())
        error_data = {"error": str(e)}
        yield "data: " + json.dumps(error_data) + "\n\n"
        yield "data: [DONE]\n\n"


def main():
    http = Config(app, host=args.host, port=args.port)
    Server(config=http).run()


if __name__ == "__main__":
    main()
